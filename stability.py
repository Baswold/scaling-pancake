"""
Stability Mechanisms for Fluid Weight Learning

This module implements various constraints and mechanisms to prevent
weight divergence, collapse, or oscillation during continuous adaptation.

Key Challenges:
==============
1. Weights growing unbounded (explosion)
2. Weights collapsing to zero (vanishing)
3. Chaotic oscillations
4. Catastrophic forgetting of base knowledge
5. Drift into degenerate solutions

Solutions Implemented:
=====================
1. Elastic Weight Consolidation (EWC) - protect important weights
2. Spectral Normalization - bound weight norms
3. Weight Decay to Origin - attraction to initial values
4. Gradient Clipping - limit update magnitudes
5. Adaptive Rate Control - dynamic learning rate based on stability metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class StabilityMetrics:
    """Metrics tracking stability of the learning process."""
    weight_norm: float = 0.0
    weight_change: float = 0.0
    spectral_norm: float = 0.0
    gradient_norm: float = 0.0
    fisher_importance: float = 0.0
    drift_from_origin: float = 0.0
    oscillation_score: float = 0.0


class StabilityMechanism(ABC):
    """
    Abstract base class for stability mechanisms.

    Stability mechanisms can:
    1. Modify proposed updates (constrain_update)
    2. Apply post-update corrections (post_update)
    3. Monitor for instability (check_stability)
    """

    @abstractmethod
    def constrain_update(
        self,
        delta_A: torch.Tensor,
        delta_B: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constrain the proposed update to maintain stability.

        Args:
            delta_A, delta_B: Proposed updates
            A, B: Current weights

        Returns:
            Constrained (delta_A, delta_B)
        """
        pass

    def post_update(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optional: Apply corrections after update."""
        return A, B

    def check_stability(self, A: torch.Tensor, B: torch.Tensor) -> StabilityMetrics:
        """Check current stability metrics."""
        return StabilityMetrics(
            weight_norm=torch.norm(A).item() + torch.norm(B).item()
        )


class ElasticWeightConsolidation(StabilityMechanism):
    """
    Elastic Weight Consolidation (EWC) - Protect important weights

    Mathematical Formulation:
    ========================

    EWC adds a penalty for changing weights that are "important" for
    previously learned tasks:

    L_ewc = Σ_i F_i * (θ_i - θ*_i)²

    Where:
    - F_i: Fisher information (importance) of weight i
    - θ_i: Current weight value
    - θ*_i: Original weight value

    For fluid weights, we apply this as a constraint on updates:

    Δθ_constrained = Δθ - λ * F * (θ - θ*)

    The second term pulls weights back toward their original values,
    with strength proportional to importance F.

    Fisher Estimation:
    - Use running estimate from activation magnitudes
    - High activation → high importance
    - Or: Use gradient magnitudes when available
    """

    def __init__(
        self,
        importance_decay: float = 0.999,
        consolidation_strength: float = 0.1,
        use_diagonal_fisher: bool = True,
    ):
        self.importance_decay = importance_decay
        self.consolidation_strength = consolidation_strength
        self.use_diagonal_fisher = use_diagonal_fisher

        # Store original weights and Fisher estimates
        self.A_original: Optional[torch.Tensor] = None
        self.B_original: Optional[torch.Tensor] = None
        self.F_A: Optional[torch.Tensor] = None
        self.F_B: Optional[torch.Tensor] = None

    def initialize(self, A: torch.Tensor, B: torch.Tensor):
        """Store original weights for consolidation."""
        self.A_original = A.detach().clone()
        self.B_original = B.detach().clone()
        self.F_A = torch.zeros_like(A)
        self.F_B = torch.zeros_like(B)

    def update_fisher(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        h: torch.Tensor,
        grad_A: Optional[torch.Tensor] = None,
        grad_B: Optional[torch.Tensor] = None,
    ):
        """Update Fisher information estimate."""
        if self.F_A is None:
            self.initialize(A, B)

        if grad_A is not None and grad_B is not None:
            # Use gradient-based Fisher estimate
            new_F_A = grad_A ** 2
            new_F_B = grad_B ** 2
        else:
            # Use activation-based proxy for Fisher
            # Weights connected to high-activation units are important
            h_importance = torch.mean(h.reshape(-1, h.shape[-1]) ** 2, dim=0)

            # F_A: importance based on hidden unit importance
            new_F_A = torch.outer(torch.ones(A.shape[0], device=A.device), h_importance)

            # F_B: importance based on hidden unit importance
            new_F_B = torch.outer(h_importance, torch.ones(B.shape[1], device=B.device))

        # Exponential moving average
        self.F_A = self.importance_decay * self.F_A + (1 - self.importance_decay) * new_F_A
        self.F_B = self.importance_decay * self.F_B + (1 - self.importance_decay) * new_F_B

    def constrain_update(
        self,
        delta_A: torch.Tensor,
        delta_B: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.A_original is None:
            self.initialize(A, B)

        # Add EWC penalty to pull toward original weights
        ewc_A = self.consolidation_strength * self.F_A * (A - self.A_original)
        ewc_B = self.consolidation_strength * self.F_B * (B - self.B_original)

        # Constrain update by subtracting EWC gradient
        delta_A_constrained = delta_A - ewc_A
        delta_B_constrained = delta_B - ewc_B

        return delta_A_constrained, delta_B_constrained

    def check_stability(self, A: torch.Tensor, B: torch.Tensor) -> StabilityMetrics:
        if self.A_original is None:
            return StabilityMetrics()

        drift_A = torch.norm(A - self.A_original).item()
        drift_B = torch.norm(B - self.B_original).item()

        return StabilityMetrics(
            drift_from_origin=drift_A + drift_B,
            fisher_importance=torch.mean(self.F_A).item() + torch.mean(self.F_B).item()
            if self.F_A is not None else 0.0
        )


class SpectralNormConstraint(StabilityMechanism):
    """
    Spectral Normalization - Keep spectral norm bounded

    Mathematical Formulation:
    ========================

    The spectral norm of a matrix is its largest singular value:
    σ(W) = max_x ||Wx|| / ||x||

    We constrain updates so that:
    σ(W + ΔW) ≤ max_spectral_norm

    Implementation:
    - Compute current spectral norm
    - If update would exceed bound, scale it down
    - Use power iteration for efficient spectral norm estimation
    """

    def __init__(
        self,
        max_spectral_norm: float = 2.0,
        n_power_iterations: int = 1,
    ):
        self.max_spectral_norm = max_spectral_norm
        self.n_power_iterations = n_power_iterations
        # Cached vectors for power iteration
        self.u_A: Optional[torch.Tensor] = None
        self.v_A: Optional[torch.Tensor] = None
        self.u_B: Optional[torch.Tensor] = None
        self.v_B: Optional[torch.Tensor] = None

    def _spectral_norm(self, W: torch.Tensor, u: Optional[torch.Tensor] = None) -> Tuple[float, torch.Tensor]:
        """Estimate spectral norm using power iteration."""
        if u is None:
            u = torch.randn(W.shape[1], device=W.device)
            u = u / torch.norm(u)

        for _ in range(self.n_power_iterations):
            v = W @ u
            v = v / (torch.norm(v) + 1e-8)
            u = W.T @ v
            u = u / (torch.norm(u) + 1e-8)

        sigma = torch.norm(W @ u).item()
        return sigma, u

    def constrain_update(
        self,
        delta_A: torch.Tensor,
        delta_B: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute spectral norm of proposed new weights
        A_new = A + delta_A
        B_new = B + delta_B

        sigma_A, self.u_A = self._spectral_norm(A_new, self.u_A)
        sigma_B, self.u_B = self._spectral_norm(B_new, self.u_B)

        # Scale updates if they would exceed bounds
        if sigma_A > self.max_spectral_norm:
            scale_A = self.max_spectral_norm / sigma_A
            delta_A = delta_A * scale_A

        if sigma_B > self.max_spectral_norm:
            scale_B = self.max_spectral_norm / sigma_B
            delta_B = delta_B * scale_B

        return delta_A, delta_B

    def check_stability(self, A: torch.Tensor, B: torch.Tensor) -> StabilityMetrics:
        sigma_A, _ = self._spectral_norm(A)
        sigma_B, _ = self._spectral_norm(B)
        return StabilityMetrics(spectral_norm=sigma_A + sigma_B)


class WeightDecayToOrigin(StabilityMechanism):
    """
    Weight Decay to Origin - Gentle attraction to initial weights

    Mathematical Formulation:
    ========================

    Simple L2 penalty pulling weights toward their initial values:

    Δθ_regularized = Δθ - λ * (θ - θ_0)

    This provides a "restoring force" that:
    - Prevents unbounded drift
    - Keeps weights near initialization
    - Strength λ controls plasticity/stability tradeoff

    Unlike EWC, this doesn't use importance weighting - all weights
    are treated equally. This is simpler but may harm important weights.
    """

    def __init__(self, decay_rate: float = 0.001):
        self.decay_rate = decay_rate
        self.A_original: Optional[torch.Tensor] = None
        self.B_original: Optional[torch.Tensor] = None

    def initialize(self, A: torch.Tensor, B: torch.Tensor):
        """Store original weights."""
        self.A_original = A.detach().clone()
        self.B_original = B.detach().clone()

    def constrain_update(
        self,
        delta_A: torch.Tensor,
        delta_B: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.A_original is None:
            self.initialize(A, B)

        # Add decay term pulling toward origin
        decay_A = self.decay_rate * (A - self.A_original)
        decay_B = self.decay_rate * (B - self.B_original)

        return delta_A - decay_A, delta_B - decay_B

    def check_stability(self, A: torch.Tensor, B: torch.Tensor) -> StabilityMetrics:
        if self.A_original is None:
            return StabilityMetrics()
        drift = torch.norm(A - self.A_original).item() + torch.norm(B - self.B_original).item()
        return StabilityMetrics(drift_from_origin=drift)


class GradientClipping(StabilityMechanism):
    """
    Gradient Clipping - Limit update magnitudes

    Mathematical Formulation:
    ========================

    Clip updates to a maximum norm:

    if ||Δθ|| > max_norm:
        Δθ = Δθ * max_norm / ||Δθ||

    This prevents any single update from being too large, which:
    - Prevents sudden jumps in weight space
    - Smooths the learning trajectory
    - Makes learning more stable
    """

    def __init__(self, max_norm: float = 0.1, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def constrain_update(
        self,
        delta_A: torch.Tensor,
        delta_B: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute combined norm
        total_norm = (
            torch.norm(delta_A, p=self.norm_type) ** self.norm_type +
            torch.norm(delta_B, p=self.norm_type) ** self.norm_type
        ) ** (1.0 / self.norm_type)

        # Clip if necessary
        if total_norm > self.max_norm:
            scale = self.max_norm / (total_norm + 1e-8)
            delta_A = delta_A * scale
            delta_B = delta_B * scale

        return delta_A, delta_B

    def check_stability(self, A: torch.Tensor, B: torch.Tensor) -> StabilityMetrics:
        return StabilityMetrics(weight_norm=torch.norm(A).item() + torch.norm(B).item())


class AdaptiveRateControl(StabilityMechanism):
    """
    Adaptive Rate Control - Dynamically adjust learning rate

    Mathematical Formulation:
    ========================

    Monitor stability metrics and adjust learning rate:

    stability_score = f(weight_change, oscillation, drift)

    if stability_score > threshold:
        effective_lr *= decay_factor
    elif stability_score < target:
        effective_lr *= increase_factor

    This provides automatic stability:
    - If learning becomes unstable, slow down
    - If learning is very stable, can speed up
    """

    def __init__(
        self,
        target_stability: float = 0.1,
        decay_factor: float = 0.9,
        increase_factor: float = 1.01,
        min_rate: float = 0.001,
        max_rate: float = 1.0,
        window_size: int = 100,
    ):
        self.target_stability = target_stability
        self.decay_factor = decay_factor
        self.increase_factor = increase_factor
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.window_size = window_size

        self.effective_rate = 1.0
        self.change_history: List[float] = []
        self.prev_A: Optional[torch.Tensor] = None
        self.prev_B: Optional[torch.Tensor] = None

    def constrain_update(
        self,
        delta_A: torch.Tensor,
        delta_B: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute change magnitude
        change_mag = torch.norm(delta_A).item() + torch.norm(delta_B).item()

        # Detect oscillation by checking if direction reversed
        if self.prev_A is not None:
            prev_delta_A = A - self.prev_A
            correlation = torch.sum(delta_A * prev_delta_A).item()
            if correlation < 0:  # Direction reversed = oscillation
                change_mag *= 2  # Penalize oscillation

        self.change_history.append(change_mag)
        if len(self.change_history) > self.window_size:
            self.change_history.pop(0)

        # Compute stability score (lower = more stable)
        stability_score = sum(self.change_history) / len(self.change_history)

        # Adjust effective rate
        if stability_score > self.target_stability:
            self.effective_rate *= self.decay_factor
        else:
            self.effective_rate *= self.increase_factor

        # Clamp to bounds
        self.effective_rate = max(self.min_rate, min(self.max_rate, self.effective_rate))

        # Store for next iteration
        self.prev_A = A.detach().clone()
        self.prev_B = B.detach().clone()

        # Apply effective rate
        return delta_A * self.effective_rate, delta_B * self.effective_rate

    def check_stability(self, A: torch.Tensor, B: torch.Tensor) -> StabilityMetrics:
        oscillation = 0.0
        if self.prev_A is not None:
            delta = A - self.prev_A
            oscillation = torch.norm(delta).item()

        return StabilityMetrics(
            oscillation_score=oscillation,
            weight_change=self.effective_rate
        )


class CompositeStability(StabilityMechanism):
    """
    Combine multiple stability mechanisms.

    Usage:
        stability = CompositeStability([
            ElasticWeightConsolidation(strength=0.1),
            SpectralNormConstraint(max_norm=2.0),
            GradientClipping(max_norm=0.1),
            AdaptiveRateControl(),
        ])
    """

    def __init__(self, mechanisms: List[StabilityMechanism]):
        self.mechanisms = mechanisms

    def constrain_update(
        self,
        delta_A: torch.Tensor,
        delta_B: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for mechanism in self.mechanisms:
            delta_A, delta_B = mechanism.constrain_update(delta_A, delta_B, A, B)
        return delta_A, delta_B

    def post_update(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for mechanism in self.mechanisms:
            A, B = mechanism.post_update(A, B)
        return A, B

    def check_stability(self, A: torch.Tensor, B: torch.Tensor) -> StabilityMetrics:
        # Aggregate metrics from all mechanisms
        metrics = StabilityMetrics()
        for mechanism in self.mechanisms:
            m = mechanism.check_stability(A, B)
            metrics.weight_norm = max(metrics.weight_norm, m.weight_norm)
            metrics.spectral_norm = max(metrics.spectral_norm, m.spectral_norm)
            metrics.drift_from_origin = max(metrics.drift_from_origin, m.drift_from_origin)
            metrics.oscillation_score = max(metrics.oscillation_score, m.oscillation_score)
        return metrics


def create_default_stability() -> CompositeStability:
    """Create a recommended stability configuration."""
    return CompositeStability([
        ElasticWeightConsolidation(
            consolidation_strength=0.05,
            importance_decay=0.999,
        ),
        SpectralNormConstraint(max_spectral_norm=2.0),
        GradientClipping(max_norm=0.1),
        AdaptiveRateControl(
            target_stability=0.05,
            decay_factor=0.95,
        ),
    ])
