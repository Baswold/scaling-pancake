"""
Update Rules for Fluid Weight Learning

This module implements various local learning rules that can update weights
during inference without global backpropagation.

Mathematical Framework:
=======================

Given a LoRA layer: y = x @ W_base + x @ A @ B

Where:
- x: input activations [batch, seq, d_in]
- A: LoRA down projection [d_in, rank]
- B: LoRA up projection [rank, d_out]
- y: output activations [batch, seq, d_out]

We want update rules ΔA, ΔB that:
1. Adapt based on activation patterns (no explicit loss)
2. Remain stable over long sequences
3. Preserve learned knowledge

Key Insight: We can derive updates from local information only:
- Pre-activation: x
- Post-activation: y
- Intermediate: h = x @ A (the low-rank representation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class UpdateContext:
    """Context passed to update rules containing activation information."""
    x: torch.Tensor           # Input activations [batch, seq, d_in]
    h: torch.Tensor           # Intermediate (low-rank) [batch, seq, rank]
    y: torch.Tensor           # Output activations [batch, seq, d_out]
    A: torch.Tensor           # Current A matrix [d_in, rank]
    B: torch.Tensor           # Current B matrix [rank, d_out]
    attention_weights: Optional[torch.Tensor] = None  # If available
    layer_idx: int = 0
    step: int = 0


class UpdateRule(ABC):
    """
    Abstract base class for fluid weight update rules.

    All update rules must implement compute_update() which returns
    the proposed changes to A and B matrices.
    """

    def __init__(self, learning_rate: float = 1e-5, **kwargs):
        self.learning_rate = learning_rate
        self.config = kwargs

    @abstractmethod
    def compute_update(self, ctx: UpdateContext) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute updates for A and B matrices.

        Args:
            ctx: UpdateContext containing activations and current weights

        Returns:
            (delta_A, delta_B): Proposed updates to A and B matrices
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.learning_rate})"


class HebbianUpdate(UpdateRule):
    """
    Classical Hebbian Learning: "Neurons that fire together, wire together"

    Mathematical Formulation:
    ========================

    For LoRA matrices A [d_in, rank] and B [rank, d_out]:

    ΔA = η * (x^T @ h) / (||x|| * ||h||)   # Correlation of input with hidden
    ΔB = η * (h^T @ y) / (||h|| * ||y||)   # Correlation of hidden with output

    Where normalization prevents unbounded growth.

    Intuition:
    - If input x and hidden h are correlated, strengthen the connection
    - If hidden h and output y are correlated, strengthen that connection
    - This is "local" - only uses information available at this layer

    Stability:
    - Normalization by norms prevents weight explosion
    - Optional: Add weight decay term for contraction
    """

    def __init__(
        self,
        learning_rate: float = 1e-5,
        normalize: bool = True,
        correlation_threshold: float = 0.0,
        **kwargs
    ):
        super().__init__(learning_rate, **kwargs)
        self.normalize = normalize
        self.correlation_threshold = correlation_threshold

    def compute_update(self, ctx: UpdateContext) -> Tuple[torch.Tensor, torch.Tensor]:
        x, h, y = ctx.x, ctx.h, ctx.y

        # Flatten batch and sequence dimensions
        x_flat = x.reshape(-1, x.shape[-1])  # [N, d_in]
        h_flat = h.reshape(-1, h.shape[-1])  # [N, rank]
        y_flat = y.reshape(-1, y.shape[-1])  # [N, d_out]

        # Compute correlations
        # ΔA: How should we change mapping from input to hidden?
        delta_A = x_flat.T @ h_flat  # [d_in, rank]

        # ΔB: How should we change mapping from hidden to output?
        delta_B = h_flat.T @ y_flat  # [rank, d_out]

        if self.normalize:
            # Normalize by product of norms to keep updates bounded
            x_norm = torch.norm(x_flat) + 1e-8
            h_norm = torch.norm(h_flat) + 1e-8
            y_norm = torch.norm(y_flat) + 1e-8

            delta_A = delta_A / (x_norm * h_norm)
            delta_B = delta_B / (h_norm * y_norm)

        # Optional: Only update if correlation exceeds threshold
        if self.correlation_threshold > 0:
            mask_A = torch.abs(delta_A) > self.correlation_threshold
            mask_B = torch.abs(delta_B) > self.correlation_threshold
            delta_A = delta_A * mask_A
            delta_B = delta_B * mask_B

        return self.learning_rate * delta_A, self.learning_rate * delta_B


class OjaUpdate(UpdateRule):
    """
    Oja's Rule: Self-normalizing Hebbian learning

    Mathematical Formulation:
    ========================

    Oja's rule adds a self-normalizing term that keeps weight norms bounded:

    ΔA = η * x^T @ (h - A^T @ x @ A)
       = η * (x^T @ h - ||h||² * A)

    The term -||h||² * A acts as a stabilizer:
    - When h is large (high activation), it contracts A
    - This prevents unbounded weight growth
    - Converges to principal components of data

    For B matrix:
    ΔB = η * (h^T @ y - ||y||² * B)

    Properties:
    - Self-normalizing: weights don't explode
    - Learns principal components
    - Biologically plausible (local information only)
    """

    def __init__(
        self,
        learning_rate: float = 1e-5,
        stabilization_strength: float = 1.0,
        **kwargs
    ):
        super().__init__(learning_rate, **kwargs)
        self.stabilization_strength = stabilization_strength

    def compute_update(self, ctx: UpdateContext) -> Tuple[torch.Tensor, torch.Tensor]:
        x, h, y, A, B = ctx.x, ctx.h, ctx.y, ctx.A, ctx.B

        # Flatten
        x_flat = x.reshape(-1, x.shape[-1])
        h_flat = h.reshape(-1, h.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])

        # Compute Hebbian term
        hebbian_A = x_flat.T @ h_flat  # [d_in, rank]
        hebbian_B = h_flat.T @ y_flat  # [rank, d_out]

        # Compute stabilization term (Oja's normalization)
        h_norm_sq = torch.mean(torch.sum(h_flat ** 2, dim=-1))  # Scalar
        y_norm_sq = torch.mean(torch.sum(y_flat ** 2, dim=-1))  # Scalar

        stabilize_A = h_norm_sq * A
        stabilize_B = y_norm_sq * B

        # Oja's update
        delta_A = hebbian_A / (x_flat.shape[0] + 1e-8) - self.stabilization_strength * stabilize_A
        delta_B = hebbian_B / (h_flat.shape[0] + 1e-8) - self.stabilization_strength * stabilize_B

        return self.learning_rate * delta_A, self.learning_rate * delta_B


class BCMUpdate(UpdateRule):
    """
    BCM (Bienenstock-Cooper-Munro) Rule: Sliding threshold plasticity

    Mathematical Formulation:
    ========================

    BCM introduces a sliding threshold θ that determines whether connections
    are strengthened (LTP) or weakened (LTD):

    ΔW = η * x * y * (y - θ)

    Where θ is a running average of y²:
    θ = E[y²]

    Properties:
    - y > θ: Potentiation (strengthen)
    - y < θ: Depression (weaken)
    - Self-stabilizing: high activity raises threshold, preventing runaway

    For LoRA:
    ΔA = η * x^T @ (h * (h - θ_h))
    ΔB = η * h^T @ (y * (y - θ_y))

    Where θ_h, θ_y are running averages of h², y² respectively.
    """

    def __init__(
        self,
        learning_rate: float = 1e-5,
        threshold_decay: float = 0.99,
        **kwargs
    ):
        super().__init__(learning_rate, **kwargs)
        self.threshold_decay = threshold_decay
        self.theta_h = None  # Running threshold for hidden
        self.theta_y = None  # Running threshold for output

    def compute_update(self, ctx: UpdateContext) -> Tuple[torch.Tensor, torch.Tensor]:
        x, h, y = ctx.x, ctx.h, ctx.y

        # Flatten
        x_flat = x.reshape(-1, x.shape[-1])
        h_flat = h.reshape(-1, h.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])

        # Compute current activation magnitudes squared
        h_sq = torch.mean(h_flat ** 2, dim=0)  # [rank]
        y_sq = torch.mean(y_flat ** 2, dim=0)  # [d_out]

        # Initialize or update thresholds
        if self.theta_h is None or self.theta_h.shape != h_sq.shape:
            self.theta_h = h_sq.detach().clone()
        else:
            self.theta_h = self.threshold_decay * self.theta_h + (1 - self.threshold_decay) * h_sq.detach()

        if self.theta_y is None or self.theta_y.shape != y_sq.shape:
            self.theta_y = y_sq.detach().clone()
        else:
            self.theta_y = self.threshold_decay * self.theta_y + (1 - self.threshold_decay) * y_sq.detach()

        # BCM modulation: y * (y - θ)
        h_modulated = h_flat * (h_flat - self.theta_h.unsqueeze(0))  # [N, rank]
        y_modulated = y_flat * (y_flat - self.theta_y.unsqueeze(0))  # [N, d_out]

        # Compute updates
        delta_A = x_flat.T @ h_modulated / (x_flat.shape[0] + 1e-8)
        delta_B = h_flat.T @ y_modulated / (h_flat.shape[0] + 1e-8)

        return self.learning_rate * delta_A, self.learning_rate * delta_B


class PredictiveCodingUpdate(UpdateRule):
    """
    Predictive Coding: Learn by minimizing prediction errors

    Mathematical Formulation:
    ========================

    Each layer tries to predict its input from its output.
    The prediction error provides the learning signal.

    Forward: y = f(x)
    Backward prediction: x_pred = g(y)
    Error: ε = x - x_pred

    For LoRA, we can formulate this as:
    - The LoRA contribution should be predictable from the output
    - Error in this prediction drives learning

    Specifically:
    h = x @ A
    y_lora = h @ B

    Prediction: h_pred = y_lora @ B^T @ (B @ B^T)^{-1}  # Pseudo-inverse
    Error: ε_h = h - h_pred

    Update: ΔB = η * h^T @ (y - ε_y)  where ε_y is output prediction error

    Simplified version (avoiding inverse):
    We use the principle that weights should adjust to reduce prediction error
    at each layer:

    ΔA = η * ε_x^T @ h  where ε_x approximates input prediction error
    ΔB = η * h^T @ ε_y  where ε_y approximates output prediction error
    """

    def __init__(
        self,
        learning_rate: float = 1e-5,
        error_decay: float = 0.9,
        use_approximate_inverse: bool = True,
        **kwargs
    ):
        super().__init__(learning_rate, **kwargs)
        self.error_decay = error_decay
        self.use_approximate_inverse = use_approximate_inverse
        self.prev_h = None
        self.prev_y = None

    def compute_update(self, ctx: UpdateContext) -> Tuple[torch.Tensor, torch.Tensor]:
        x, h, y, A, B = ctx.x, ctx.h, ctx.y, ctx.A, ctx.B

        # Flatten
        x_flat = x.reshape(-1, x.shape[-1])
        h_flat = h.reshape(-1, h.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])

        # Compute prediction errors
        # Error 1: Can we predict h from y? (using B's pseudo-inverse)
        if self.use_approximate_inverse:
            # Approximate: h_pred ≈ y @ B^T (ignoring normalization)
            # This is a simplified version that works well in practice
            h_pred = y_flat @ B.T
            h_pred = h_pred / (torch.norm(B, dim=0, keepdim=True).T + 1e-8)
        else:
            # Full pseudo-inverse (more expensive)
            B_pinv = torch.linalg.pinv(B)
            h_pred = y_flat @ B_pinv.T

        eps_h = h_flat - h_pred  # Prediction error for hidden

        # Error 2: Can we predict x from h? (using A's pseudo-inverse)
        if self.use_approximate_inverse:
            x_pred = h_flat @ A.T
            x_pred = x_pred / (torch.norm(A, dim=0, keepdim=True).T + 1e-8)
        else:
            A_pinv = torch.linalg.pinv(A)
            x_pred = h_flat @ A_pinv.T

        eps_x = x_flat - x_pred  # Prediction error for input

        # Updates to minimize prediction error
        # ΔA should reduce eps_x: make A better at encoding x into h
        delta_A = eps_x.T @ h_flat / (x_flat.shape[0] + 1e-8)

        # ΔB should reduce eps_h: make B better at encoding h into y
        delta_B = eps_h.T @ y_flat / (h_flat.shape[0] + 1e-8)
        delta_B = delta_B.T  # Transpose to get [rank, d_out]

        return self.learning_rate * delta_A, self.learning_rate * delta_B


class EnergyBasedUpdate(UpdateRule):
    """
    Energy-Based Learning: Minimize an energy function

    Mathematical Formulation:
    ========================

    Define an energy function E that captures "coherence" of representations:

    E = -Σ_i Σ_j w_ij * h_i · h_j / T  (softmax attention energy)

    Or simpler:
    E = ||h||² - λ * h · h_mean  (deviation from mean + magnitude)

    The model should have LOW energy for coherent representations
    and HIGH energy for incoherent ones.

    Update rule: ΔW = -η * ∂E/∂W

    We use a contrastive formulation:
    - Positive: current activations (should have low energy)
    - Negative: perturbed activations (should have high energy)

    ΔA = η * (∂E_neg/∂A - ∂E_pos/∂A)

    Simplified local version:
    E_local = ||h - h_target||² where h_target is a "settled" state

    We approximate by tracking running statistics and updating
    toward more "typical" activation patterns.
    """

    def __init__(
        self,
        learning_rate: float = 1e-5,
        temperature: float = 1.0,
        energy_type: str = "coherence",  # "coherence", "sparsity", "entropy"
        running_mean_decay: float = 0.99,
        **kwargs
    ):
        super().__init__(learning_rate, **kwargs)
        self.temperature = temperature
        self.energy_type = energy_type
        self.running_mean_decay = running_mean_decay
        self.h_running_mean = None
        self.y_running_mean = None

    def compute_energy(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute energy for given activations."""
        if self.energy_type == "coherence":
            # Low energy = high coherence (activations align with running mean)
            if self.h_running_mean is not None:
                energy = torch.mean((h - self.h_running_mean) ** 2)
            else:
                energy = torch.tensor(0.0, device=h.device)
        elif self.energy_type == "sparsity":
            # Low energy = sparse activations
            energy = torch.mean(torch.abs(h))
        elif self.energy_type == "entropy":
            # Low energy = low entropy (peaked distributions)
            h_softmax = F.softmax(h / self.temperature, dim=-1)
            energy = -torch.mean(torch.sum(h_softmax * torch.log(h_softmax + 1e-8), dim=-1))
        else:
            raise ValueError(f"Unknown energy type: {self.energy_type}")
        return energy

    def compute_update(self, ctx: UpdateContext) -> Tuple[torch.Tensor, torch.Tensor]:
        x, h, y, A, B = ctx.x, ctx.h, ctx.y, ctx.A, ctx.B

        # Flatten
        x_flat = x.reshape(-1, x.shape[-1])
        h_flat = h.reshape(-1, h.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])

        # Update running means
        h_mean = torch.mean(h_flat, dim=0)
        y_mean = torch.mean(y_flat, dim=0)

        if self.h_running_mean is None or self.h_running_mean.shape != h_mean.shape:
            self.h_running_mean = h_mean.detach().clone()
            self.y_running_mean = y_mean.detach().clone()
        else:
            self.h_running_mean = (self.running_mean_decay * self.h_running_mean +
                                   (1 - self.running_mean_decay) * h_mean.detach())
            self.y_running_mean = (self.running_mean_decay * self.y_running_mean +
                                   (1 - self.running_mean_decay) * y_mean.detach())

        # Compute energy gradient (move toward lower energy)
        # For coherence energy: E = ||h - h_mean||²
        # ∂E/∂A = 2 * (h - h_mean) * ∂h/∂A = 2 * (h - h_mean) * x
        h_error = h_flat - self.h_running_mean.unsqueeze(0)
        y_error = y_flat - self.y_running_mean.unsqueeze(0)

        # Gradient descent on energy
        delta_A = -x_flat.T @ h_error / (x_flat.shape[0] + 1e-8)
        delta_B = -h_flat.T @ y_error / (h_flat.shape[0] + 1e-8)

        return self.learning_rate * delta_A, self.learning_rate * delta_B


class SelfSupervisedMicroUpdate(UpdateRule):
    """
    Self-Supervised Micro-Updates: Next-token prediction with tiny learning rate

    Mathematical Formulation:
    ========================

    This is the "Version 2" from the spec - uses standard loss but with
    extremely small updates that approximate fluid drift.

    loss = CrossEntropy(model_output, next_token)
    ΔW = -η * ∂loss/∂W  where η ≈ 1e-8 (very tiny)

    Key insight: With small enough η, discrete gradient updates approximate
    continuous drift. This is the most practical approach.

    For LoRA, we can compute the gradient locally:
    - During forward pass, store x, h
    - After getting the loss gradient from the model, backprop through LoRA

    Note: This requires gradients, unlike pure local rules.
    We provide a "gradient-free approximation" mode that estimates
    the gradient using finite differences or other techniques.
    """

    def __init__(
        self,
        learning_rate: float = 1e-8,  # Extremely small!
        use_gradient_estimate: bool = False,
        gradient_noise_scale: float = 0.01,
        **kwargs
    ):
        super().__init__(learning_rate, **kwargs)
        self.use_gradient_estimate = use_gradient_estimate
        self.gradient_noise_scale = gradient_noise_scale
        self.cached_grads = {}

    def cache_gradients(self, A_grad: torch.Tensor, B_grad: torch.Tensor):
        """Cache gradients from backprop for update computation."""
        self.cached_grads['A'] = A_grad.clone()
        self.cached_grads['B'] = B_grad.clone()

    def compute_update(self, ctx: UpdateContext) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_gradient_estimate:
            # Estimate gradient without backprop using local information
            # This is less accurate but doesn't require loss
            return self._estimate_gradient(ctx)
        else:
            # Use cached gradients from backprop
            if 'A' in self.cached_grads and 'B' in self.cached_grads:
                delta_A = -self.learning_rate * self.cached_grads['A']
                delta_B = -self.learning_rate * self.cached_grads['B']
                return delta_A, delta_B
            else:
                # No gradients available, return zero update
                return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B)

    def _estimate_gradient(self, ctx: UpdateContext) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate gradient without explicit backprop.

        Uses the principle that output changes approximate gradients:
        If y changes when A changes, that indicates ∂loss/∂A direction.

        We use activation statistics as a proxy for loss gradient.
        """
        x, h, y, A, B = ctx.x, ctx.h, ctx.y, ctx.A, ctx.B

        x_flat = x.reshape(-1, x.shape[-1])
        h_flat = h.reshape(-1, h.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])

        # Proxy gradient: direction that increases output magnitude
        # (assuming loss wants to increase confidence/decrease entropy)
        y_direction = torch.sign(y_flat - y_flat.mean(dim=-1, keepdim=True))
        h_direction = torch.sign(h_flat - h_flat.mean(dim=-1, keepdim=True))

        # Add noise for exploration (like in evolution strategies)
        noise_A = torch.randn_like(A) * self.gradient_noise_scale
        noise_B = torch.randn_like(B) * self.gradient_noise_scale

        # Approximate gradient
        delta_A = x_flat.T @ h_direction / (x_flat.shape[0] + 1e-8) + noise_A
        delta_B = h_flat.T @ y_direction / (h_flat.shape[0] + 1e-8) + noise_B

        return self.learning_rate * delta_A, self.learning_rate * delta_B


class HybridUpdate(UpdateRule):
    """
    Hybrid Update: Combine multiple update rules

    This is the recommended approach - combine the strengths of different rules:
    1. Hebbian: Captures correlations
    2. Oja: Provides stability
    3. BCM: Adds threshold dynamics
    4. Predictive Coding: Adds error correction

    Combined update:
    Δ = α_hebb * Δ_hebb + α_oja * Δ_oja + α_bcm * Δ_bcm + α_pred * Δ_pred

    The weights can be tuned or learned.
    """

    def __init__(
        self,
        learning_rate: float = 1e-5,
        rules: Optional[Dict[str, Tuple[UpdateRule, float]]] = None,
        **kwargs
    ):
        super().__init__(learning_rate, **kwargs)

        if rules is None:
            # Default configuration
            self.rules = {
                'oja': (OjaUpdate(learning_rate=1.0), 0.5),
                'bcm': (BCMUpdate(learning_rate=1.0), 0.3),
                'energy': (EnergyBasedUpdate(learning_rate=1.0), 0.2),
            }
        else:
            self.rules = rules

    def compute_update(self, ctx: UpdateContext) -> Tuple[torch.Tensor, torch.Tensor]:
        total_delta_A = torch.zeros_like(ctx.A)
        total_delta_B = torch.zeros_like(ctx.B)

        for name, (rule, weight) in self.rules.items():
            delta_A, delta_B = rule.compute_update(ctx)
            total_delta_A += weight * delta_A
            total_delta_B += weight * delta_B

        return self.learning_rate * total_delta_A, self.learning_rate * total_delta_B

    def __repr__(self):
        rule_strs = [f"{name}:{weight:.2f}" for name, (_, weight) in self.rules.items()]
        return f"HybridUpdate({', '.join(rule_strs)})"


# Factory function for easy creation
def create_update_rule(rule_type: str, **kwargs) -> UpdateRule:
    """Create an update rule by name."""
    rules = {
        'hebbian': HebbianUpdate,
        'oja': OjaUpdate,
        'bcm': BCMUpdate,
        'predictive': PredictiveCodingUpdate,
        'energy': EnergyBasedUpdate,
        'micro': SelfSupervisedMicroUpdate,
        'hybrid': HybridUpdate,
    }

    if rule_type not in rules:
        raise ValueError(f"Unknown rule type: {rule_type}. Available: {list(rules.keys())}")

    return rules[rule_type](**kwargs)
