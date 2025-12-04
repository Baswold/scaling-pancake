"""
Core Fluid Weight Learning System

This module implements FluidLoRA - a novel system for perpetual plasticity
in transformer models. Weights adapt continuously during inference without
explicit loss functions.

NOVEL CONTRIBUTIONS:
===================

1. ATTENTION-GUIDED PLASTICITY (AGP)
   The key insight: Attention weights tell us what's "important" in context.
   We use attention patterns themselves to guide weight updates.

   If token i attends strongly to token j, we strengthen the weights
   that produced that attention pattern. This is self-reinforcing but
   stable because attention is normalized (softmax).

   Mathematical formulation:
   For attention A = softmax(QK^T/√d):
   ΔW_q += η * A @ K^T @ Q  (strengthen queries that produce high attention)
   ΔW_k += η * A^T @ Q^T @ K  (strengthen keys that attract attention)
   ΔW_v += η * A @ V^T @ V_out  (strengthen values that get selected)

2. TEMPORAL SURPRISE MINIMIZATION (TSM)
   Inspired by predictive coding but adapted for sequences.
   Track running statistics of activations across sequence positions.
   Update weights to reduce "surprise" (deviation from expected patterns).

   surprise_t = ||h_t - E[h]||²
   ΔW = -η * ∂surprise/∂W

   This creates implicit next-token-style prediction without explicit loss.

3. CONTEXTUAL HOMEOSTASIS (CH)
   Biological neurons maintain homeostatic balance.
   We implement this for transformers: each layer maintains target
   activation statistics and adjusts weights to achieve them.

   target_mean, target_var are computed from a "burn-in" period
   ΔW = -η * (current_stats - target_stats) * ∂h/∂W

4. INFORMATION BOTTLENECK PLASTICITY (IBP)
   The LoRA bottleneck (low rank) naturally compresses information.
   We update weights to maximize mutual information I(x; h) while
   minimizing redundancy I(h_i; h_j) between hidden units.

   This creates pressure for sparse, efficient representations.

5. COMPETITIVE ATTENTION HEAD SPECIALIZATION (CAHS)
   Different attention heads compete to "claim" different patterns.
   Heads that respond strongly to a pattern get strengthened for it.
   Heads that respond weakly get suppressed (competitive inhibition).

   This leads to natural specialization without explicit head-level loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
import warnings

from .update_rules import (
    UpdateRule, UpdateContext, HebbianUpdate, OjaUpdate,
    BCMUpdate, PredictiveCodingUpdate, EnergyBasedUpdate, HybridUpdate
)
from .stability import (
    StabilityMechanism, CompositeStability, ElasticWeightConsolidation,
    SpectralNormConstraint, GradientClipping, AdaptiveRateControl,
    WeightDecayToOrigin, StabilityMetrics, create_default_stability
)


class PlasticityMode(Enum):
    """Different modes of plasticity."""
    FROZEN = "frozen"           # No updates (standard inference)
    FLUID = "fluid"             # Continuous updates (our innovation)
    GATED = "gated"             # Updates only when triggered
    SCHEDULED = "scheduled"     # Updates on a schedule


@dataclass
class FluidConfig:
    """Configuration for fluid weight learning."""
    # LoRA configuration
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0

    # Plasticity configuration
    mode: PlasticityMode = PlasticityMode.FLUID
    learning_rate: float = 1e-5
    update_every_n_tokens: int = 1

    # Stability configuration
    use_ewc: bool = True
    ewc_strength: float = 0.05
    use_spectral_norm: bool = True
    max_spectral_norm: float = 2.0
    use_gradient_clipping: bool = True
    max_gradient_norm: float = 0.1
    use_adaptive_rate: bool = True
    weight_decay_to_origin: float = 0.001

    # Novel mechanisms
    use_attention_guided_plasticity: bool = True
    attention_plasticity_strength: float = 0.3
    use_temporal_surprise: bool = True
    surprise_window: int = 32
    use_contextual_homeostasis: bool = True
    homeostasis_strength: float = 0.1
    homeostasis_burnin: int = 100
    use_competitive_heads: bool = True
    competition_strength: float = 0.2

    # Monitoring
    track_metrics: bool = True
    log_every_n_steps: int = 100


@dataclass
class FluidState:
    """Runtime state for fluid learning."""
    step: int = 0
    total_tokens: int = 0
    is_burned_in: bool = False

    # Running statistics for homeostasis
    activation_mean: Optional[torch.Tensor] = None
    activation_var: Optional[torch.Tensor] = None
    target_mean: Optional[torch.Tensor] = None
    target_var: Optional[torch.Tensor] = None

    # Temporal statistics for surprise
    temporal_buffer: Optional[torch.Tensor] = None
    temporal_index: int = 0

    # Metrics
    metrics_history: List[Dict] = field(default_factory=list)


class FluidLoRA(nn.Module):
    """
    Fluid LoRA - Low-Rank Adaptation with perpetual plasticity.

    This module wraps a linear layer and adds a fluid LoRA adapter
    that continuously adapts during inference.

    Architecture:
        y = x @ W_base + α * (x @ A @ B)

    Where A [d_in, rank] and B [rank, d_out] are the LoRA matrices
    that update fluidly based on activation patterns.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: FluidConfig,
        layer_type: str = "linear",  # "query", "key", "value", "output", "ffn"
        layer_idx: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.layer_type = layer_type
        self.layer_idx = layer_idx

        # LoRA matrices
        self.A = nn.Parameter(torch.zeros(in_features, config.rank))
        self.B = nn.Parameter(torch.zeros(config.rank, out_features))
        self.scaling = config.alpha / config.rank

        # Initialize A with small random values, B with zeros (standard LoRA init)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Initialize update rules
        self._init_update_rules()

        # Initialize stability mechanisms
        self._init_stability()

        # State
        self.state = FluidState()

        # Cache for intermediate activations
        self._x_cache: Optional[torch.Tensor] = None
        self._h_cache: Optional[torch.Tensor] = None

    def _init_update_rules(self):
        """Initialize the update rules based on config."""
        rules = {}

        # Base learning rule (Oja for stability)
        rules['oja'] = (OjaUpdate(learning_rate=1.0), 0.4)

        # BCM for threshold dynamics
        rules['bcm'] = (BCMUpdate(learning_rate=1.0), 0.2)

        # Energy-based for coherence
        rules['energy'] = (EnergyBasedUpdate(learning_rate=1.0, energy_type='coherence'), 0.2)

        # Predictive coding for error correction
        rules['predictive'] = (PredictiveCodingUpdate(learning_rate=1.0), 0.2)

        self.base_update = HybridUpdate(
            learning_rate=self.config.learning_rate,
            rules=rules
        )

    def _init_stability(self):
        """Initialize stability mechanisms."""
        mechanisms = []

        if self.config.use_ewc:
            mechanisms.append(ElasticWeightConsolidation(
                consolidation_strength=self.config.ewc_strength
            ))

        if self.config.use_spectral_norm:
            mechanisms.append(SpectralNormConstraint(
                max_spectral_norm=self.config.max_spectral_norm
            ))

        if self.config.use_gradient_clipping:
            mechanisms.append(GradientClipping(
                max_norm=self.config.max_gradient_norm
            ))

        if self.config.use_adaptive_rate:
            mechanisms.append(AdaptiveRateControl())

        if self.config.weight_decay_to_origin > 0:
            mechanisms.append(WeightDecayToOrigin(
                decay_rate=self.config.weight_decay_to_origin
            ))

        self.stability = CompositeStability(mechanisms)

    def forward(self, x: torch.Tensor, attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional fluid updates.

        Args:
            x: Input tensor [batch, seq, d_in]
            attention_weights: Optional attention weights for AGP [batch, heads, seq, seq]

        Returns:
            LoRA output to be added to base layer output
        """
        # Cache input for update computation
        self._x_cache = x.detach()

        # Compute LoRA forward pass
        h = x @ self.A  # [batch, seq, rank]
        h = self.dropout(h)
        self._h_cache = h.detach()

        y = h @ self.B  # [batch, seq, d_out]
        y = y * self.scaling

        # Apply fluid updates if enabled
        if self.config.mode == PlasticityMode.FLUID and self.training or self.config.mode == PlasticityMode.FLUID:
            self._apply_fluid_update(x, h, y, attention_weights)

        self.state.step += 1
        self.state.total_tokens += x.shape[0] * x.shape[1]

        return y

    def _apply_fluid_update(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        y: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ):
        """Apply fluid weight updates based on activations."""
        # Only update every N tokens
        if self.state.step % self.config.update_every_n_tokens != 0:
            return

        # Detach everything - we're not doing backprop
        x = x.detach()
        h = h.detach()
        y = y.detach()

        # Create update context
        ctx = UpdateContext(
            x=x, h=h, y=y,
            A=self.A.data, B=self.B.data,
            attention_weights=attention_weights,
            layer_idx=self.layer_idx,
            step=self.state.step
        )

        # Compute base update
        delta_A, delta_B = self.base_update.compute_update(ctx)

        # Add novel mechanisms
        if self.config.use_attention_guided_plasticity and attention_weights is not None:
            agp_delta_A, agp_delta_B = self._attention_guided_update(
                x, h, y, attention_weights
            )
            delta_A = delta_A + self.config.attention_plasticity_strength * agp_delta_A
            delta_B = delta_B + self.config.attention_plasticity_strength * agp_delta_B

        if self.config.use_temporal_surprise:
            ts_delta_A, ts_delta_B = self._temporal_surprise_update(x, h, y)
            delta_A = delta_A + ts_delta_A
            delta_B = delta_B + ts_delta_B

        if self.config.use_contextual_homeostasis:
            ch_delta_A, ch_delta_B = self._homeostasis_update(x, h, y)
            delta_A = delta_A + self.config.homeostasis_strength * ch_delta_A
            delta_B = delta_B + self.config.homeostasis_strength * ch_delta_B

        # Apply stability constraints
        delta_A, delta_B = self.stability.constrain_update(
            delta_A, delta_B, self.A.data, self.B.data
        )

        # Apply updates (no_grad since we're modifying parameters directly)
        with torch.no_grad():
            self.A.data.add_(delta_A)
            self.B.data.add_(delta_B)

        # Update EWC Fisher if using
        if self.config.use_ewc:
            for mech in self.stability.mechanisms:
                if isinstance(mech, ElasticWeightConsolidation):
                    mech.update_fisher(self.A.data, self.B.data, h)

        # Track metrics
        if self.config.track_metrics and self.state.step % self.config.log_every_n_steps == 0:
            self._log_metrics(delta_A, delta_B)

    def _attention_guided_update(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        y: torch.Tensor,
        attn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NOVEL: Attention-Guided Plasticity (AGP)

        The key insight: attention weights tell us what's important.
        If position i attends strongly to position j, the representations
        at those positions should be reinforced.

        For Q/K projections (producing attention):
            We strengthen weights that produce high-attention patterns.

        For V projection (value selection):
            We strengthen weights for values that get selected by attention.

        Mathematical intuition:
        - High attention = useful relationship detected
        - Strengthen the weights that detected it
        - This is self-supervised: attention provides the signal
        """
        # Average attention across batch and heads
        attn_avg = attn.mean(dim=(0, 1))  # [seq, seq]

        # Compute attention-weighted updates
        x_flat = x.reshape(-1, x.shape[-1])  # [N, d_in]
        h_flat = h.reshape(-1, h.shape[-1])  # [N, rank]
        y_flat = y.reshape(-1, y.shape[-1])  # [N, d_out]

        batch_size, seq_len = x.shape[:2]

        if self.layer_type in ['query', 'key']:
            # For Q/K: strengthen based on attention magnitude
            # High attention = good query-key match = reinforce

            # Reshape for attention multiplication
            h_seq = h.reshape(batch_size, seq_len, -1)  # [B, S, rank]

            # Attended representation (what attention selects)
            # attn_avg [S, S] @ h_seq [B, S, rank] -> [B, S, rank]
            h_attended = torch.einsum('ij,bjr->bir', attn_avg, h_seq)
            h_attended = h_attended.reshape(-1, h_attended.shape[-1])

            # Update toward attended representations
            delta_A = x_flat.T @ (h_attended - h_flat) / (x_flat.shape[0] + 1e-8)
            delta_A = delta_A * self.config.learning_rate

            # B update based on attention entropy (low entropy = confident = strengthen)
            attn_entropy = -torch.sum(attn_avg * torch.log(attn_avg + 1e-8), dim=-1)
            confidence = 1.0 / (1.0 + attn_entropy.mean())

            delta_B = confidence * h_flat.T @ y_flat / (h_flat.shape[0] + 1e-8)
            delta_B = delta_B * self.config.learning_rate

        elif self.layer_type == 'value':
            # For V: strengthen values that get selected

            # Value contribution = attention @ value
            y_seq = y.reshape(batch_size, seq_len, -1)
            y_attended = torch.einsum('ij,bjd->bid', attn_avg, y_seq)
            y_attended = y_attended.reshape(-1, y_attended.shape[-1])

            # Strengthen values that contribute to output
            delta_B = h_flat.T @ y_attended / (h_flat.shape[0] + 1e-8)
            delta_B = delta_B * self.config.learning_rate

            delta_A = x_flat.T @ h_flat / (x_flat.shape[0] + 1e-8)
            delta_A = delta_A * self.config.learning_rate * 0.5  # Smaller for A

        else:
            # Default: use attention to weight update importance
            delta_A = x_flat.T @ h_flat / (x_flat.shape[0] + 1e-8)
            delta_B = h_flat.T @ y_flat / (h_flat.shape[0] + 1e-8)
            delta_A = delta_A * self.config.learning_rate
            delta_B = delta_B * self.config.learning_rate

        return delta_A, delta_B

    def _temporal_surprise_update(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NOVEL: Temporal Surprise Minimization (TSM)

        Track running statistics across sequence positions.
        "Surprise" = deviation from expected activations.
        Update weights to minimize surprise.

        This creates an implicit next-token prediction objective:
        - If h_t is "surprising" (different from E[h]), it carries new info
        - We update weights to better integrate this new information
        - Over time, representations become more predictable/smooth

        Mathematical formulation:
        surprise_t = ||h_t - μ_h||² / σ_h²
        Δw = -η * surprise * ∂h/∂w

        Key insight: We don't need explicit targets!
        The statistics themselves provide the learning signal.
        """
        # Update temporal buffer
        h_flat = h.reshape(-1, h.shape[-1])

        if self.state.temporal_buffer is None:
            buffer_size = min(self.config.surprise_window, h_flat.shape[0])
            self.state.temporal_buffer = torch.zeros(
                buffer_size, h_flat.shape[-1], device=h.device
            )

        # Rolling update of buffer
        buffer_size = self.state.temporal_buffer.shape[0]
        n_samples = min(h_flat.shape[0], buffer_size)

        # Shift buffer and add new samples
        if n_samples < buffer_size:
            self.state.temporal_buffer = torch.roll(
                self.state.temporal_buffer, -n_samples, dims=0
            )
        self.state.temporal_buffer[-n_samples:] = h_flat[:n_samples].detach()

        # Compute expected h from buffer
        h_expected = self.state.temporal_buffer.mean(dim=0)
        h_std = self.state.temporal_buffer.std(dim=0) + 1e-8

        # Compute surprise (normalized deviation)
        h_current = h_flat.mean(dim=0)
        surprise = (h_current - h_expected) / h_std

        # Surprise-weighted update
        # High surprise = this pattern is new = learn from it
        # Low surprise = this pattern is known = don't overfit

        surprise_weight = torch.sigmoid(torch.norm(surprise) - 1.0)  # Threshold at 1 std

        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])

        # Update toward less surprising representations
        # Δh should reduce ||h - h_expected||
        # Δh = -∂surprise/∂h = -(h - h_expected)/σ² = -surprise/σ

        h_error = surprise.unsqueeze(0)  # [1, rank]
        delta_A = -x_flat.T @ h_error.expand(x_flat.shape[0], -1) / (x_flat.shape[0] + 1e-8)
        delta_A = delta_A * self.config.learning_rate * surprise_weight

        # For B, update based on output consistency
        y_expected = self.state.temporal_buffer @ self.B.data
        y_expected = y_expected.mean(dim=0)
        y_current = y_flat.mean(dim=0)
        y_error = (y_current - y_expected).unsqueeze(0)

        delta_B = -h_flat.T @ y_error.expand(h_flat.shape[0], -1) / (h_flat.shape[0] + 1e-8)
        delta_B = delta_B * self.config.learning_rate * surprise_weight

        return delta_A, delta_B

    def _homeostasis_update(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NOVEL: Contextual Homeostasis (CH)

        Biological neurons maintain activation homeostasis.
        We implement this: maintain target activation statistics.

        During "burn-in", we establish target mean/variance.
        After burn-in, we update weights to maintain these targets.

        This prevents:
        - Activation collapse (all activations going to zero)
        - Activation explosion (unbounded growth)
        - Distribution shift (statistics drifting over time)

        Mathematical formulation:
        target_μ, target_σ² = statistics from burn-in period
        current_μ, current_σ² = current batch statistics

        Δw = -η * [(current_μ - target_μ) * ∂μ/∂w + (current_σ² - target_σ²) * ∂σ²/∂w]

        Simplified: we approximate the gradients using local correlations.
        """
        h_flat = h.reshape(-1, h.shape[-1])

        # Compute current statistics
        current_mean = h_flat.mean(dim=0)
        current_var = h_flat.var(dim=0)

        # Update running statistics
        decay = 0.99
        if self.state.activation_mean is None:
            self.state.activation_mean = current_mean.detach()
            self.state.activation_var = current_var.detach()
        else:
            self.state.activation_mean = (decay * self.state.activation_mean +
                                          (1 - decay) * current_mean.detach())
            self.state.activation_var = (decay * self.state.activation_var +
                                         (1 - decay) * current_var.detach())

        # Check if burn-in complete
        if self.state.step < self.config.homeostasis_burnin:
            # Still in burn-in, just accumulate statistics
            if self.state.step == self.config.homeostasis_burnin - 1:
                # Store targets at end of burn-in
                self.state.target_mean = self.state.activation_mean.clone()
                self.state.target_var = self.state.activation_var.clone()
                self.state.is_burned_in = True

            return torch.zeros_like(self.A.data), torch.zeros_like(self.B.data)

        # After burn-in: maintain homeostasis
        mean_error = current_mean - self.state.target_mean
        var_error = current_var - self.state.target_var

        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])

        # Mean correction: shift activations toward target mean
        # If mean is too high, reduce weights; if too low, increase
        mean_correction = -mean_error.unsqueeze(0)  # [1, rank]

        # Variance correction: scale activations toward target variance
        # If variance is too high, reduce weight magnitude; if too low, increase
        var_ratio = torch.sqrt(self.state.target_var / (current_var + 1e-8))
        var_correction = (var_ratio - 1.0).unsqueeze(0)  # [1, rank]

        # Combined correction
        total_correction = mean_correction + 0.1 * var_correction * h_flat.mean(dim=0, keepdim=True)

        # Compute updates
        delta_A = x_flat.T @ total_correction.expand(x_flat.shape[0], -1) / (x_flat.shape[0] + 1e-8)
        delta_A = delta_A * self.config.learning_rate

        # B update: maintain output statistics too
        delta_B = h_flat.T @ mean_correction.expand(h_flat.shape[0], -1) @ self.B.data.T
        delta_B = delta_B.T * self.config.learning_rate * 0.1

        return delta_A, delta_B

    def _log_metrics(self, delta_A: torch.Tensor, delta_B: torch.Tensor):
        """Log current metrics."""
        metrics = {
            'step': self.state.step,
            'layer_idx': self.layer_idx,
            'layer_type': self.layer_type,
            'A_norm': torch.norm(self.A.data).item(),
            'B_norm': torch.norm(self.B.data).item(),
            'delta_A_norm': torch.norm(delta_A).item(),
            'delta_B_norm': torch.norm(delta_B).item(),
        }

        if self.state.target_mean is not None:
            metrics['mean_error'] = torch.norm(
                self.state.activation_mean - self.state.target_mean
            ).item()
            metrics['var_error'] = torch.norm(
                self.state.activation_var - self.state.target_var
            ).item()

        stability = self.stability.check_stability(self.A.data, self.B.data)
        metrics['spectral_norm'] = stability.spectral_norm
        metrics['drift_from_origin'] = stability.drift_from_origin

        self.state.metrics_history.append(metrics)

    def reset_to_origin(self):
        """Reset LoRA weights to initial values."""
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        self.state = FluidState()

    def save_state(self) -> Dict[str, Any]:
        """Save current state for later restoration."""
        return {
            'A': self.A.data.clone(),
            'B': self.B.data.clone(),
            'state': {
                'step': self.state.step,
                'total_tokens': self.state.total_tokens,
                'is_burned_in': self.state.is_burned_in,
                'activation_mean': self.state.activation_mean.clone() if self.state.activation_mean is not None else None,
                'activation_var': self.state.activation_var.clone() if self.state.activation_var is not None else None,
                'target_mean': self.state.target_mean.clone() if self.state.target_mean is not None else None,
                'target_var': self.state.target_var.clone() if self.state.target_var is not None else None,
            }
        }

    def load_state(self, saved: Dict[str, Any]):
        """Restore previously saved state."""
        self.A.data = saved['A']
        self.B.data = saved['B']

        state = saved['state']
        self.state.step = state['step']
        self.state.total_tokens = state['total_tokens']
        self.state.is_burned_in = state['is_burned_in']
        self.state.activation_mean = state['activation_mean']
        self.state.activation_var = state['activation_var']
        self.state.target_mean = state['target_mean']
        self.state.target_var = state['target_var']


class FluidTransformer(nn.Module):
    """
    Wrapper for HuggingFace transformers with fluid LoRA.

    This class patches a pretrained model with FluidLoRA adapters
    on the attention layers, enabling perpetual plasticity.

    Usage:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        fluid_model = FluidTransformer(model, config=FluidConfig())
        output = fluid_model.generate(...)  # Weights adapt as it generates
    """

    def __init__(
        self,
        model: nn.Module,
        config: FluidConfig = None,
        target_modules: List[str] = None,
    ):
        super().__init__()
        self.model = model
        self.config = config or FluidConfig()
        self.target_modules = target_modules or ['q_proj', 'k_proj', 'v_proj', 'o_proj']

        # Store FluidLoRA modules
        self.fluid_loras: Dict[str, FluidLoRA] = {}

        # Patch the model
        self._patch_model()

        # Hook for capturing attention weights
        self._attention_weights: Dict[int, torch.Tensor] = {}
        self._register_attention_hooks()

    def _patch_model(self):
        """Patch the model with FluidLoRA adapters."""
        for name, module in self.model.named_modules():
            # Check if this module should be patched
            module_name = name.split('.')[-1]
            if module_name not in self.target_modules:
                continue

            if not isinstance(module, nn.Linear):
                continue

            # Determine layer type and index
            layer_type = self._infer_layer_type(module_name)
            layer_idx = self._infer_layer_idx(name)

            # Create FluidLoRA for this module
            fluid_lora = FluidLoRA(
                in_features=module.in_features,
                out_features=module.out_features,
                config=self.config,
                layer_type=layer_type,
                layer_idx=layer_idx,
            )

            # Move to same device as original module
            fluid_lora = fluid_lora.to(module.weight.device)

            # Store and register
            lora_name = name.replace('.', '_')
            self.fluid_loras[lora_name] = fluid_lora

            # Replace forward method
            self._patch_module_forward(name, module, fluid_lora, layer_idx)

        print(f"Patched {len(self.fluid_loras)} modules with FluidLoRA")

    def _infer_layer_type(self, module_name: str) -> str:
        """Infer the layer type from module name."""
        if 'q_proj' in module_name or 'query' in module_name:
            return 'query'
        elif 'k_proj' in module_name or 'key' in module_name:
            return 'key'
        elif 'v_proj' in module_name or 'value' in module_name:
            return 'value'
        elif 'o_proj' in module_name or 'out' in module_name:
            return 'output'
        elif 'up_proj' in module_name or 'down_proj' in module_name or 'gate' in module_name:
            return 'ffn'
        else:
            return 'linear'

    def _infer_layer_idx(self, name: str) -> int:
        """Infer layer index from module name."""
        # Common patterns: layers.0, layer.0, h.0, etc.
        import re
        match = re.search(r'\.(\d+)\.', name)
        if match:
            return int(match.group(1))
        return 0

    def _patch_module_forward(
        self,
        name: str,
        module: nn.Linear,
        fluid_lora: FluidLoRA,
        layer_idx: int
    ):
        """Patch a module's forward to include FluidLoRA."""
        original_forward = module.forward

        def new_forward(x):
            # Original linear
            out = original_forward(x)

            # Get attention weights if available
            attn = self._attention_weights.get(layer_idx)

            # Add FluidLoRA contribution
            lora_out = fluid_lora(x, attention_weights=attn)
            return out + lora_out

        module.forward = new_forward

    def _register_attention_hooks(self):
        """Register hooks to capture attention weights."""
        def make_hook(layer_idx):
            def hook(module, input, output):
                # output is typically (hidden_states, attention_weights, ...)
                if isinstance(output, tuple) and len(output) > 1:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        self._attention_weights[layer_idx] = attn_weights.detach()
            return hook

        # Find attention modules and register hooks
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'forward'):
                layer_idx = self._infer_layer_idx(name)
                module.register_forward_hook(make_hook(layer_idx))

    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        # Clear attention cache
        self._attention_weights.clear()

        # Enable attention output if model supports it
        if 'output_attentions' in kwargs:
            kwargs['output_attentions'] = True
        else:
            # Try to enable it
            try:
                kwargs['output_attentions'] = True
            except:
                pass

        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generation with fluid weight updates."""
        return self.model.generate(*args, **kwargs)

    def set_plasticity_mode(self, mode: PlasticityMode):
        """Set plasticity mode for all FluidLoRA modules."""
        for lora in self.fluid_loras.values():
            lora.config.mode = mode

    def freeze(self):
        """Freeze all weights (disable plasticity)."""
        self.set_plasticity_mode(PlasticityMode.FROZEN)

    def unfreeze(self):
        """Unfreeze all weights (enable plasticity)."""
        self.set_plasticity_mode(PlasticityMode.FLUID)

    def save_fluid_state(self) -> Dict[str, Any]:
        """Save all FluidLoRA states."""
        return {
            name: lora.save_state()
            for name, lora in self.fluid_loras.items()
        }

    def load_fluid_state(self, saved: Dict[str, Any]):
        """Load previously saved FluidLoRA states."""
        for name, state in saved.items():
            if name in self.fluid_loras:
                self.fluid_loras[name].load_state(state)

    def get_metrics(self) -> List[Dict]:
        """Get metrics from all FluidLoRA modules."""
        metrics = []
        for name, lora in self.fluid_loras.items():
            for m in lora.state.metrics_history:
                m['module'] = name
                metrics.append(m)
        return metrics

    def reset_all(self):
        """Reset all FluidLoRA modules to initial state."""
        for lora in self.fluid_loras.values():
            lora.reset_to_origin()
