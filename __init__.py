"""
Fluid Weights: Perpetual Plasticity for Transformer Models

A novel learning system that enables continuous weight adaptation during inference
without explicit loss functions or training phases.

Author: Basil Jackson (with Claude)
Version: 1.0
"""

from .core import FluidLoRA, FluidTransformer
from .update_rules import (
    UpdateRule,
    HebbianUpdate,
    OjaUpdate,
    BCMUpdate,
    PredictiveCodingUpdate,
    EnergyBasedUpdate,
    SelfSupervisedMicroUpdate,
    HybridUpdate,
)
from .stability import (
    StabilityMechanism,
    ElasticWeightConsolidation,
    SpectralNormConstraint,
    WeightDecayToOrigin,
    GradientClipping,
    AdaptiveRateControl,
)

__version__ = "1.0.0"
__all__ = [
    "FluidLoRA",
    "FluidTransformer",
    "UpdateRule",
    "HebbianUpdate",
    "OjaUpdate",
    "BCMUpdate",
    "PredictiveCodingUpdate",
    "EnergyBasedUpdate",
    "SelfSupervisedMicroUpdate",
    "HybridUpdate",
    "StabilityMechanism",
    "ElasticWeightConsolidation",
    "SpectralNormConstraint",
    "WeightDecayToOrigin",
    "GradientClipping",
    "AdaptiveRateControl",
]
