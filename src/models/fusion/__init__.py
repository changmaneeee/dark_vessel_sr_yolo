"""Fusion Module

Feature fusion mechanisms for SR-Detection integration:
- AttentionFusion: Cross-attention based fusion
- GateNetwork: Gating mechanism for feature selection
"""

from src.models.fusion.attention_fusion import MultiScaleAttentionFusion, SingleScaleFusion
__all__ = ["MultiScaleAttentionFusion", "SingleScaleFusion"]