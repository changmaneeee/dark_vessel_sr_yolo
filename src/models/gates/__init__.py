"""
=============================================================================
Gates Module
=============================================================================

조건부 SR 적용을 위한 Gate 네트워크 모음

[사용 가능한 Gate]
- LightweightGate: 경량 CNN (~50K params)
- LightweightGateV2: 더 경량 (~25K params, Depthwise Separable)
- SoftGateModule: Gate + SR 결합 모듈

[사용 예시]
from src.models.gates import LightweightGate, SoftGateModule

gate_net = LightweightGate()
soft_gate = SoftGateModule(gate_net, sr_model, upscale=4)

output, gate_value = soft_gate(lr_image, return_gate=True)
"""

from src.models.gates.soft_gate import (
    LightweightGate,
    LightweightGateV2,
    SoftGateModule,
    DepthwiseSeparableConv
)

__all__ = [
    "LightweightGate",
    "LightweightGateV2",
    "SoftGateModule",
    "DepthwiseSeparableConv",
]