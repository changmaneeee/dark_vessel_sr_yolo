"""
=============================================================================
Inference Module for Arch 0, 2, 4
=============================================================================
개별 학습된 SR + YOLO 가중치를 조합하여 inference 수행

[지원 아키텍처]
- Arch0: Sequential (LR → SR → YOLO)
- Arch2: Soft Gate (LR → Gate → SR/Bypass → YOLO)
- Arch4: Adaptive 2-Pass (LR → YOLO → [조건부 SR] → YOLO)

[가정]
- SR, YOLO: 개별 학습 완료
- Arch5B만 별도 Pipeline 학습 필요 (training/ 모듈 참조)
"""

from .inference import (
    BaseInference,
    Arch0Inference,
    Arch2Inference,
    Arch4Inference,
    Arch5BInference,
    process_folder,
    print_stats
)

from .soft_gate import (
    LightweightGate,
    EdgeBasedGate,
    SoftGateModule
)

__all__ = [
    # Inference Engines
    'BaseInference',
    'Arch0Inference',
    'Arch2Inference',
    'Arch4Inference',
    'Arch5BInference',
    
    # Utilities
    'process_folder',
    'print_stats',
    
    # Gate Modules
    'LightweightGate',
    'EdgeBasedGate',
    'SoftGateModule'
]