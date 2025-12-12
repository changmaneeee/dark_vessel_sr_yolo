"""
=============================================================================
Models Module
=============================================================================

모든 모델 관련 모듈을 포함

[구조]
models/
├── sr_models/     # SR 모델
│   ├── BaseSRModel
│   ├── RFDN
│   ├── MambaSR (TODO)
│   └── TTST (TODO)
├── detectors/     # Detector
│   └── YOLOWrapper
├── gates/         # Gate 네트워크 ✓
│   ├── LightweightGate
│   └── SoftGateModule
├── fusion/        # Fusion 모듈 ✓
│   ├── MultiScaleAttentionFusion
│   └── SingleScaleFusion
└── pipelines/     # 통합 파이프라인
    ├── BasePipeline
    ├── Arch0Sequential ✓
    ├── Arch2SoftGate ✓
    ├── Arch4Adaptive (TODO)
    └── Arch5BFusion ✓

[사용 예시]
# 개별 모델
from src.models.sr_models import RFDN
from src.models.detectors import YOLOWrapper
from src.models.gates import LightweightGate
from src.models.fusion import MultiScaleAttentionFusion

# 파이프라인
from src.models.pipelines import Arch0Sequential, Arch2SoftGate, Arch5BFusion, get_pipeline
"""

from src.models import sr_models
from src.models import detectors
from src.models import gates
from src.models import fusion
from src.models import pipelines

__all__ = [
    "sr_models",
    "detectors",
    "gates",
    "fusion",
    "pipelines",
]