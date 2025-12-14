"""
=============================================================================
Pipelines Module
=============================================================================

SR-Detection 통합 파이프라인 모음

[아키텍처 비교]

| Arch | 이름 | 흐름 | 특징 |
|------|------|------|------|
| 0 | Sequential | LR→SR→HR→YOLO | Baseline, 최고 성능, 느림 |
| 2 | SoftGate | LR→Gate→조건부SR→YOLO | 연산량 절약 ✓ |
| 4 | Adaptive | LR→YOLO→ROI SR→YOLO→병합 | FN 감소 ✓ |
| 5-B | Fusion | SR_feat+YOLO_feat→Fusion→Det | 주력, 빠름 ✓ |

[사용 예시]
from src.models.pipelines import get_pipeline

model = get_pipeline('arch0', config)   # Sequential
model = get_pipeline('arch2', config)   # SoftGate
model = get_pipeline('arch4', config)   # Adaptive
model = get_pipeline('arch5b', config)  # Fusion
"""

from src.models.pipelines.base_pipeline import BasePipeline
from src.models.pipelines.arch0_sequential import Arch0Sequential
from src.models.pipelines.arch2_softgate import Arch2SoftGate
from src.models.pipelines.arch4_adaptive import Arch4Adaptive
from src.models.pipelines.arch5b_fusion import Arch5BFusion


def get_pipeline(arch_name: str, config):
    """
    파이프라인 팩토리 함수
    
    Args:
        arch_name: 아키텍처 이름
        config: 설정 객체
    
    Returns:
        해당 아키텍처의 파이프라인 인스턴스
    
    [사용 예시]
    model = get_pipeline('arch0', config)
    model = get_pipeline('arch2', config)
    model = get_pipeline('arch4', config)
    model = get_pipeline('arch5b', config)
    """
    pipelines = {
        'arch0': Arch0Sequential,
        'arch0_sequential': Arch0Sequential,
        'arch2': Arch2SoftGate,
        'arch2_softgate': Arch2SoftGate,
        'arch4': Arch4Adaptive,
        'arch4_adaptive': Arch4Adaptive,
        'arch5b': Arch5BFusion,
        'arch5b_fusion': Arch5BFusion,
    }
    
    arch_name_lower = arch_name.lower()
    if arch_name_lower not in pipelines:
        available = list(pipelines.keys())
        raise ValueError(f"Unknown architecture: {arch_name}. Available: {available}")
    
    return pipelines[arch_name_lower](config)


__all__ = [
    "BasePipeline",
    "Arch0Sequential",
    "create_arch0_pipeline",
    "Arch2SoftGate",
    "Arch4Adaptive",
    "Arch5BFusion",
    "get_pipeline",
]