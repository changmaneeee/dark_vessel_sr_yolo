"""Pipelines Module

Complete SR-Detection pipelines:
- BasePipeline: Abstract base class
- Arch0Sequential: Sequential SR-then-Detect
- Arch2SoftGate: Soft gate fusion
- Arch4Adaptive: Confidence-adaptive pipeline
- Arch5BFusion: Feature fusion (main architecture)
"""

from src.models.pipelines.base_pipeline import BasePipeline

# TODO: Uncomment when implemented
# from src.models.pipelines.arch0_sequential import Arch0Sequential
# from src.models.pipelines.arch2_softgate import Arch2SoftGate
# from src.models.pipelines.arch4_adaptive import Arch4Adaptive
# from src.models.pipelines.arch5b_fusion import Arch5BFusion

__all__ = [
    "BasePipeline",
    # "Arch0Sequential",
    # "Arch2SoftGate",
    # "Arch4Adaptive",
    # "Arch5BFusion",
]
