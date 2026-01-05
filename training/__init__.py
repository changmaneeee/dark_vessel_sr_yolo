"""
=============================================================================
Arch5B Training Module
=============================================================================
Arch5B (Feature-Level Fusion) 전용 학습 모듈

[가정]
- SR (MambaSR/RFDN): 개별 학습 완료
- YOLO: 개별 학습 완료
- Arch0, 2, 4: 개별 학습된 가중치 조합 → Inference만
- Arch5B: Fusion 모듈 학습 필요 → 이 모듈 사용

[학습 모드]
- scratch: 기본 pretrained로 SR + Fusion 학습
- finetune: 선박 특화 가중치로 Fusion만 학습
"""

from .dataset import SRDetectionDataset, create_dataloader, collate_fn

__all__ = [
    'SRDetectionDataset',
    'create_dataloader',
    'collate_fn'
]