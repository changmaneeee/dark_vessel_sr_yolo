"""
=============================================================================
base_pipeline.py - 파이프라인 공통 인터페이스
=============================================================================
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class BasePipeline(nn.Module, ABC):
    """
    파이프라인 공통 인터페이스
    
    [구현 클래스]
    - Arch0Sequential
    - Arch2SoftGate
    - Arch4Adaptive
    - Arch5BFusion
    """
    
    def __init__(self, config: Any):
        super().__init__()
        
        self.config = config
        
        # Device
        self.device = getattr(config, 'device', 'cuda')
        if isinstance(self.device, str):
            self.device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        
        # Loss weights (training config에서)
        training_config = getattr(config, 'training', config)
        self._sr_weight = getattr(training_config, 'sr_weight', 0.0)
        self._det_weight = getattr(training_config, 'det_weight', 1.0)
    
    @abstractmethod
    def forward(
        self,
        lr_image: torch.Tensor,
        **kwargs
    ) -> Tuple[Any, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass
        
        Args:
            lr_image: LR 입력 이미지
            **kwargs: 추가 인자
        
        Returns:
            outputs: 탐지 결과
            features: (선택) 중간 feature들
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        outputs: Any,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Loss 계산
        
        Args:
            outputs: forward()의 출력
            targets: GT
            **kwargs: 추가 인자
        
        Returns:
            loss_dict: {total, det_loss, sr_loss, ...}
        """
        pass
    
    @torch.no_grad()
    def inference(
        self,
        lr_image: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """
        추론 모드
        
        Args:
            lr_image: LR 입력 이미지
            **kwargs: conf_threshold, iou_threshold 등
        
        Returns:
            결과 딕셔너리
        """
        self.eval()
        outputs, features = self.forward(lr_image, return_features=True)
        return {
            'outputs': outputs,
            'features': features
        }
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """아키텍처 정보"""
        return {
            'name': self.__class__.__name__,
            'sr_weight': self._sr_weight,
            'det_weight': self._det_weight,
            'device': str(self.device)
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """파라미터 수"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }