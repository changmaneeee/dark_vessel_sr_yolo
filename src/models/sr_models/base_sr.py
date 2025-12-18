"""
=============================================================================
base_sr.py - SR 모델 공통 인터페이스
=============================================================================
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseSRModel(nn.Module, ABC):
    """
    SR 모델 공통 인터페이스
    
    [구현해야 하는 메서드]
    - forward(x): 전체 SR 수행
    - forward_features(x): Feature만 추출 (Arch5-B용)
    - forward_reconstruct(features): Feature에서 HR 복원
    
    [제공되는 메서드]
    - count_parameters(): 파라미터 수
    - get_feature_info(): Feature 정보
    """
    
    def __init__(
        self,
        scale_factor: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        feature_channels: int = 64
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_channels = feature_channels
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        전체 SR 수행: LR → HR
        
        Args:
            x: LR 이미지 [B, 3, H, W]
        
        Returns:
            HR 이미지 [B, 3, H*scale, W*scale]
        """
        pass
    
    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature만 추출 (Arch5-B Fusion용)
        
        Args:
            x: LR 이미지 [B, 3, H, W]
        
        Returns:
            Feature map [B, C, H, W]
        """
        pass
    
    @abstractmethod
    def forward_reconstruct(self, features: torch.Tensor) -> torch.Tensor:
        """
        Feature에서 HR 이미지 복원
        
        Args:
            features: Feature map [B, C, H, W]
        
        Returns:
            HR 이미지 [B, 3, H*scale, W*scale]
        """
        pass
    
    def count_parameters(self) -> Dict[str, int]:
        """파라미터 수 계산"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Feature 정보 반환"""
        return {
            'scale_factor': self.scale_factor,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'feature_channels': self.feature_channels
        }
    
    def freeze(self):
        """모델 전체 freeze"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """모델 전체 unfreeze"""
        for param in self.parameters():
            param.requires_grad = True


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("BaseSRModel은 추상 클래스입니다.")
    print("RFDN, MambaSR 등에서 상속하여 사용합니다.")