"""
=============================================================================
soft_gate.py - Lightweight Gate for Arch2
=============================================================================
이미지 복잡도를 판단하여 SR 적용 여부 결정

[Gate 판단 기준]
- 높은 score (>threshold): 복잡한 이미지 → SR 필요
- 낮은 score (<=threshold): 단순한 이미지 → Bypass 가능

[구성]
- LightweightGate: CNN 기반 경량 게이트 (~50K params)
- EdgeBasedGate: 엣지 기반 휴리스틱 게이트 (학습 불필요)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class LightweightGate(nn.Module):
    """
    경량 CNN 기반 Gate
    
    이미지의 복잡도/품질을 판단하여 0~1 score 출력
    
    Args:
        in_channels: 입력 채널 (기본 3)
        hidden_channels: 히든 채널 수
    
    Output:
        score: [B, 1] - SR 필요도 (0: bypass, 1: SR 필요)
    """
    
    def __init__(self, in_channels: int = 3, hidden_channels: int = 32):
        super().__init__()
        
        # 특징 추출
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
        # 파라미터 수 출력
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[LightweightGate] Parameters: {total_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 입력 이미지
        
        Returns:
            score: [B, 1] SR 필요도
        """
        features = self.features(x)
        score = self.classifier(features)
        return score


class EdgeBasedGate(nn.Module):
    """
    엣지 기반 휴리스틱 Gate (학습 불필요)
    
    Sobel 필터로 엣지 강도를 측정하여 이미지 복잡도 판단
    - 엣지가 약함 → 저해상도/블러 → SR 필요
    - 엣지가 강함 → 고해상도/선명 → Bypass 가능
    
    Note: 이 게이트는 학습이 필요 없어서 간단한 baseline으로 사용 가능
    """
    
    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold
        
        # Sobel 필터 (고정)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 입력 이미지
        
        Returns:
            score: [B, 1] SR 필요도 (엣지가 약하면 높은 값)
        """
        # Grayscale 변환
        if x.size(1) == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x[:, 0:1]
        
        # Sobel 필터 적용
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # 엣지 강도
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        
        # 평균 엣지 강도 (낮을수록 SR 필요)
        avg_edge = edge_magnitude.mean(dim=[2, 3], keepdim=True)
        
        # Score 변환 (엣지가 약하면 높은 score)
        # score = 1 - normalized_edge
        max_edge = avg_edge.max() + 1e-8
        score = 1.0 - (avg_edge / max_edge)
        
        return score.view(-1, 1)


class SoftGateModule(nn.Module):
    """
    Soft Gate 통합 모듈
    
    Gate score에 따라 SR과 Bypass를 soft하게 결합
    
    Args:
        gate: Gate 네트워크
        hard_decision: True면 binary 결정, False면 soft blending
    """
    
    def __init__(
        self,
        gate: Optional[nn.Module] = None,
        hard_decision: bool = True,
        threshold: float = 0.5
    ):
        super().__init__()
        
        self.gate = gate if gate else LightweightGate()
        self.hard_decision = hard_decision
        self.threshold = threshold
    
    def forward(
        self,
        lr_image: torch.Tensor,
        sr_image: torch.Tensor,
        bypass_image: torch.Tensor
    ) -> tuple:
        """
        Args:
            lr_image: LR 입력 (gate 판단용)
            sr_image: SR 처리된 이미지
            bypass_image: Bypass (upscale만) 이미지
        
        Returns:
            output: 최종 출력 이미지
            gate_score: Gate score
        """
        # Gate score 계산
        gate_score = self.gate(lr_image)
        
        if self.hard_decision:
            # Binary 결정
            mask = (gate_score > self.threshold).float()
            output = mask * sr_image + (1 - mask) * bypass_image
        else:
            # Soft blending
            output = gate_score * sr_image + (1 - gate_score) * bypass_image
        
        return output, gate_score


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Gate Module Test")
    print("=" * 60)
    
    # 테스트 입력
    x = torch.randn(2, 3, 64, 64)
    
    # LightweightGate 테스트
    print("\n[LightweightGate]")
    gate = LightweightGate()
    score = gate(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {score.shape}")
    print(f"  Scores: {score.squeeze().tolist()}")
    
    # EdgeBasedGate 테스트
    print("\n[EdgeBasedGate]")
    edge_gate = EdgeBasedGate()
    score = edge_gate(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {score.shape}")
    print(f"  Scores: {score.squeeze().tolist()}")
    
    # SoftGateModule 테스트
    print("\n[SoftGateModule]")
    module = SoftGateModule(hard_decision=True)
    sr_img = torch.randn(2, 3, 256, 256)
    bypass_img = torch.randn(2, 3, 256, 256)
    output, gate_score = module(x, sr_img, bypass_img)
    print(f"  Output: {output.shape}")
    print(f"  Gate scores: {gate_score.squeeze().tolist()}")
    
    print("\n✓ Test completed!")