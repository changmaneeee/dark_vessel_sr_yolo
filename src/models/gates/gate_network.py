"""
=============================================================================
gate_network.py - Lightweight Gate Networks for Arch2
=============================================================================

SR 적용 여부를 결정하는 경량 게이트 네트워크
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LightweightGate(nn.Module):
    """
    Lightweight Gate Network V1
    
    [구조]
    Input [B, 3, H, W]
        ↓
    Conv 3×3, stride=4 → [B, 32, H/4, W/4]
        ↓
    Conv 3×3, stride=4 → [B, 64, H/16, W/16]
        ↓
    Global Average Pooling → [B, 64]
        ↓
    FC → Sigmoid → gate ∈ [0, 1]
    
    [파라미터]
    약 50K 파라미터 (매우 경량)
    
    [출력 의미]
    gate ≈ 1.0: "이 이미지는 SR이 필요함" (품질 나쁨)
    gate ≈ 0.0: "이 이미지는 SR 불필요" (품질 좋음)
    """
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 입력 → 1/4 크기
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2: 1/4 → 1/16 크기
            nn.Conv2d(32, 64, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LR 이미지 [B, 3, H, W]
        
        Returns:
            gate: [B, 1] 범위 [0, 1]
        """
        feat = self.features(x)
        feat = self.pool(feat)
        feat = feat.view(feat.size(0), -1)
        gate = self.classifier(feat)
        return gate


class LightweightGateV2(nn.Module):
    """
    Lightweight Gate Network V2 (더 경량)
    
    [개선점]
    - Depthwise Separable Conv 사용
    - 약 25K 파라미터
    - 더 빠른 추론
    """
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=4, 
                     padding=2, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Depthwise
            nn.Conv2d(16, 16, kernel_size=5, stride=4, padding=2, groups=16),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Pointwise
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.pool(feat)
        feat = feat.view(feat.size(0), -1)
        gate = self.classifier(feat)
        return gate


class SpatialGate(nn.Module):
    """
    Spatial-aware Gate Network
    
    이미지의 각 영역마다 다른 gate 값을 출력
    (고급 버전, 필요시 사용)
    """
    
    def __init__(self, in_channels: int = 3, out_size: int = 8):
        super().__init__()
        
        self.out_size = out_size
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.gate_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            global_gate: [B, 1] 전역 gate
            spatial_gate: [B, 1, out_size, out_size] 공간별 gate
        """
        feat = self.features(x)
        spatial_gate = self.gate_conv(feat)
        spatial_gate = F.interpolate(
            spatial_gate, size=(self.out_size, self.out_size),
            mode='bilinear', align_corners=False
        )
        global_gate = spatial_gate.mean(dim=[2, 3], keepdim=False)
        return global_gate, spatial_gate


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("Gate Network 테스트")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(2, 3, 192, 192, device=device)
    
    # V1
    gate_v1 = LightweightGate().to(device)
    out_v1 = gate_v1(x)
    params_v1 = sum(p.numel() for p in gate_v1.parameters())
    print(f"V1: output={out_v1.shape}, params={params_v1:,}")
    
    # V2
    gate_v2 = LightweightGateV2().to(device)
    out_v2 = gate_v2(x)
    params_v2 = sum(p.numel() for p in gate_v2.parameters())
    print(f"V2: output={out_v2.shape}, params={params_v2:,}")
    
    # Spatial
    gate_spatial = SpatialGate().to(device)
    global_g, spatial_g = gate_spatial(x)
    params_spatial = sum(p.numel() for p in gate_spatial.parameters())
    print(f"Spatial: global={global_g.shape}, spatial={spatial_g.shape}, params={params_spatial:,}")
    
    print("\n✓ 테스트 완료!")