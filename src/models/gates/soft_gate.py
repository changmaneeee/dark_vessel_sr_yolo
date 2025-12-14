##soft gate code

"""
=============================================================================
soft_gate.py - Soft Gate Network for Conditional SR (Arch 2)
=============================================================================

[역할]
입력 이미지의 품질을 분석하여 SR 적용 여부를 결정하는 Gate 네트워크

[핵심 개념]
- gate ∈ [0, 1]
- gate ≈ 1: SR 적용 (품질 나쁜 이미지)
- gate ≈ 0: SR 스킵 (품질 좋은 이미지)

[사용 예시]
gate_net = LightweightGate(in_channels=3, base_channels=32)
gate_value = gate_net(lr_image)  # [B, 1]

soft_gate = SoftGateModule(gate_net, sr_model, upscale=4)
output, gate = soft_gate(lr_image, return_gate=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# =============================================================================
# Depthwise Separable Convolution (경량화)
# =============================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution
    
    [구조]
    일반 Conv: (C_in × C_out × K × K) 파라미터
    DSConv:    (C_in × K × K) + (C_in × C_out) 파라미터
    
    → 약 K² 배 파라미터 감소!
    
    [예시]
    3×3 Conv, 64→128: 64×128×3×3 = 73,728
    DSConv:           64×3×3 + 64×128 = 576 + 8,192 = 8,768
    → 8.4배 감소
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        
        # Depthwise: 각 채널 독립적으로 convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # 핵심! 채널별로 분리
            bias=False
        )
        
        # Pointwise: 1×1 conv로 채널 믹싱
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# =============================================================================
# Lightweight Gate Network V1 (~50K params)
# =============================================================================

class LightweightGate(nn.Module):
    """
    경량 Gate 네트워크
    
    [역할]
    입력 이미지의 "품질 점수"를 예측
    - 블러, 노이즈, 저해상도 정도를 분석
    - 0~1 사이의 gate 값 출력
    
    [구조]
    Input [B, 3, H, W]
        │
        ├── Conv 3×3 (stride=2) ────→ [B, 32, H/2, W/2]
        ├── DSConv (stride=2) ──────→ [B, 64, H/4, W/4]
        ├── DSConv (stride=2) ──────→ [B, 128, H/8, W/8]
        ├── DSConv (stride=2) ──────→ [B, 128, H/16, W/16]
        │
        ├── Global Average Pool ────→ [B, 128, 1, 1]
        │
        ├── FC 128 → 64 → ReLU
        ├── FC 64 → 1 → Sigmoid
        │
        └── Output: gate [B, 1]
    
    [파라미터 수]
    약 50K (RFDN 550K의 ~9%)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Feature extraction
        layers = []
        
        # 첫 번째 레이어: 일반 Conv (입력 채널 처리)
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        ))
        
        # 나머지 레이어: DSConv로 경량화
        current_channels = base_channels
        for i in range(num_layers - 1):
            next_channels = min(current_channels * 2, 128)
            layers.append(DepthwiseSeparableConv(
                current_channels,
                next_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ))
            current_channels = next_channels
        
        self.features = nn.Sequential(*layers)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Gate prediction head
        self.gate_head = nn.Sequential(
            nn.Linear(current_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0~1 출력
        )
        
        # 파라미터 수 출력
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[LightweightGate] Parameters: {total_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 이미지 [B, 3, H, W]
        
        Returns:
            gate: [B, 1] ∈ [0, 1]
        """
        # Feature extraction
        feat = self.features(x)  # [B, C, H', W']
        
        # Global pooling
        feat = self.pool(feat)  # [B, C, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, C]
        
        # Gate prediction
        gate = self.gate_head(feat)  # [B, 1]
        
        return gate


# =============================================================================
# Lightweight Gate Network V2 (~25K params)
# =============================================================================

class LightweightGateV2(nn.Module):
    """
    더 경량화된 Gate 네트워크
    
    [V1 대비 변경]
    - 채널 수 감소
    - 레이어 수 감소
    - MobileNet 스타일
    
    [파라미터 수]
    약 25K (V1의 절반)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 16
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            # Stage 1: 3 → 16
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU6(inplace=True),
            
            # Stage 2: 16 → 32
            DepthwiseSeparableConv(base_channels, base_channels * 2, stride=2),
            
            # Stage 3: 32 → 64
            DepthwiseSeparableConv(base_channels * 2, base_channels * 4, stride=2),
            
            # Stage 4: 64 → 64
            DepthwiseSeparableConv(base_channels * 4, base_channels * 4, stride=2),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.gate_head = nn.Sequential(
            nn.Linear(base_channels * 4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[LightweightGateV2] Parameters: {total_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.pool(feat).view(feat.size(0), -1)
        gate = self.gate_head(feat)
        return gate


# =============================================================================
# Soft Gate Module (Gate + SR 결합)
# =============================================================================

class SoftGateModule(nn.Module):
    """
    Soft Gate + SR 결합 모듈
    
    [동작]
    output = gate × SR(LR) + (1 - gate) × Bilinear_Upsample(LR)
    
    - gate ≈ 1: SR 결과 사용
    - gate ≈ 0: 단순 업샘플 사용 (SR 스킵)
    
    [장점]
    1. 연산량 절약: 품질 좋은 이미지는 SR 스킵
    2. Soft blending: 완전 binary가 아닌 부드러운 전환
    3. End-to-end 학습: Detection loss가 Gate까지 역전파
    """
    
    def __init__(
        self,
        gate_network: nn.Module,
        sr_model: nn.Module,
        upscale: int = 4
    ):
        super().__init__()
        
        self.gate_network = gate_network
        self.sr_model = sr_model
        self.upscale = upscale
    
    def forward(
        self,
        lr_image: torch.Tensor,
        return_gate: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            lr_image: LR 입력 [B, 3, H, W]
            return_gate: gate 값도 반환할지
        
        Returns:
            output: 출력 이미지 [B, 3, H*scale, W*scale]
            gate: (optional) gate 값 [B, 1]
        """
        B = lr_image.size(0)
        
        # 1. Gate 예측
        gate = self.gate_network(lr_image)  # [B, 1]
        
        # 2. SR 수행
        sr_image = self.sr_model(lr_image)  # [B, 3, H*scale, W*scale]
        
        # 3. 단순 업샘플 (bypass path)
        upsampled = F.interpolate(
            lr_image,
            scale_factor=self.upscale,
            mode='bilinear',
            align_corners=False
        )
        
        # 4. Soft blending
        # gate: [B, 1] → [B, 1, 1, 1] for broadcasting
        gate_expanded = gate.view(B, 1, 1, 1)
        output = gate_expanded * sr_image + (1 - gate_expanded) * upsampled
        
        if return_gate:
            return output, gate
        return output, None
    
    def forward_with_intermediates(
        self,
        lr_image: torch.Tensor
    ) -> dict:
        """
        중간 결과도 모두 반환 (디버깅/분석용)
        """
        B = lr_image.size(0)
        
        gate = self.gate_network(lr_image)
        sr_image = self.sr_model(lr_image)
        upsampled = F.interpolate(
            lr_image,
            scale_factor=self.upscale,
            mode='bilinear',
            align_corners=False
        )
        
        gate_expanded = gate.view(B, 1, 1, 1)
        output = gate_expanded * sr_image + (1 - gate_expanded) * upsampled
        
        return {
            'output': output,
            'gate': gate,
            'sr_image': sr_image,
            'upsampled': upsampled,
            'gate_expanded': gate_expanded
        }


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Soft Gate 테스트")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. LightweightGate V1 테스트
    print("\n[1. LightweightGate V1]")
    gate_v1 = LightweightGate(in_channels=3, base_channels=32).to(device)
    
    x = torch.randn(2, 3, 192, 192, device=device)
    gate_val = gate_v1(x)
    print(f"  Input: {x.shape}")
    print(f"  Gate output: {gate_val.shape}")
    print(f"  Gate values: {gate_val.squeeze().tolist()}")
    
    # 2. LightweightGate V2 테스트
    print("\n[2. LightweightGate V2 (더 경량)]")
    gate_v2 = LightweightGateV2(in_channels=3, base_channels=16).to(device)
    
    gate_val = gate_v2(x)
    print(f"  Gate output: {gate_val.shape}")
    print(f"  Gate values: {gate_val.squeeze().tolist()}")
    
    # 3. SoftGateModule 테스트 (더미 SR 모델)
    print("\n[3. SoftGateModule]")
    
    # 간단한 더미 SR 모델
    class DummySR(nn.Module):
        def __init__(self, scale=4):
            super().__init__()
            self.scale = scale
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
        
        def forward(self, x):
            x = self.conv(x)
            return F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
    
    dummy_sr = DummySR(scale=4).to(device)
    soft_gate = SoftGateModule(gate_v1, dummy_sr, upscale=4).to(device)
    
    output, gate = soft_gate(x, return_gate=True)
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Gate: {gate.squeeze().tolist()}")
    
    # 4. Gradient flow 테스트
    print("\n[4. Gradient Flow 테스트]")
    x.requires_grad = True
    output, gate = soft_gate(x, return_gate=True)
    loss = output.mean() + gate.mean()  # 더미 loss
    loss.backward()
    
    gate_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in gate_v1.parameters())
    print(f"  Gate network gradient: {gate_has_grad}")
    
    # 5. 중간 결과 분석
    print("\n[5. 중간 결과 분석]")
    x = torch.randn(2, 3, 192, 192, device=device)
    results = soft_gate.forward_with_intermediates(x)
    
    for key, val in results.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
    
    print("\n" + "=" * 70)
    print("✓ Soft Gate 테스트 완료!")
    print("=" * 70)
