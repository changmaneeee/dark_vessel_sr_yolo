"""
=============================================================================
attention_fusion.py - Multi-scale Attention Fusion for Arch 5-B
=============================================================================

[역할]
SR Feature와 YOLO Feature를 Attention 기반으로 융합

[Arch 5-B 구조]

LR Image
    │
    ├───────────────────────────────────────┐
    │                                        │
    ▼                                        ▼
┌─────────┐                           ┌───────────┐
│  RFDN   │ (SR Encoder)              │   YOLO    │ (Backbone+Neck)
│ forward_│                           │ extract_  │
│ features│                           │ features  │
└────┬────┘                           └─────┬─────┘
     │                                      │
     │ [B, 50, H, W]                        │ P3: [B, C3, H/8, W/8]
     │                                      │ P4: [B, C4, H/16, W/16]
     │                                      │ P5: [B, C5, H/32, W/32]
     │                                      │
     └──────────────┬───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ MultiScaleAttention   │
        │      Fusion           │
        │                       │
        │ 1. SR → P3/P4/P5 size │
        │ 2. Channel alignment  │
        │ 3. Cross attention    │
        │ 4. Feature fusion     │
        └───────────┬───────────┘
                    │
                    ▼
         Fused Features (P3', P4', P5')
                    │
                    ▼
              YOLO Detect Head
                    │
                    ▼
             Detection Results

[Fusion 방법]
1. Channel Projection: SR feature 채널을 YOLO feature 채널에 맞춤
2. Spatial Alignment: SR feature 크기를 각 scale에 맞춤
3. Attention: SR feature로 YOLO feature를 강화
4. Residual: 원본 YOLO feature + Fused feature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (SE Block 기반)
    
    "어떤 채널이 중요한가?"를 학습
    
    [구조]
    Input [B, C, H, W]
        │
        ├── Global Avg Pool → [B, C, 1, 1]
        │           │
        │           ▼
        │       FC → ReLU → FC → Sigmoid
        │           │
        │           ▼ [B, C, 1, 1] (attention weights)
        │
        └── × ─────────────────┘
                    │
                    ▼
              Output [B, C, H, W]
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: 입력 채널 수
            reduction: 채널 축소 비율
        """
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    
    "어떤 위치가 중요한가?"를 학습
    
    [구조]
    Input [B, C, H, W]
        │
        ├── Channel-wise Max → [B, 1, H, W]
        ├── Channel-wise Avg → [B, 1, H, W]
        │           │
        │           ▼ Concat [B, 2, H, W]
        │           │
        │       Conv 7×7 → Sigmoid
        │           │
        │           ▼ [B, 1, H, W] (attention map)
        │
        └── × ─────────────────┘
                    │
                    ▼
              Output [B, C, H, W]
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    
    Channel Attention + Spatial Attention 결합
    
    [논문]
    "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CrossAttention(nn.Module):
    """
    Cross Attention Module
    
    SR feature를 Query로, YOLO feature를 Key/Value로 사용
    SR 정보로 YOLO feature를 강화
    
    [구조]
    SR Feature → Q
    YOLO Feature → K, V
    
    Attention(Q, K, V) = softmax(QK^T / √d) × V
    """
    
    def __init__(
        self,
        sr_channels: int,
        yolo_channels: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            sr_channels: SR feature 채널 수
            yolo_channels: YOLO feature 채널 수
            num_heads: Attention head 수
            dropout: Dropout 비율
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = yolo_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query from SR, Key/Value from YOLO
        self.q_proj = nn.Conv2d(sr_channels, yolo_channels, 1)
        self.k_proj = nn.Conv2d(yolo_channels, yolo_channels, 1)
        self.v_proj = nn.Conv2d(yolo_channels, yolo_channels, 1)
        self.out_proj = nn.Conv2d(yolo_channels, yolo_channels, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        sr_feat: torch.Tensor,
        yolo_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sr_feat: SR feature [B, C_sr, H, W]
            yolo_feat: YOLO feature [B, C_yolo, H, W]
        
        Returns:
            attended feature [B, C_yolo, H, W]
        """
        B, _, H, W = yolo_feat.shape
        
        # Spatial size 맞추기
        if sr_feat.shape[2:] != yolo_feat.shape[2:]:
            sr_feat = F.interpolate(
                sr_feat, size=(H, W),
                mode='bilinear', align_corners=False
            )
        
        # Q, K, V projection
        q = self.q_proj(sr_feat)  # [B, C_yolo, H, W]
        k = self.k_proj(yolo_feat)
        v = self.v_proj(yolo_feat)
        
        # Reshape for multi-head attention
        # [B, num_heads, head_dim, H*W]
        q = q.view(B, self.num_heads, self.head_dim, H * W)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, self.head_dim, H * W)
        
        # Attention: [B, num_heads, H*W, H*W]
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output: [B, num_heads, head_dim, H*W]
        out = torch.matmul(v, attn.transpose(-2, -1))
        
        # Reshape back
        out = out.view(B, -1, H, W)
        out = self.out_proj(out)
        
        return out


class SingleScaleFusion(nn.Module):
    """
    단일 스케일 Fusion 모듈
    
    SR feature와 YOLO feature를 하나의 스케일에서 융합
    
    [구조]
    SR Feature ─────────────────────────────┐
         │                                   │
         ▼ Projection (채널 맞춤)            │
         │                                   │
         └─────────► Cross Attention ◄───────┘
                          │                  
                          ▼                  
                        CBAM                 
                          │                  
                          ▼                  
                    + (Residual)             
                          │                  
              YOLO Feature                   
                          │                  
                          ▼                  
                   Fused Feature             
    """
    
    def __init__(
        self,
        sr_channels: int,
        yolo_channels: int,
        use_cross_attention: bool = True,
        use_cbam: bool = True,
        num_heads: int = 4
    ):
        """
        Args:
            sr_channels: SR feature 채널 수
            yolo_channels: YOLO feature 채널 수 (출력 채널)
            use_cross_attention: Cross attention 사용 여부
            use_cbam: CBAM 사용 여부
            num_heads: Cross attention head 수
        """
        super().__init__()
        
        self.sr_channels = sr_channels
        self.yolo_channels = yolo_channels
        
        # SR channel projection
        self.sr_proj = nn.Sequential(
            nn.Conv2d(sr_channels, yolo_channels, 1, bias=False),
            nn.BatchNorm2d(yolo_channels),
            nn.ReLU(inplace=True)
        )
        
        # Cross attention (선택)
        self.cross_attn = None
        if use_cross_attention:
            self.cross_attn = CrossAttention(
                sr_channels=yolo_channels,  # projection 후
                yolo_channels=yolo_channels,
                num_heads=num_heads
            )
        
        # CBAM (선택)
        self.cbam = CBAM(yolo_channels) if use_cbam else None
        
        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(yolo_channels * 2, yolo_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(yolo_channels),
            nn.ReLU(inplace=True)
        )
        
        # Learnable fusion weight
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        sr_feat: torch.Tensor,
        yolo_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sr_feat: SR feature [B, C_sr, H_sr, W_sr]
            yolo_feat: YOLO feature [B, C_yolo, H, W]
        
        Returns:
            fused feature [B, C_yolo, H, W]
        """
        H, W = yolo_feat.shape[2:]
        
        # SR feature projection + resize
        sr_proj = self.sr_proj(sr_feat)
        if sr_proj.shape[2:] != (H, W):
            sr_proj = F.interpolate(
                sr_proj, size=(H, W),
                mode='bilinear', align_corners=False
            )
        
        # Cross attention (SR로 YOLO 강화)
        if self.cross_attn is not None:
            attended = self.cross_attn(sr_proj, yolo_feat)
        else:
            attended = sr_proj
        
        # CBAM
        if self.cbam is not None:
            attended = self.cbam(attended)
        
        # Concatenate and fuse
        concat = torch.cat([attended, yolo_feat], dim=1)
        fused = self.fusion_conv(concat)
        
        # Residual connection with learnable weight
        alpha = torch.sigmoid(self.alpha)
        output = alpha * fused + (1 - alpha) * yolo_feat
        
        return output


class MultiScaleAttentionFusion(nn.Module):
    """
    Multi-scale Attention Fusion Module for Arch 5-B
    
    SR feature를 P3, P4, P5 각각의 스케일에서 YOLO feature와 융합
    
    [입력]
    - SR Features: [B, C_sr, H, W] (단일 스케일 또는 다중 스케일)
    - YOLO Features: {'p3': ..., 'p4': ..., 'p5': ...}
    
    [출력]
    - Fused Features: {'p3': ..., 'p4': ..., 'p5': ...}
    - 각 feature는 원본 YOLO feature와 같은 shape
    
    [사용 예시]
    fusion = MultiScaleAttentionFusion(
        sr_channels=50,
        yolo_channels={'p3': 128, 'p4': 256, 'p5': 512}
    )
    
    fused = fusion(sr_feat, yolo_feats)
    # fused['p3'], fused['p4'], fused['p5']를 YOLO Detect head에 전달
    """
    
    def __init__(
        self,
        sr_channels: int,
        yolo_channels: Dict[str, int],
        use_cross_attention: bool = True,
        use_cbam: bool = True,
        num_heads: int = 4
    ):
        """
        Args:
            sr_channels: SR feature 채널 수
            yolo_channels: 각 스케일의 YOLO feature 채널 수
                          {'p3': 128, 'p4': 256, 'p5': 512}
            use_cross_attention: Cross attention 사용 여부
            use_cbam: CBAM 사용 여부
            num_heads: Cross attention head 수
        """
        super().__init__()
        
        self.sr_channels = sr_channels
        self.yolo_channels = yolo_channels
        self.scales = list(yolo_channels.keys())
        
        # 각 스케일별 fusion 모듈
        self.fusion_modules = nn.ModuleDict()
        for scale, channels in yolo_channels.items():
            self.fusion_modules[scale] = SingleScaleFusion(
                sr_channels=sr_channels,
                yolo_channels=channels,
                use_cross_attention=use_cross_attention,
                use_cbam=use_cbam,
                num_heads=min(num_heads, channels // 32)  # 채널 수에 따라 조정
            )
        
        print(f"[MultiScaleAttentionFusion] Initialized for scales: {self.scales}")
        print(f"[MultiScaleAttentionFusion] SR channels: {sr_channels}")
        print(f"[MultiScaleAttentionFusion] YOLO channels: {yolo_channels}")
    
    def forward(
        self,
        sr_features: torch.Tensor,
        yolo_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            sr_features: SR encoder 출력 [B, C_sr, H, W]
            yolo_features: YOLO multi-scale features
                          {'p3': [B, C3, H3, W3], 'p4': ..., 'p5': ...}
        
        Returns:
            fused_features: 융합된 multi-scale features
                           {'p3': [B, C3, H3, W3], 'p4': ..., 'p5': ...}
        """
        fused_features = {}
        
        for scale in self.scales:
            if scale in yolo_features:
                yolo_feat = yolo_features[scale]
                fused = self.fusion_modules[scale](sr_features, yolo_feat)
                fused_features[scale] = fused
        
        return fused_features
    
    def get_output_channels(self) -> Dict[str, int]:
        """출력 채널 수 반환 (YOLO feature와 동일)"""
        return self.yolo_channels.copy()


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Attention Fusion 테스트")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. 개별 모듈 테스트
    print("\n1. 개별 Attention 모듈 테스트")
    
    x = torch.randn(2, 64, 32, 32, device=device)
    
    # Channel Attention
    ca = ChannelAttention(64).to(device)
    out = ca(x)
    print(f"   ChannelAttention: {x.shape} → {out.shape}")
    
    # Spatial Attention
    sa = SpatialAttention().to(device)
    out = sa(x)
    print(f"   SpatialAttention: {x.shape} → {out.shape}")
    
    # CBAM
    cbam = CBAM(64).to(device)
    out = cbam(x)
    print(f"   CBAM: {x.shape} → {out.shape}")
    
    # 2. Single Scale Fusion 테스트
    print("\n2. SingleScaleFusion 테스트")
    
    sr_feat = torch.randn(2, 50, 192, 192, device=device)  # RFDN output
    yolo_feat = torch.randn(2, 128, 80, 80, device=device)  # P3
    
    fusion = SingleScaleFusion(
        sr_channels=50,
        yolo_channels=128,
        use_cross_attention=True,
        use_cbam=True
    ).to(device)
    
    fused = fusion(sr_feat, yolo_feat)
    print(f"   SR: {sr_feat.shape}")
    print(f"   YOLO: {yolo_feat.shape}")
    print(f"   Fused: {fused.shape}")
    
    # 3. Multi-scale Fusion 테스트
    print("\n3. MultiScaleAttentionFusion 테스트")
    
    sr_feat = torch.randn(2, 50, 192, 192, device=device)
    yolo_feats = {
        'p3': torch.randn(2, 128, 80, 80, device=device),
        'p4': torch.randn(2, 256, 40, 40, device=device),
        'p5': torch.randn(2, 512, 20, 20, device=device),
    }
    
    multi_fusion = MultiScaleAttentionFusion(
        sr_channels=50,
        yolo_channels={'p3': 128, 'p4': 256, 'p5': 512}
    ).to(device)
    
    fused_feats = multi_fusion(sr_feat, yolo_feats)
    
    print(f"   SR Feature: {sr_feat.shape}")
    print(f"   YOLO Features:")
    for k, v in yolo_feats.items():
        print(f"      {k}: {v.shape}")
    print(f"   Fused Features:")
    for k, v in fused_feats.items():
        print(f"      {k}: {v.shape}")
    
    # 4. Gradient flow 테스트
    print("\n4. Gradient flow 테스트")
    
    sr_feat.requires_grad = True
    for feat in yolo_feats.values():
        feat.requires_grad = True
    
    fused_feats = multi_fusion(sr_feat, yolo_feats)
    loss = sum(f.sum() for f in fused_feats.values())
    loss.backward()
    
    print(f"   SR gradient exists: {sr_feat.grad is not None}")
    print(f"   YOLO p3 gradient exists: {yolo_feats['p3'].grad is not None}")
    print(f"   ✓ Gradient flows through fusion module!")
    
    print("\n" + "=" * 70)
    print("✓ 모든 Fusion 테스트 완료!")
    print("=" * 70)