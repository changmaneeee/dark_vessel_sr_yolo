"""RFDN: Residual Feature Distillation Network

Lightweight SR model with residual feature distillation.
Reference: "Residual Feature Distillation Network for Lightweight Image Super-Resolution"
"""

# =========================================================================
# License: MIT License
# Original Author: njulj (https://github.com/njulj/rfdn)
# Adapted by: AIS-SAT-PIPELINE Team for VLEO Ship Detection
# =========================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from base_sr import BaseSRModel


# =============================================================================
# Helper Functions
# =============================================================================

def conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    padding: int = None
) -> nn.Conv2d:

    if padding is None:
        
        padding = int(((kernel_size - 1) / 2) * dilation)
    
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias
    )


# =============================================================================
# ESA (Enhanced Spatial Attention) Module
# =============================================================================

class ESA(nn.Module):
 
    
    def __init__(self, n_feats: int, conv=nn.Conv2d):

        super(ESA, self).__init__()
        
        
        f = n_feats // 4  # 50 → 12
        
        
        self.conv1 = conv(n_feats, f, kernel_size=1)
        
        self.conv_f = conv(f, f, kernel_size=1)

        self.conv_max = conv(f, f, kernel_size=3, padding=1)

        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)

        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        
        self.conv3_up = conv(f, f, kernel_size=3, padding=1)

        self.conv4 = conv(f, n_feats, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        c1_ = self.conv1(x)  # [B, C/4, H, W]
        c1 = self.conv2(c1_)  # [B, C/4, H', W']
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_up(c3)
        
        c3 = F.interpolate(
            c3, 
            size=(x.size(2), x.size(3)),  
            mode='bilinear', 
            align_corners=False
        )

        cf = self.conv_f(c1_)  
        c4 = self.conv4(c3 + cf)  
        m = self.sigmoid(c4)  
        return x * m


# =============================================================================
# RFDB (Residual Feature Distillation Block)
# =============================================================================

class RFDB(nn.Module):
    
    def __init__(self, in_channels: int, distillation_rate: float = 0.25):

        super(RFDB, self).__init__()

        self.dc = self.distilled_channels = int(in_channels * distillation_rate)  # 12
        self.rc = self.remaining_channels = in_channels # 50
        
        # Stage 1
        self.c1_d = conv_layer(in_channels, self.dc, kernel_size=1)  # 50→12 
        self.c1_r = conv_layer(in_channels, self.rc, kernel_size=3)  # 50→38 
        
        # Stage 2
        self.c2_d = conv_layer(self.rc, self.dc, kernel_size=1)  # 38→12
        self.c2_r = conv_layer(self.rc, self.rc, kernel_size=3)  # 38→38
        
        # Stage 3
        self.c3_d = conv_layer(self.rc, self.dc, kernel_size=1)  # 38→12
        self.c3_r = conv_layer(self.rc, self.rc, kernel_size=3)  # 38→38
        
        # Stage 4
        self.c4 = conv_layer(self.rc, self.dc, kernel_size=3)  # 38→12
        
       
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = conv_layer(self.dc * 4, in_channels, kernel_size=1)
        self.esa = ESA(in_channels, nn.Conv2d)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # Stage 1
        distilled_c1 = self.act(self.c1_d(input))  # [B, 12, H, W]
        r_c1 = self.act(self.c1_r(input) + input)  # [B, 38, H, W] + residual
        
        # Stage 2
        distilled_c2 = self.act(self.c2_d(r_c1))   # [B, 12, H, W]
        r_c2 = self.act(self.c2_r(r_c1) + r_c1)    # [B, 38, H, W]
        
        # Stage 3
        distilled_c3 = self.act(self.c3_d(r_c2))   # [B, 12, H, W]
        r_c3 = self.act(self.c3_r(r_c2) + r_c2)    # [B, 38, H, W]
        
        # Stage 4
        r_c4 = self.act(self.c4(r_c3))             # [B, 12, H, W]
        
        # Concatenate all distilled features
        # [B, 12, H, W] × 4 → [B, 48, H, W]
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        
        # Feature aggregation + ESA
        out_fused = self.esa(self.c5(out))  # [B, 50, H, W]
        
        # Global residual connection
        return out_fused + input


# =============================================================================
# PixelShuffle Upsampler
# =============================================================================

class PixelShufflePack(nn.Module):
   
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        upscale_factor: int = 4
    ):

        super(PixelShufflePack, self).__init__()

        self.conv = conv_layer(
            in_channels, 
            out_channels * (upscale_factor ** 2),
            kernel_size=3
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv(x)           # [B, 48, 192, 192]
        x = self.pixel_shuffle(x)  # [B, 3, 768, 768]
        return x


# =============================================================================
# RFDN Main Model
# =============================================================================

class RFDN(BaseSRModel):
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        nf: int = 50,
        num_modules: int = 4,
        upscale: int = 4
    ):
        super(RFDN, self).__init__(
            scale_factor=upscale,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=nf  # Arch 5-B Fusion에서 사용
        )
        
        # 설정 저장
        self.nf = nf
        self.num_modules = num_modules
        
        # =====================================================================
        # Encoder
        # =====================================================================
        
        self.fea_conv = conv_layer(in_channels, nf, kernel_size=3)

        self.B1 = RFDB(in_channels=nf)
        self.B2 = RFDB(in_channels=nf)
        self.B3 = RFDB(in_channels=nf)
        self.B4 = RFDB(in_channels=nf)

        self.c = conv_layer(nf * num_modules, nf, kernel_size=1)

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)
        
        # =====================================================================
        # Decoder
        # =====================================================================

        self.upsampler = PixelShufflePack(nf, out_channels, upscale_factor=upscale)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
 
        out_fea = self.fea_conv(x)  # [B, 50, 192, 192]
        
        # Deep feature extraction through RFDB blocks
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        
        # Feature aggregation
        # Concatenate: [B, 50*4, H, W] = [B, 200, 192, 192]
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        
        # Global residual learning
        out_lr = self.LR_conv(out_B) + out_fea
        
        return out_lr  # [B, 50, 192, 192]
    
    def forward_reconstruct(self, features: torch.Tensor) -> torch.Tensor:
 
        hr_image = self.upsampler(features)
        return hr_image
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        features = self.forward_features(x)
        hr_image = self.forward_reconstruct(features)
        return hr_image
    
    def get_feature_info(self) -> Dict[str, Any]:
 
        info = super().get_feature_info()
        info.update({
            'num_modules': self.num_modules,
            'nf': self.nf
        })
        return info


# =============================================================================
# 테스트 코드
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RFDN 테스트")
    print("=" * 60)
    
    # 모델 생성
    model = RFDN(nf=50, num_modules=4, upscale=4)
    print(f"\n모델 정보:")
    #print(f"  - 파라미터 수: {model.count_parameters()}")
    print(f"  - Feature 정보: {model.get_feature_info()}")
    
    # 더미 입력 생성
    batch_size = 2
    lr_image = torch.randn(batch_size, 3, 192, 192)
    print(f"\n입력 shape: {lr_image.shape}")
    
    # Forward 테스트
    with torch.no_grad():
        # 전체 SR
        hr_image = model(lr_image)
        print(f"HR 출력 shape: {hr_image.shape}")
        
        # Feature만 추출 (Arch 5-B용)
        features = model.forward_features(lr_image)
        print(f"Feature shape: {features.shape}")
        
        # Feature에서 HR 복원
        hr_from_feat = model.forward_reconstruct(features)
        print(f"Feature→HR shape: {hr_from_feat.shape}")
    
    print("\n✓ RFDN 테스트 완료!")







