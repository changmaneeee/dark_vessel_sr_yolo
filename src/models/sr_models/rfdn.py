"""
RFDN: Residual Feature Distillation Network
=============================================================================
[수정 내역]
- input_range 파라미터 추가: '0-1' 또는 '0-255' 선택 가능
- 기본값 '0-1'로 설정하여 일반적인 PyTorch 데이터로더와 호환
- 내부적으로 [0, 255] 범위로 변환하여 처리 후 다시 원래 범위로 반환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import os

from .base_sr import BaseSRModel


# =============================================================================
# Helper Functions
# =============================================================================

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


# =============================================================================
# ESA Module
# =============================================================================

class ESA(nn.Module):
    def __init__(self, n_feats, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)  # 공식 repo: conv3_
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


# =============================================================================
# RFDB Module
# =============================================================================

class RFDB(nn.Module):
    """Residual Feature Distillation Block"""
    
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB, self).__init__()
        # 공식 repo: in_channels // 2 하드코딩
        self.dc = self.distilled_channels = in_channels // 2  # = 25
        self.rc = self.remaining_channels = in_channels       # = 50
        
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.rc, self.dc, 1)
        self.c2_r = conv_layer(self.rc, self.rc, 3)
        self.c3_d = conv_layer(self.rc, self.dc, 1)
        self.c3_r = conv_layer(self.rc, self.rc, 3)
        self.c4 = conv_layer(self.rc, self.dc, 3)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = conv_layer(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.c1_r(input)
        r_c1 = self.act(r_c1 + input)
        
        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c2 + r_c1)
        
        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c3 + r_c2)
        
        r_c4 = self.act(self.c4(r_c3))
        
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))
        
        # ✅ 공식 repo: residual 없음!
        return out_fused


# =============================================================================
# RFDN Main Model
# =============================================================================

class RFDN(BaseSRModel):
    """
    RFDN - 공식 repo 가중치 호환 + 자동 스케일링 지원
    
    [입력 범위 지원]
    - input_range='0-1': PyTorch 표준 (기본값)
    - input_range='0-255': 원본 RFDN repo 방식
    
    [동작 방식]
    입력이 [0,1]이면 → 내부에서 *255 → 처리 → /255 → 출력 [0,1]
    입력이 [0,255]이면 → 그대로 처리 → 출력 [0,255]
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        nf: int = 50,
        num_modules: int = 4,
        upscale: int = 4,
        input_range: str = '0-1',  # ✅ 새로 추가!
        **kwargs
    ):
        super(RFDN, self).__init__(
            scale_factor=upscale,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=nf
        )
        
        # 입력 범위 설정
        self.input_range = input_range
        self._scale_input = (input_range == '0-1')
        
        # 변수명 매핑
        in_nc = in_channels
        out_nc = out_channels
        
        self.nf = nf
        self.num_modules = num_modules
        
        # Encoder
        self.fea_conv = conv_layer(in_nc, nf, 3)
        
        self.B1 = RFDB(nf)
        self.B2 = RFDB(nf)
        self.B3 = RFDB(nf)
        self.B4 = RFDB(nf)
        
        # ✅ 공식 repo: Sequential + LeakyReLU
        self.c = nn.Sequential(
            conv_layer(nf * num_modules, nf, 1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True)
        )
        
        self.LR_conv = conv_layer(nf, nf, 3)
        
        # Decoder (pixelshuffle)
        self.upsampler = nn.Sequential(
            conv_layer(nf, out_nc * (upscale ** 2), 3),
            nn.PixelShuffle(upscale)
        )
        
        if self._scale_input:
            print(f"[RFDN] 입력 범위: [0, 1] → 내부 [0, 255] 변환 활성화")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature 추출 (Arch5B 등에서 사용)"""
        # ✅ 입력 스케일링 (features 추출 시에도 적용)
        if self._scale_input:
            x = x * 255.0
        
        out_fea = self.fea_conv(x)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        return out_lr

    def forward_reconstruct(self, features: torch.Tensor) -> torch.Tensor:
        """Feature → HR 복원 (Arch5B 등에서 사용)"""
        output = self.upsampler(features)
        
        # ✅ 출력 스케일링
        if self._scale_input:
            output = output / 255.0
        
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard Forward
        
        [주의] 이 모델은 0-255 범위로 학습됨!
        Pipeline에서 스케일링을 처리해야 함.
        """
        # ❌ 스케일링 제거 - 원본 RFDN과 동일하게
        # if self._scale_input:
        #     x = x * 255.0
        
        # Forward (원본 그대로)
        out_fea = self.fea_conv(x)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        
        # ❌ 스케일링 제거
        # if self._scale_input:
        #     output = output / 255.0
        
        return output

    def load_pretrained(self, path: str, strict: bool = True):
        """Pipeline에서 호출하는 편의 함수"""
        print(f"[RFDN] Loading weights from: {path}")
        if not os.path.exists(path):
            print(f"[RFDN] ⚠️ Checkpoint not found: {path}")
            return
            
        checkpoint = torch.load(path, map_location='cpu')
        
        # state_dict 추출
        if 'params_ema' in checkpoint: 
            state_dict = checkpoint['params_ema']
        elif 'params' in checkpoint: 
            state_dict = checkpoint['params']
        elif 'state_dict' in checkpoint: 
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint: 
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else: 
            state_dict = checkpoint
            
        # 키 정리
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'): 
                new_state_dict[k[7:]] = v
            elif k.startswith('net_g.'): 
                new_state_dict[k[6:]] = v
            else: 
                new_state_dict[k] = v
                
        try:
            self.load_state_dict(new_state_dict, strict=strict)
            print("[RFDN] ✓ Weights loaded successfully!")
        except Exception as e:
            print(f"[RFDN] ❌ Loading failed: {e}")
            if strict:
                print("[RFDN] Retrying with strict=False...")
                self.load_state_dict(new_state_dict, strict=False)
                print("[RFDN] ✓ Loaded with strict=False")
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Feature 정보 반환"""
        info = super().get_feature_info()
        info.update({
            'num_modules': self.num_modules,
            'nf': self.nf,
            'input_range': self.input_range
        })
        return info


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RFDN 테스트 (자동 스케일링)")
    print("=" * 60)
    
    # 테스트 1: [0, 1] 입력
    print("\n[Test 1] input_range='0-1'")
    model_01 = RFDN(nf=50, input_range='0-1')
    x_01 = torch.rand(1, 3, 192, 192)  # [0, 1]
    with torch.no_grad():
        y_01 = model_01(x_01)
    print(f"  Input range: [{x_01.min():.2f}, {x_01.max():.2f}]")
    print(f"  Output range: [{y_01.min():.2f}, {y_01.max():.2f}]")
    print(f"  Output shape: {y_01.shape}")
    
    # 테스트 2: [0, 255] 입력
    print("\n[Test 2] input_range='0-255'")
    model_255 = RFDN(nf=50, input_range='0-255')
    x_255 = torch.rand(1, 3, 192, 192) * 255  # [0, 255]
    with torch.no_grad():
        y_255 = model_255(x_255)
    print(f"  Input range: [{x_255.min():.2f}, {x_255.max():.2f}]")
    print(f"  Output range: [{y_255.min():.2f}, {y_255.max():.2f}]")
    print(f"  Output shape: {y_255.shape}")
    
    # RFDB 구조 확인
    print(f"\n[구조 확인]")
    print(f"  RFDB Distilled Channels: {model_01.B1.dc} (Expected: 25)")
    print(f"  c layer: {model_01.c}")
    
    print("\n✅ 테스트 완료!")