#!/usr/bin/env python
"""
=============================================================================
test_rfdn_residual.py - RFDN Residual 유무 테스트
=============================================================================

[목적]
가중치가 어떤 RFDN 구조로 학습되었는지 확인
- 버전 A: RFDB에 residual 없음 (return out_fused)
- 버전 B: RFDB에 residual 있음 (return out_fused + input)

두 버전으로 테스트해서 PSNR이 높은 쪽이 정답!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import sys

try:
    from skimage.metrics import peak_signal_noise_ratio as calc_psnr
except ImportError:
    def calc_psnr(img1, img2, data_range=1.0):
        mse = np.mean((img1 - img2) ** 2)
        return 10 * np.log10(data_range ** 2 / mse) if mse > 0 else float('inf')


# =============================================================================
# Helper Functions
# =============================================================================

def conv_layer(in_channels, out_channels, kernel_size, stride=1, bias=True):
    padding = (kernel_size - 1) // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)


class ESA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.conv_max = nn.Conv2d(f, f, 3, padding=1)
        self.conv2 = nn.Conv2d(f, f, 3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, 3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, 3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, 1)
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
        return x * self.sigmoid(c4)


# =============================================================================
# RFDB - 버전 A (Residual 없음)
# =============================================================================

class RFDB_NoResidual(nn.Module):
    """RFDB without final residual"""
    def __init__(self, nf=50):
        super().__init__()
        self.dc = nf // 2  # 25
        self.rc = nf       # 50
        
        self.c1_d = conv_layer(nf, self.dc, 1)
        self.c1_r = conv_layer(nf, self.rc, 3)
        self.c2_d = conv_layer(self.rc, self.dc, 1)
        self.c2_r = conv_layer(self.rc, self.rc, 3)
        self.c3_d = conv_layer(self.rc, self.dc, 1)
        self.c3_r = conv_layer(self.rc, self.rc, 3)
        self.c4 = conv_layer(self.rc, self.dc, 3)
        
        self.act = nn.LeakyReLU(0.05, True)
        self.c5 = conv_layer(self.dc * 4, nf, 1)
        self.esa = ESA(nf)

    def forward(self, input):
        d1 = self.act(self.c1_d(input))
        r1 = self.act(self.c1_r(input) + input)
        
        d2 = self.act(self.c2_d(r1))
        r2 = self.act(self.c2_r(r1) + r1)
        
        d3 = self.act(self.c3_d(r2))
        r3 = self.act(self.c3_r(r2) + r2)
        
        r4 = self.act(self.c4(r3))
        
        out = torch.cat([d1, d2, d3, r4], dim=1)
        out_fused = self.esa(self.c5(out))
        
        return out_fused  # ← NO residual


# =============================================================================
# RFDB - 버전 B (Residual 있음)
# =============================================================================

class RFDB_WithResidual(nn.Module):
    """RFDB with final residual (Original RFDN paper)"""
    def __init__(self, nf=50):
        super().__init__()
        self.dc = nf // 2
        self.rc = nf
        
        self.c1_d = conv_layer(nf, self.dc, 1)
        self.c1_r = conv_layer(nf, self.rc, 3)
        self.c2_d = conv_layer(self.rc, self.dc, 1)
        self.c2_r = conv_layer(self.rc, self.rc, 3)
        self.c3_d = conv_layer(self.rc, self.dc, 1)
        self.c3_r = conv_layer(self.rc, self.rc, 3)
        self.c4 = conv_layer(self.rc, self.dc, 3)
        
        self.act = nn.LeakyReLU(0.05, True)
        self.c5 = conv_layer(self.dc * 4, nf, 1)
        self.esa = ESA(nf)

    def forward(self, input):
        d1 = self.act(self.c1_d(input))
        r1 = self.act(self.c1_r(input) + input)
        
        d2 = self.act(self.c2_d(r1))
        r2 = self.act(self.c2_r(r1) + r1)
        
        d3 = self.act(self.c3_d(r2))
        r3 = self.act(self.c3_r(r2) + r2)
        
        r4 = self.act(self.c4(r3))
        
        out = torch.cat([d1, d2, d3, r4], dim=1)
        out_fused = self.esa(self.c5(out))
        
        return out_fused + input  # ← WITH residual


# =============================================================================
# RFDN 모델 (두 버전)
# =============================================================================

class RFDN_Test(nn.Module):
    def __init__(self, nf=50, with_residual=False):
        super().__init__()
        self.nf = nf
        self.with_residual = with_residual
        
        RFDB = RFDB_WithResidual if with_residual else RFDB_NoResidual
        
        self.fea_conv = conv_layer(3, nf, 3)
        self.B1 = RFDB(nf)
        self.B2 = RFDB(nf)
        self.B3 = RFDB(nf)
        self.B4 = RFDB(nf)
        
        self.c = nn.Sequential(
            conv_layer(nf * 4, nf, 1),
            nn.LeakyReLU(0.05, True)
        )
        
        self.LR_conv = conv_layer(nf, nf, 3)
        
        self.upsampler = nn.Sequential(
            conv_layer(nf, 3 * 16, 3),
            nn.PixelShuffle(4)
        )

    def forward(self, x):
        fea = self.fea_conv(x)
        b1 = self.B1(fea)
        b2 = self.B2(b1)
        b3 = self.B3(b2)
        b4 = self.B4(b3)
        out = self.c(torch.cat([b1, b2, b3, b4], dim=1))
        out = self.LR_conv(out) + fea
        return self.upsampler(out)


def load_image(path):
    img = Image.open(path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--lr_image', type=str, required=True)
    parser.add_argument('--hr_image', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"🧪 RFDN Residual 유무 테스트")
    print(f"{'='*70}")
    
    # 가중치 로드
    checkpoint = torch.load(args.weights, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 이미지 로드
    lr = load_image(args.lr_image).to(device)
    hr = load_image(args.hr_image).to(device)
    
    print(f"\n[이미지]")
    print(f"  LR: {args.lr_image}")
    print(f"  HR: {args.hr_image}")
    
    results = []
    
    for with_res, name in [(False, "Version A (No Residual)"), 
                           (True, "Version B (With Residual)")]:
        print(f"\n{'='*70}")
        print(f"📦 {name}")
        print(f"{'='*70}")
        
        # 모델 생성 및 가중치 로드
        model = RFDN_Test(nf=50, with_residual=with_res).to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # 추론
        with torch.no_grad():
            sr = model(lr)
        
        # 크기 맞추기
        if sr.shape[-2:] != hr.shape[-2:]:
            sr = F.interpolate(sr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        
        # 결과 분석
        sr_np = sr.squeeze().cpu().numpy().transpose(1, 2, 0)
        hr_np = hr.squeeze().cpu().numpy().transpose(1, 2, 0)
        
        sr_min, sr_max = sr_np.min(), sr_np.max()
        sr_clamped = np.clip(sr_np, 0, 1)
        psnr = calc_psnr(hr_np, sr_clamped, data_range=1.0)
        
        print(f"  출력 범위: [{sr_min:.3f}, {sr_max:.3f}]")
        print(f"  PSNR: {psnr:.2f} dB")
        
        results.append({
            'name': name,
            'with_residual': with_res,
            'sr_min': sr_min,
            'sr_max': sr_max,
            'psnr': psnr
        })
    
    # 결론
    print(f"\n{'='*70}")
    print(f"🎯 결론")
    print(f"{'='*70}")
    
    best = max(results, key=lambda x: x['psnr'])
    
    print(f"\n  {'버전':<30} {'출력 범위':<25} {'PSNR':<12}")
    print(f"  {'-'*65}")
    for r in results:
        marker = " ✅" if r == best else ""
        print(f"  {r['name']:<30} [{r['sr_min']:.2f}, {r['sr_max']:.2f}]{'':<10} {r['psnr']:.2f} dB{marker}")
    
    print(f"\n  🏆 최적 버전: {best['name']}")
    
    if best['with_residual']:
        print(f"\n  📝 코드 수정 필요:")
        print(f"     rfdn.py의 RFDB.forward() 마지막 줄을")
        print(f"     'return out_fused + input' 로 변경하세요!")
    else:
        print(f"\n  ✅ 현재 코드 (residual 없음)가 맞습니다.")
        print(f"     다른 문제를 찾아야 합니다.")


if __name__ == '__main__':
    main()