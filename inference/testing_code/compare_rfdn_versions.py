#!/usr/bin/env python
"""
=============================================================================
compare_rfdn_versions.py - RFDN 버전 직접 비교
=============================================================================

inference_compare.py의 RFDN (정상 동작) vs 
현재 프로젝트 RFDN (비정상) 비교

같은 이미지, 같은 가중치로 결과 비교!
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
# inference_compare.py에서 복사한 RFDN (정상 동작 버전)
# =============================================================================

def conv_layer_orig(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


class ESA_Orig(nn.Module):
    def __init__(self, n_feats, conv=nn.Conv2d):
        super(ESA_Orig, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
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


class RFDB_Orig(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB_Orig, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        
        self.c1_d = conv_layer_orig(in_channels, self.dc, 1)
        self.c1_r = conv_layer_orig(in_channels, self.rc, 3)
        self.c2_d = conv_layer_orig(self.rc, self.dc, 1)
        self.c2_r = conv_layer_orig(self.rc, self.rc, 3)
        self.c3_d = conv_layer_orig(self.rc, self.dc, 1)
        self.c3_r = conv_layer_orig(self.rc, self.rc, 3)
        self.c4 = conv_layer_orig(self.rc, self.dc, 3)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = conv_layer_orig(self.dc * 4, in_channels, 1)
        self.esa = ESA_Orig(in_channels, nn.Conv2d)

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
        return out_fused  # NO residual


class RFDN_Original(nn.Module):
    """inference_compare.py에서 복사한 RFDN (정상 동작)"""
    
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(RFDN_Original, self).__init__()
        
        self.fea_conv = conv_layer_orig(in_nc, nf, 3)
        
        self.B1 = RFDB_Orig(nf)
        self.B2 = RFDB_Orig(nf)
        self.B3 = RFDB_Orig(nf)
        self.B4 = RFDB_Orig(nf)
        
        self.c = nn.Sequential(
            conv_layer_orig(nf * num_modules, nf, 1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True)
        )
        
        self.LR_conv = conv_layer_orig(nf, nf, 3)
        
        self.upsampler = nn.Sequential(
            conv_layer_orig(nf, out_nc * (upscale ** 2), 3),
            nn.PixelShuffle(upscale)
        )

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output


# =============================================================================
# 테스트 함수
# =============================================================================

def load_image_255(path):
    """[0, 255] uint8 -> [0, 255] float tensor"""
    img = Image.open(path).convert('RGB')
    img_np = np.array(img).astype(np.float32)  # [0, 255] 유지!
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)


def load_image_01(path):
    """[0, 255] uint8 -> [0, 1] float tensor"""
    img = Image.open(path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0  # [0, 1] 정규화
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--lr_image', type=str, required=True)
    parser.add_argument('--hr_image', type=str, required=True)
    parser.add_argument('--project_root', type=str, default='/home/changmin/dark_vessel_sr_yolo')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"🔍 RFDN 버전 직접 비교")
    print(f"{'='*70}")
    print(f"가중치: {args.weights}")
    print(f"LR: {args.lr_image}")
    print(f"HR: {args.hr_image}")
    
    # 가중치 로드
    checkpoint = torch.load(args.weights, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    print(f"\n가중치 키 수: {len(state_dict)}")
    print(f"샘플 키: {list(state_dict.keys())[:3]}")
    
    # =========================================================================
    # Version 1: inference_compare.py RFDN (입력 [0, 255])
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"📦 Version 1: inference_compare.py RFDN")
    print(f"   입력 범위: [0, 255]")
    print(f"{'='*70}")
    
    model_orig = RFDN_Original(in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4)
    model_orig.load_state_dict(state_dict, strict=False)
    model_orig.to(device)
    model_orig.eval()
    
    lr_255 = load_image_255(args.lr_image).to(device)
    hr_255 = load_image_255(args.hr_image).to(device)
    
    with torch.no_grad():
        sr_255 = model_orig(lr_255)
    
    print(f"  입력 범위: [{lr_255.min():.1f}, {lr_255.max():.1f}]")
    print(f"  출력 범위: [{sr_255.min():.1f}, {sr_255.max():.1f}]")
    
    # PSNR (uint8로 변환 후)
    sr_np = sr_255.squeeze().cpu().numpy().transpose(1, 2, 0)
    hr_np = hr_255.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    if sr_np.shape != hr_np.shape:
        import cv2
        sr_np = cv2.resize(sr_np, (hr_np.shape[1], hr_np.shape[0]))
    
    sr_clipped = np.clip(sr_np, 0, 255).astype(np.uint8).astype(np.float32)
    hr_clipped = hr_np.astype(np.uint8).astype(np.float32)
    
    psnr_orig = calc_psnr(hr_clipped, sr_clipped, data_range=255.0)
    print(f"  PSNR: {psnr_orig:.2f} dB")
    
    # =========================================================================
    # Version 2: 프로젝트 RFDN (입력 [0, 1])
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"📦 Version 2: 프로젝트 RFDN (input_range='0-1')")
    print(f"   입력 범위: [0, 1]")
    print(f"{'='*70}")
    
    sys.path.insert(0, args.project_root)
    from src.models.sr_models.rfdn import RFDN as RFDN_Project
    
    model_proj = RFDN_Project(nf=50, input_range='0-1')
    model_proj.load_pretrained(args.weights)
    model_proj.to(device)
    model_proj.eval()
    
    lr_01 = load_image_01(args.lr_image).to(device)
    hr_01 = load_image_01(args.hr_image).to(device)
    
    with torch.no_grad():
        sr_01 = model_proj(lr_01)
    
    print(f"  입력 범위: [{lr_01.min():.3f}, {lr_01.max():.3f}]")
    print(f"  출력 범위: [{sr_01.min():.3f}, {sr_01.max():.3f}]")
    
    # PSNR ([0,1] 범위)
    sr_np_01 = sr_01.squeeze().cpu().numpy().transpose(1, 2, 0)
    hr_np_01 = hr_01.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    if sr_np_01.shape != hr_np_01.shape:
        import cv2
        sr_np_01 = cv2.resize(sr_np_01, (hr_np_01.shape[1], hr_np_01.shape[0]))
    
    sr_clipped_01 = np.clip(sr_np_01, 0, 1)
    psnr_proj = calc_psnr(hr_np_01, sr_clipped_01, data_range=1.0)
    print(f"  PSNR: {psnr_proj:.2f} dB")
    
    # =========================================================================
    # Version 3: 프로젝트 RFDN (입력 [0, 255])
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"📦 Version 3: 프로젝트 RFDN (input_range='0-255')")
    print(f"   입력 범위: [0, 255]")
    print(f"{'='*70}")
    
    model_proj_255 = RFDN_Project(nf=50, input_range='0-255')
    model_proj_255.load_pretrained(args.weights)
    model_proj_255.to(device)
    model_proj_255.eval()
    
    with torch.no_grad():
        sr_proj_255 = model_proj_255(lr_255)
    
    print(f"  입력 범위: [{lr_255.min():.1f}, {lr_255.max():.1f}]")
    print(f"  출력 범위: [{sr_proj_255.min():.1f}, {sr_proj_255.max():.1f}]")
    
    sr_np_proj_255 = sr_proj_255.squeeze().cpu().numpy().transpose(1, 2, 0)
    if sr_np_proj_255.shape != hr_np.shape:
        import cv2
        sr_np_proj_255 = cv2.resize(sr_np_proj_255, (hr_np.shape[1], hr_np.shape[0]))
    
    sr_clipped_proj_255 = np.clip(sr_np_proj_255, 0, 255).astype(np.uint8).astype(np.float32)
    psnr_proj_255 = calc_psnr(hr_clipped, sr_clipped_proj_255, data_range=255.0)
    print(f"  PSNR: {psnr_proj_255:.2f} dB")
    
    # =========================================================================
    # 결과 비교
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"📊 결과 비교")
    print(f"{'='*70}")
    
    print(f"\n{'버전':<45} {'출력 범위':<20} {'PSNR':<10}")
    print("-" * 75)
    print(f"{'V1: inference_compare.py (입력 [0,255])':<45} [{sr_255.min():.1f}, {sr_255.max():.1f}]{'':<5} {psnr_orig:.2f} dB")
    print(f"{'V2: 프로젝트 RFDN (input_range=0-1)':<45} [{sr_01.min():.2f}, {sr_01.max():.2f}]{'':<5} {psnr_proj:.2f} dB")
    print(f"{'V3: 프로젝트 RFDN (input_range=0-255)':<45} [{sr_proj_255.min():.1f}, {sr_proj_255.max():.1f}]{'':<5} {psnr_proj_255:.2f} dB")
    
    # 차이 분석
    print(f"\n{'='*70}")
    print(f"🔍 분석")
    print(f"{'='*70}")
    
    if abs(psnr_orig - psnr_proj_255) < 0.5:
        print(f"\n✅ V1과 V3 PSNR 동일! → 프로젝트 RFDN 구조 정상")
        if psnr_proj < psnr_proj_255 - 5:
            print(f"❌ V2 PSNR 낮음 → 스케일링 로직 문제")
    else:
        print(f"\n❌ V1과 V3 PSNR 다름! → 구조적 차이 존재")
        
        # 가중치 비교
        print(f"\n[가중치 키 비교]")
        orig_keys = set(model_orig.state_dict().keys())
        proj_keys = set(model_proj_255.state_dict().keys())
        
        diff_orig = orig_keys - proj_keys
        diff_proj = proj_keys - orig_keys
        
        if diff_orig:
            print(f"  V1에만 있는 키: {list(diff_orig)[:5]}")
        if diff_proj:
            print(f"  V3에만 있는 키: {list(diff_proj)[:5]}")
        
        # Shape 비교
        print(f"\n[Shape 비교]")
        for key in list(orig_keys & proj_keys)[:5]:
            s1 = model_orig.state_dict()[key].shape
            s2 = model_proj_255.state_dict()[key].shape
            match = "✓" if s1 == s2 else "✗"
            print(f"  {key}: {list(s1)} vs {list(s2)} {match}")


if __name__ == '__main__':
    main()