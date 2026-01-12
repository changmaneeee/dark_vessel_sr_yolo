#!/usr/bin/env python
"""PSNR 문제 디버깅 - MeanShift 없음 (RFDN은 shift_mean 미적용)"""

import torch
import cv2
import numpy as np
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


class ESA(nn.Module):
    def __init__(self, n_feats, conv=nn.Conv2d):
        super(ESA, self).__init__()
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


class RFDB(nn.Module):
    """block.py와 완전 동일"""
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB, self).__init__()
        self.dc = in_channels // 2  # 50//2 = 25
        self.rc = in_channels       # 50
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
        return out_fused  # block.py: NO residual!


class RFDN(nn.Module):
    """rfdn.py와 완전 동일"""
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(RFDN, self).__init__()
        self.fea_conv = conv_layer(in_nc, nf, 3)
        self.B1 = RFDB(nf)
        self.B2 = RFDB(nf)
        self.B3 = RFDB(nf)
        self.B4 = RFDB(nf)
        # conv_block with lrelu
        self.c = nn.Sequential(
            conv_layer(nf * num_modules, nf, 1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True)
        )
        self.LR_conv = conv_layer(nf, nf, 3)
        self.upsampler = nn.Sequential(
            conv_layer(nf, out_nc * (upscale ** 2), 3),
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


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    lr_root = Path('/home/changmin/smart_airbus_data_lr')
    hr_root = Path('/home/changmin/smart_airbus_data')
    rfdn_path = '/home/changmin/dark_vessel_sr_yolo/weights/rfdn/model_best.pt'
    
    print("Loading RFDN (no MeanShift)...")
    model = RFDN()
    state_dict = torch.load(rfdn_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing: {missing}")
    print(f"  Unexpected: {unexpected}")
    
    model.to(device)
    model.eval()
    print("RFDN loaded!")
    
    lr_img_dir = lr_root / 'images' / 'val'
    hr_img_dir = hr_root / 'images' / 'val'
    
    lr_files = sorted(list(lr_img_dir.glob('*.jpg'))[:5])
    
    print(f"\nTesting {len(lr_files)} images...")
    print("=" * 70)
    
    for lr_path in lr_files:
        hr_path = hr_img_dir / lr_path.name
        
        print(f"\n[{lr_path.name}]")
        
        lr_img = cv2.imread(str(lr_path))
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        print(f"  LR: shape={lr_img.shape}, range=[{lr_img.min()}, {lr_img.max()}]")
        
        if not hr_path.exists():
            print(f"  HR: NOT FOUND")
            continue
        
        hr_img = cv2.imread(str(hr_path))
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        print(f"  HR: shape={hr_img.shape}")
        
        # SR 추론 ([0, 255] 입력)
        lr_tensor = torch.from_numpy(lr_img.astype(np.float32))
        lr_tensor = lr_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        
        sr_img = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        print(f"  SR raw: range=[{sr_img.min():.2f}, {sr_img.max():.2f}]")
        
        sr_img_clipped = np.clip(sr_img, 0, 255).astype(np.uint8)
        
        # PSNR
        psnr = calculate_psnr(sr_img_clipped, hr_img)
        print(f"  ✅ SR PSNR: {psnr:.2f} dB")
        
        bicubic = cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        psnr_bicubic = calculate_psnr(bicubic, hr_img)
        print(f"  📊 Bicubic PSNR: {psnr_bicubic:.2f} dB")
        print(f"  📈 Gain: {psnr - psnr_bicubic:+.2f} dB")
    
    print("\n" + "=" * 70)
    print("Done!")


if __name__ == '__main__':
    main()