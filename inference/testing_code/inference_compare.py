#!/usr/bin/env python
"""
=============================================================================
inference_compare.py - Arch0 vs Arch2 비교 추론 (4060 PC용)
=============================================================================
Arch0 (Sequential): 항상 SR 적용
Arch2 (SoftGate): Gate가 SR 적용 여부 결정

사용법:
    cd ~/dark_vessel_sr_yolo
    
    python inference_compare.py \
        --lr_root /home/changmin/smart_airbus_data_lr \
        --hr_root /home/changmin/smart_airbus_data \
        --rfdn_weights /home/changmin/dark_vessel_sr_yolo/weights/rfdn/model_best.pt \
        --yolo_weights /home/changmin/dark_vessel_sr_yolo/weights/yolo_lr/8s/best.pt \
        --gate_weights /home/changmin/dark_vessel_sr_yolo/training/gate_arch2/checkpoints/gate_gt/gate_best.pt \
        --output /home/changmin/dark_vessel_sr_yolo/results/arch0_vs_arch2 \
        --max_samples 1000
"""

#!/usr/bin/env python
"""
=============================================================================
inference_compare.py - Arch0 vs Arch2 비교 추론 (4060 PC용)
=============================================================================
Arch0 (Sequential): 항상 SR 적용
Arch2 (SoftGate): Gate가 SR 적용 여부 결정

사용법:
    cd ~/dark_vessel_sr_yolo
    
    python inference_compare.py \
        --lr_root /home/changmin/smart_airbus_data_lr \
        --hr_root /home/changmin/smart_airbus_data_hr \
        --rfdn_weights ./weights/rfdn/model_best.pt \
        --yolo_weights ./weights/yolo_lr/8s/best.pt \
        --gate_weights ./training/gate_arch2/checkpoints/gate_gt/gate_best.pt \
        --output ./results/arch0_vs_arch2 \
        --max_samples 1000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

# Ultralytics
from ultralytics import YOLO


# =============================================================================
# Gate Model
# =============================================================================

class LightweightGate(nn.Module):
    """경량 Gate Network (~50K params)"""
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_channels * 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.gap(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.classifier(feat)
        return torch.sigmoid(out)


# =============================================================================
# RFDN Model (공식 repo 가중치 호환 버전)
# =============================================================================

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


class ESA(nn.Module):
    """Enhanced Spatial Attention (공식 repo 키 호환: conv3_)"""
    
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


class RFDB(nn.Module):
    """Residual Feature Distillation Block"""
    
    def __init__(self, in_channels, distillation_rate=0.5):  # 서버 학습 설정: 0.5
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = int(in_channels * distillation_rate)
        self.rc = self.remaining_channels = in_channels
        
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
        r_c1 = self.act(self.c1_r(input) + input)
        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.act(self.c2_r(r_c1) + r_c1)
        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.act(self.c3_r(r_c2) + r_c2)
        r_c4 = self.act(self.c4(r_c3))
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))
        return out_fused + input


class RFDN(nn.Module):
    """RFDN - 공식 repo 가중치 호환 (c.0, upsampler.0 구조)"""
    
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(RFDN, self).__init__()
        
        self.fea_conv = conv_layer(in_nc, nf, 3)
        
        self.B1 = RFDB(nf)
        self.B2 = RFDB(nf)
        self.B3 = RFDB(nf)
        self.B4 = RFDB(nf)
        
        # 공식 repo: Sequential[0]
        self.c = nn.Sequential(
            conv_layer(nf * num_modules, nf, 1)
        )
        
        self.LR_conv = conv_layer(nf, nf, 3)
        
        # 공식 repo: Sequential[0]
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


def load_rfdn_model(weights_path: str, device: torch.device):
    """RFDN 모델 로드 (공식 repo 가중치 호환)"""
    
    print(f"  Loading RFDN from {weights_path}...")
    
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    
    # state_dict 추출
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'params' in ckpt:
            state_dict = ckpt['params']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    
    print(f"    State dict: {len(state_dict)} keys")
    
    # 모델 생성
    model = RFDN(in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4)
    
    # 로드
    model.load_state_dict(state_dict)
    print("  [✓] RFDN loaded successfully!")
    
    model.to(device)
    model.eval()
    return model


# =============================================================================
# Dataset
# =============================================================================

class InferenceDataset(Dataset):
    """추론용 데이터셋"""
    
    def __init__(self, lr_root: str, hr_root: Optional[str] = None,
                 split: str = 'val', max_samples: Optional[int] = None):
        self.lr_root = Path(lr_root)
        self.hr_root = Path(hr_root) if hr_root else None
        
        self.lr_img_dir = self.lr_root / 'images' / split
        self.lr_label_dir = self.lr_root / 'labels' / split
        
        if self.hr_root:
            self.hr_img_dir = self.hr_root / 'images' / split
        
        if not self.lr_img_dir.exists():
            raise FileNotFoundError(f"Not found: {self.lr_img_dir}")
        
        self.image_files = sorted(
            list(self.lr_img_dir.glob('*.jpg')) + 
            list(self.lr_img_dir.glob('*.png'))
        )
        
        if max_samples:
            self.image_files = self.image_files[:max_samples]
        
        print(f"[Dataset] {len(self.image_files)} images from {self.lr_img_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        lr_path = self.image_files[idx]
        img_name = lr_path.stem
        
        lr_img = cv2.imread(str(lr_path))
        if lr_img is None:
            raise ValueError(f"Failed: {lr_path}")
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        
        hr_img = None
        if self.hr_root:
            hr_path = self.hr_img_dir / lr_path.name
            if hr_path.exists():
                hr_img = cv2.imread(str(hr_path))
                if hr_img is not None:
                    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        label_path = self.lr_label_dir / f"{img_name}.txt"
        has_ship = label_path.exists() and label_path.stat().st_size > 0
        
        return {
            'lr_img': lr_img,
            'hr_img': hr_img,
            'img_name': img_name,
            'has_ship': has_ship,
        }


def collate_fn(batch):
    return batch


# =============================================================================
# Utilities
# =============================================================================

def preprocess_for_sr(img, device):
    t = torch.from_numpy(img.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def postprocess_sr(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


# =============================================================================
# Arch0 Inference
# =============================================================================

def run_arch0_inference(dataloader, rfdn, yolo, device, output_dir):
    print("\n" + "=" * 60)
    print("🔵 Arch0 (Sequential) - Always SR")
    print("=" * 60)
    
    results = {'detections': [], 'times': [], 'psnr_values': [], 'sr_ratio': 1.0}
    
    pred_dir = output_dir / 'arch0_predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    total_time = 0
    total_boxes = 0
    
    for batch in tqdm(dataloader, desc="Arch0"):
        for sample in batch:
            lr_img = sample['lr_img']
            hr_img = sample['hr_img']
            img_name = sample['img_name']
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            
            # SR
            lr_t = preprocess_for_sr(lr_img, device)
            with torch.no_grad():
                sr_t = rfdn(lr_t)
            sr_img = postprocess_sr(sr_t)
            
            # YOLO
            yolo_res = yolo(sr_img, verbose=False)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start
            results['times'].append(elapsed)
            total_time += elapsed
            
            # PSNR
            if hr_img is not None:
                if sr_img.shape[:2] != hr_img.shape[:2]:
                    sr_r = cv2.resize(sr_img, (hr_img.shape[1], hr_img.shape[0]))
                else:
                    sr_r = sr_img
                results['psnr_values'].append(calculate_psnr(sr_r, hr_img))
            
            # Save
            boxes = yolo_res[0].boxes
            num = len(boxes) if boxes is not None else 0
            total_boxes += num
            
            with open(pred_dir / f"{img_name}.txt", 'w') as f:
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls = int(box.cls.item())
                        conf = box.conf.item()
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        h, w = sr_img.shape[:2]
                        f.write(f"{cls} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} "
                                f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f} {conf:.6f}\n")
            
            results['detections'].append({
                'img_name': img_name, 'num_boxes': num, 'has_gt': sample['has_ship']
            })
    
    n = len(results['times'])
    results['n_images'] = n
    results['avg_time_ms'] = np.mean(results['times']) * 1000
    results['fps'] = n / total_time if total_time > 0 else 0
    results['total_detections'] = total_boxes
    if results['psnr_values']:
        results['avg_psnr'] = np.mean(results['psnr_values'])
    
    print(f"\n[Arch0] {n} images, {total_boxes} detections")
    print(f"        {results['avg_time_ms']:.2f} ms/img, {results['fps']:.2f} FPS")
    if 'avg_psnr' in results:
        print(f"        PSNR: {results['avg_psnr']:.2f} dB")
    
    return results


# =============================================================================
# Arch2 Inference
# =============================================================================

def run_arch2_inference(dataloader, rfdn, yolo, gate, device, output_dir, threshold=0.5):
    print("\n" + "=" * 60)
    print(f"🟢 Arch2 (SoftGate) - Threshold: {threshold}")
    print("=" * 60)
    
    results = {'detections': [], 'times': [], 'psnr_values': [], 
               'gate_values': [], 'sr_applied': []}
    
    pred_dir = output_dir / 'arch2_predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    total_time = 0
    total_boxes = 0
    sr_count = 0
    
    for batch in tqdm(dataloader, desc="Arch2"):
        for sample in batch:
            lr_img = sample['lr_img']
            hr_img = sample['hr_img']
            img_name = sample['img_name']
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            
            # Gate
            lr_t = preprocess_for_sr(lr_img, device)
            gate_in = F.interpolate(lr_t, size=(160, 160), mode='bilinear', align_corners=False)
            
            with torch.no_grad():
                gate_prob = gate(gate_in).item()
            
            results['gate_values'].append(gate_prob)
            apply_sr = gate_prob >= threshold
            results['sr_applied'].append(apply_sr)
            
            # SR or Bypass
            if apply_sr:
                sr_count += 1
                with torch.no_grad():
                    sr_t = rfdn(lr_t)
                out_img = postprocess_sr(sr_t)
            else:
                h, w = lr_img.shape[:2]
                out_img = cv2.resize(lr_img, (w*4, h*4), interpolation=cv2.INTER_LINEAR)
            
            # YOLO
            yolo_res = yolo(out_img, verbose=False)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start
            results['times'].append(elapsed)
            total_time += elapsed
            
            # PSNR (SR만)
            if hr_img is not None and apply_sr:
                if out_img.shape[:2] != hr_img.shape[:2]:
                    out_r = cv2.resize(out_img, (hr_img.shape[1], hr_img.shape[0]))
                else:
                    out_r = out_img
                results['psnr_values'].append(calculate_psnr(out_r, hr_img))
            
            # Save
            boxes = yolo_res[0].boxes
            num = len(boxes) if boxes is not None else 0
            total_boxes += num
            
            with open(pred_dir / f"{img_name}.txt", 'w') as f:
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls = int(box.cls.item())
                        conf = box.conf.item()
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        h, w = out_img.shape[:2]
                        f.write(f"{cls} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} "
                                f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f} {conf:.6f}\n")
            
            results['detections'].append({
                'img_name': img_name, 'num_boxes': num, 
                'sr_applied': apply_sr, 'gate_value': gate_prob,
                'has_gt': sample['has_ship']
            })
    
    n = len(results['times'])
    results['n_images'] = n
    results['avg_time_ms'] = np.mean(results['times']) * 1000
    results['fps'] = n / total_time if total_time > 0 else 0
    results['total_detections'] = total_boxes
    results['sr_ratio'] = sr_count / n if n > 0 else 0
    results['avg_gate'] = np.mean(results['gate_values'])
    results['sr_count'] = sr_count
    results['bypass_count'] = n - sr_count
    if results['psnr_values']:
        results['avg_psnr'] = np.mean(results['psnr_values'])
    
    print(f"\n[Arch2] {n} images, {total_boxes} detections")
    print(f"        SR: {sr_count}/{n} ({results['sr_ratio']*100:.1f}%)")
    print(f"        Gate avg: {results['avg_gate']:.4f}")
    print(f"        {results['avg_time_ms']:.2f} ms/img, {results['fps']:.2f} FPS")
    if 'avg_psnr' in results:
        print(f"        PSNR (SR): {results['avg_psnr']:.2f} dB")
    
    return results


# =============================================================================
# Analysis
# =============================================================================

def analyze_gate(arch2_results, output_dir):
    print("\n" + "=" * 60)
    print("📊 Gate Decision Analysis")
    print("=" * 60)
    
    dets = arch2_results['detections']
    tp = fp = tn = fn = 0
    
    for d in dets:
        has_gt, sr = d['has_gt'], d['sr_applied']
        if sr and has_gt: tp += 1
        elif sr and not has_gt: fp += 1
        elif not sr and not has_gt: tn += 1
        else: fn += 1
    
    total = len(dets)
    
    print(f"\n  ┌────────────────┬──────────────┬──────────────┐")
    print(f"  │                │  GT: Ship    │  GT: Empty   │")
    print(f"  ├────────────────┼──────────────┼──────────────┤")
    print(f"  │  Gate: SR      │  {tp:5d} (TP)  │  {fp:5d} (FP)  │")
    print(f"  │  Gate: Bypass  │  {fn:5d} (FN)  │  {tn:5d} (TN)  │")
    print(f"  └────────────────┴──────────────┴──────────────┘")
    
    print(f"\n  TP: {tp} ({tp/total*100:.1f}%) - 선박에 SR ✓")
    print(f"  FP: {fp} ({fp/total*100:.1f}%) - 빈 이미지에 SR")
    print(f"  TN: {tn} ({tn/total*100:.1f}%) - 빈 이미지 Bypass ✓")
    print(f"  FN: {fn} ({fn/total*100:.1f}%) - 선박 Bypass ⚠️")
    
    with open(output_dir / 'gate_analysis.json', 'w') as f:
        json.dump({'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}, f, indent=2)
    
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def print_comparison(a0, a2):
    print("\n" + "=" * 75)
    print("📊 COMPARISON: Arch0 vs Arch2")
    print("=" * 75)
    
    print(f"\n{'Metric':<25} {'Arch0':<18} {'Arch2':<18} {'Diff':<15}")
    print("-" * 75)
    
    d0, d2 = a0['total_detections'], a2['total_detections']
    print(f"{'Total Detections':<25} {d0:<18} {d2:<18} {d2-d0:+d}")
    print(f"{'SR Applied':<25} {'100.0%':<18} {a2['sr_ratio']*100:.1f}%{'':<12} {(a2['sr_ratio']-1)*100:+.1f}%")
    
    t0, t2 = a0['avg_time_ms'], a2['avg_time_ms']
    print(f"{'Avg Time (ms)':<25} {t0:.2f}{'':<14} {t2:.2f}{'':<14} {t2-t0:+.2f}")
    
    f0, f2 = a0['fps'], a2['fps']
    print(f"{'FPS':<25} {f0:.2f}{'':<14} {f2:.2f}{'':<14} {f2-f0:+.2f}")
    
    speedup = f2 / f0 if f0 > 0 else 0
    print(f"{'Speedup':<25} {'1.00x':<18} {speedup:.2f}x{'':<12}")
    
    if 'avg_psnr' in a0 and 'avg_psnr' in a2:
        print(f"{'Avg PSNR (dB)':<25} {a0['avg_psnr']:.2f}{'':<14} {a2['avg_psnr']:.2f}{'':<14} {a2['avg_psnr']-a0['avg_psnr']:+.2f}")
    
    print("-" * 75)
    print(f"\n📈 Summary:")
    if d0 > 0:
        print(f"  • Detection: {d2-d0:+d} ({(d2-d0)/d0*100:+.1f}%)")
    print(f"  • SR 절약: {(1-a2['sr_ratio'])*100:.1f}%")
    print(f"  • Speedup: {speedup:.2f}x")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_root', type=str, required=True)
    parser.add_argument('--hr_root', type=str, default=None)
    parser.add_argument('--rfdn_weights', type=str, required=True)
    parser.add_argument('--yolo_weights', type=str, required=True)
    parser.add_argument('--gate_weights', type=str, required=True)
    parser.add_argument('--gate_threshold', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--output', type=str, default='./results/arch0_vs_arch2')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Models
    print("\n" + "=" * 60)
    print("🔧 Loading Models")
    print("=" * 60)
    
    rfdn = load_rfdn_model(args.rfdn_weights, device)
    
    print(f"  Loading YOLO from {args.yolo_weights}...")
    yolo = YOLO(args.yolo_weights)
    yolo.to(device)
    print("  [✓] YOLO loaded")
    
    print(f"  Loading Gate from {args.gate_weights}...")
    gate = LightweightGate()
    ckpt = torch.load(args.gate_weights, map_location=device, weights_only=False)
    gate.load_state_dict(ckpt.get('model_state_dict', ckpt))
    gate.to(device)
    gate.eval()
    print("  [✓] Gate loaded")
    
    # Data
    print("\n" + "=" * 60)
    print("📁 Loading Dataset")
    print("=" * 60)
    
    dataset = InferenceDataset(args.lr_root, args.hr_root, 'val', args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)
    
    # Run
    a0 = run_arch0_inference(loader, rfdn, yolo, device, output_dir)
    a2 = run_arch2_inference(loader, rfdn, yolo, gate, device, output_dir, args.gate_threshold)
    
    # Analysis
    analyze_gate(a2, output_dir)
    print_comparison(a0, a2)
    
    # Save
    summary = {
        'arch0': {k: v for k, v in a0.items() if k != 'detections'},
        'arch2': {k: v for k, v in a2.items() if k not in ['detections', 'gate_values', 'sr_applied']},
        'settings': vars(args)
    }
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    
    print(f"\n✅ Results saved to {output_dir}")


if __name__ == '__main__':
    main()