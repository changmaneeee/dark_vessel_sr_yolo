#!/usr/bin/env python
"""
=============================================================================
evaluate_arch2.py - Arch2 SoftGate 평가
=============================================================================
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import argparse
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import shutil
import yaml
from typing import Dict, Any
from types import SimpleNamespace

from ultralytics import YOLO
from src.models.pipelines.arch2_softgate import Arch2SoftGate


def load_config(config_path: str) -> Any:
    """YAML config 로드"""
    def dict_to_namespace(d):
        if isinstance(d, dict):
            for key, value in d.items():
                d[key] = dict_to_namespace(value)
            return SimpleNamespace(**d)
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        return d
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return dict_to_namespace(config_dict)


def calculate_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """PSNR 계산 (0-1 범위 기준)"""
    mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def calculate_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """SSIM 계산"""
    try:
        from torchmetrics.functional import structural_similarity_index_measure as ssim
        return ssim(sr, hr).item()
    except ImportError:
        c1, c2 = 0.01**2, 0.03**2
        mu_sr = sr.mean()
        mu_hr = hr.mean()
        sigma_sr = sr.var()
        sigma_hr = hr.var()
        sigma_sr_hr = ((sr - mu_sr) * (hr - mu_hr)).mean()
        
        ssim_val = ((2*mu_sr*mu_hr + c1) * (2*sigma_sr_hr + c2)) / \
                   ((mu_sr**2 + mu_hr**2 + c1) * (sigma_sr + sigma_hr + c2))
        return ssim_val.item()


class Arch2Evaluator:
    """Arch2 SoftGate 평가 (Ultralytics 통일)"""
    
    def __init__(
        self,
        arch2_config_path: str,
        arch2_weights_path: str,
        yolo_weights_path: str,
        hr_data_yaml: str,
        lr_data_yaml: str,
        sr_output_dir: str = '/tmp/arch2_images_eval',
        device: str = 'cuda',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.hr_data_yaml = hr_data_yaml
        self.lr_data_yaml = lr_data_yaml
        self.sr_output_dir = Path(sr_output_dir)
        self.yolo_weights_path = yolo_weights_path
        
        # 경로 설정
        with open(hr_data_yaml, 'r') as f:
            hr_config = yaml.safe_load(f)
        with open(lr_data_yaml, 'r') as f:
            lr_config = yaml.safe_load(f)
        
        hr_path = Path(hr_config.get('path', '/home/changmin/smart_airbus_data'))
        lr_path = Path(lr_config.get('path', '/home/changmin/smart_airbus_data_lr'))
        
        self.hr_val_images_dir = hr_path / 'images' / 'val'
        self.hr_val_labels_dir = hr_path / 'labels' / 'val'
        self.lr_val_images_dir = lr_path / 'images' / 'val'
        
        print(f"\n{'='*70}")
        print(f"📊 Arch2 SoftGate 평가")
        print(f"{'='*70}")
        print(f"HR images: {self.hr_val_images_dir}")
        print(f"LR images: {self.lr_val_images_dir}")
        print(f"Output: {self.sr_output_dir}")
        
        # YOLO 로드 (baseline 평가용)
        print(f"\n[YOLO 로드]: {yolo_weights_path}")
        self.yolo = YOLO(yolo_weights_path, verbose=False)
        
        # Arch2 로드
        print(f"[Arch2 로드]")
        config = load_config(arch2_config_path)
        self.arch2 = Arch2SoftGate(config)
        
        if arch2_weights_path and Path(arch2_weights_path).exists():
            checkpoint = torch.load(arch2_weights_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.arch2.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.arch2.load_state_dict(checkpoint)
            print(f"  ✓ Arch2 weights 로드 완료")
        else:
            print(f"  ⚠️ Arch2 전체 weights 없음 - 개별 모듈 weights 사용")
        
        self.arch2 = self.arch2.to(device)
        self.arch2.eval()
        
        self.transform = T.ToTensor()
    
    def generate_arch2_images(self, max_images: int = None) -> Path:
        """LR → Arch2 (Gate-based SR) 이미지 생성"""
        print(f"\n{'='*70}")
        print(f"[Arch2 이미지 생성]")
        print(f"{'='*70}")
        
        # 출력 디렉토리 설정 (기존 삭제)
        if self.sr_output_dir.exists():
            shutil.rmtree(self.sr_output_dir)
        
        sr_images_dir = self.sr_output_dir / 'images' / 'val'
        sr_labels_dir = self.sr_output_dir / 'labels' / 'val'
        sr_images_dir.mkdir(parents=True, exist_ok=True)
        sr_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # LR 이미지 목록
        lr_images = sorted(self.lr_val_images_dir.glob('*.jpg'))
        if max_images:
            lr_images = lr_images[:max_images]
        
        print(f"  LR 이미지 수: {len(lr_images)}")
        
        psnr_values = []
        ssim_values = []
        gate_values = []
        
        for img_path in tqdm(lr_images, desc="Arch2 생성"):
            # LR 이미지 로드 (0-1 범위)
            lr_img = Image.open(img_path).convert('RGB')
            lr_tensor = self.transform(lr_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                result = self.arch2(lr_tensor, return_intermediates=True)
            
            hr_image = result['hr_image']
            gate = result['gate']
            
            # Gate 값 기록
            gate_values.append(gate.mean().item())
            
            # hr_image가 0-255 범위면 0-1로 변환
            if hr_image.max() > 1.0:
                hr_image_01 = hr_image / 255.0
            else:
                hr_image_01 = hr_image
            hr_image_01 = torch.clamp(hr_image_01, 0, 1)
            
            # 이미지 저장
            hr_pil = T.ToPILImage()(hr_image_01[0].cpu())
            hr_pil.save(sr_images_dir / img_path.name)
            
            # 라벨 복사
            label_src = self.hr_val_labels_dir / f'{img_path.stem}.txt'
            label_dst = sr_labels_dir / f'{img_path.stem}.txt'
            if label_src.exists() and not label_dst.exists():
                shutil.copy(label_src, label_dst)
            
            # PSNR/SSIM 계산 (HR이 있을 때)
            hr_img_path = self.hr_val_images_dir / img_path.name
            if hr_img_path.exists():
                hr_gt_img = Image.open(hr_img_path).convert('RGB')
                hr_gt_tensor = self.transform(hr_gt_img).unsqueeze(0)
                
                hr_for_metric = hr_image_01.cpu()
                if hr_for_metric.shape[-2:] != hr_gt_tensor.shape[-2:]:
                    hr_for_metric = torch.nn.functional.interpolate(
                        hr_for_metric, size=hr_gt_tensor.shape[-2:], 
                        mode='bilinear', align_corners=False
                    )
                
                psnr_values.append(calculate_psnr(hr_for_metric, hr_gt_tensor))
                ssim_values.append(calculate_ssim(hr_for_metric, hr_gt_tensor))
        
        # data.yaml 생성
        sr_data_yaml = {
            'path': str(self.sr_output_dir),
            'train': 'images/val',
            'val': 'images/val',
            'names': {0: 'ship'}
        }
        sr_yaml_path = self.sr_output_dir / 'data.yaml'
        with open(sr_yaml_path, 'w') as f:
            yaml.dump(sr_data_yaml, f)
        
        # 품질 및 Gate 통계 저장
        self.sr_quality = {
            'psnr': np.mean(psnr_values) if psnr_values else 0.0,
            'ssim': np.mean(ssim_values) if ssim_values else 0.0
        }
        self.gate_stats = {
            'mean': np.mean(gate_values) if gate_values else 0.5,
            'std': np.std(gate_values) if gate_values else 0.0,
            'min': np.min(gate_values) if gate_values else 0.0,
            'max': np.max(gate_values) if gate_values else 1.0,
            'sr_ratio': np.mean([g > 0.5 for g in gate_values]) if gate_values else 0.5
        }
        
        print(f"\n  ✓ Arch2 이미지 생성 완료")
        print(f"  PSNR: {self.sr_quality['psnr']:.2f} dB")
        print(f"  SSIM: {self.sr_quality['ssim']:.4f}")
        print(f"\n  [Gate 통계]")
        print(f"  Mean: {self.gate_stats['mean']:.4f}")
        print(f"  Std:  {self.gate_stats['std']:.4f}")
        print(f"  Min:  {self.gate_stats['min']:.4f}")
        print(f"  Max:  {self.gate_stats['max']:.4f}")
        print(f"  SR 우세 비율 (gate>0.5): {self.gate_stats['sr_ratio']*100:.1f}%")
        
        return sr_yaml_path
    
    def evaluate_with_ultralytics(self, data_yaml: str, name: str) -> Dict[str, float]:
        """Ultralytics val()로 평가"""
        print(f"\n[{name} 평가]")
        
        results = self.yolo.val(
            data=data_yaml,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1': float(2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-10))
        }
        
        print(f"  mAP@0.5:     {metrics['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1:          {metrics['f1']:.4f}")
        
        return metrics
    
    def compare_and_save(
        self, 
        output_path: str = 'results/arch2_evaluation.json',
        max_images: int = None
    ):
        """전체 3-way 비교 (LR, Arch2, HR)"""
        
        # 1. Arch2 이미지 생성
        arch2_yaml_path = self.generate_arch2_images(max_images=max_images)
        
        # 2. Upper Bound: YOLO on HR
        print(f"\n{'='*70}")
        print(f"[Upper Bound] YOLO on HR")
        print(f"{'='*70}")
        hr_metrics = self.evaluate_with_ultralytics(self.hr_data_yaml, "HR")
        
        # 3. Lower Bound: YOLO on LR
        print(f"\n{'='*70}")
        print(f"[Lower Bound] YOLO on LR")
        print(f"{'='*70}")
        lr_metrics = self.evaluate_with_ultralytics(self.lr_data_yaml, "LR")
        
        # 4. Arch2: YOLO on Gate-SR
        print(f"\n{'='*70}")
        print(f"[Arch2] YOLO on SoftGate(LR)")
        print(f"{'='*70}")
        arch2_metrics = self.evaluate_with_ultralytics(str(arch2_yaml_path), "Arch2")
        arch2_metrics['psnr'] = self.sr_quality['psnr']
        arch2_metrics['ssim'] = self.sr_quality['ssim']
        arch2_metrics['gate_mean'] = self.gate_stats['mean']
        arch2_metrics['gate_std'] = self.gate_stats['std']
        arch2_metrics['sr_ratio'] = self.gate_stats['sr_ratio']
        
        # 결과 정리
        comparison = {
            'upper_bound_hr': hr_metrics,
            'lower_bound_lr': lr_metrics,
            'arch2_softgate': arch2_metrics,
            'gate_stats': self.gate_stats,
            'improvement_over_lr': {
                key: arch2_metrics[key] - lr_metrics[key]
                for key in ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']
            },
            'gap_to_hr': {
                key: hr_metrics[key] - arch2_metrics[key]
                for key in ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']
            }
        }
        
        # 결과 출력
        print(f"\n{'='*70}")
        print(f"📊 3-Way 비교 결과")
        print(f"{'='*70}")
        
        print(f"\n{'Metric':<15} {'LR(Lower)':>12} {'Arch2(Gate)':>12} {'HR(Upper)':>12}")
        print("-" * 55)
        
        for metric in ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']:
            lr_val = lr_metrics[metric]
            arch2_val = arch2_metrics[metric]
            hr_val = hr_metrics[metric]
            print(f"{metric:<15} {lr_val:>12.4f} {arch2_val:>12.4f} {hr_val:>12.4f}")
        
        print(f"\n[Arch2 vs LR 개선]")
        for metric in ['mAP50', 'mAP50-95', 'f1']:
            delta = comparison['improvement_over_lr'][metric]
            sign = '+' if delta > 0 else ''
            pct = delta / (lr_metrics[metric] + 1e-10) * 100
            print(f"  {metric}: {sign}{delta:.4f} ({sign}{pct:.1f}%)")
        
        print(f"\n[HR 대비 회복률]")
        for metric in ['mAP50', 'mAP50-95', 'f1']:
            recovery = arch2_metrics[metric] / (hr_metrics[metric] + 1e-10) * 100
            print(f"  {metric}: {recovery:.1f}%")
        
        print(f"\n[Arch2 품질]")
        print(f"  PSNR: {arch2_metrics['psnr']:.2f} dB")
        print(f"  SSIM: {arch2_metrics['ssim']:.4f}")
        
        print(f"\n[Gate 분석]")
        print(f"  Gate Mean: {self.gate_stats['mean']:.4f}")
        print(f"  Gate Std:  {self.gate_stats['std']:.4f}")
        print(f"  SR 우세 비율: {self.gate_stats['sr_ratio']*100:.1f}%")
        
        # 저장
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n결과 저장: {output_path}")
        
        return comparison


def main():
    parser = argparse.ArgumentParser(description='Arch2 SoftGate 평가')
    
    parser.add_argument('--config', type=str, 
                        default='configs/experiment/arch2_softgate.yaml')
    parser.add_argument('--weights', type=str, default=None,
                        help='Arch2 전체 weights (optional)')
    parser.add_argument('--yolo_weights', type=str, 
                        default='weights/yolohr/8s/best.pt')
    parser.add_argument('--hr_data_yaml', type=str,
                        default='/home/changmin/smart_airbus_data/data.yaml')
    parser.add_argument('--lr_data_yaml', type=str,
                        default='/home/changmin/smart_airbus_data_lr/data.yaml')
    parser.add_argument('--sr_output_dir', type=str,
                        default='/tmp/arch2_images_eval')
    parser.add_argument('--output', type=str,
                        default='results/arch2_evaluation.json')
    parser.add_argument('--max_images', type=int, default=None,
                        help='평가할 최대 이미지 수 (None=전체)')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    evaluator = Arch2Evaluator(
        arch2_config_path=args.config,
        arch2_weights_path=args.weights,
        yolo_weights_path=args.yolo_weights,
        hr_data_yaml=args.hr_data_yaml,
        lr_data_yaml=args.lr_data_yaml,
        sr_output_dir=args.sr_output_dir,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    evaluator.compare_and_save(
        output_path=args.output,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()
