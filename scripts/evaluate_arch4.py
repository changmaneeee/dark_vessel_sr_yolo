#!/usr/bin/env python
"""
=============================================================================
evaluate_arch4.py - Arch4 Adaptive 2-Pass 평가
=============================================================================
- 2-Pass 방식: Pass1(LR) → 조건부 Pass2(SR) → 병합
- Pass2 트리거 비율 분석 포함
- Ultralytics 통일 평가
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
import torch.nn.functional as F
import shutil
import yaml
from typing import Dict, Any, List
from types import SimpleNamespace

from ultralytics import YOLO
from src.models.pipelines.arch4_adaptive import Arch4Adaptive


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


class Arch4Evaluator:
    """Arch4 Adaptive 2-Pass 평가 (Ultralytics 통일)"""
    
    def __init__(
        self,
        arch4_config_path: str,
        arch4_weights_path: str,
        yolo_weights_path: str,
        hr_data_yaml: str,
        lr_data_yaml: str,
        sr_output_dir: str = '/tmp/arch4_images_eval',
        device: str = 'cuda',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        # Arch4 특화 threshold (합리적 튜닝)
        low_conf_threshold: float = 0.15,
        high_conf_threshold: float = 0.45,
        merge_iou_threshold: float = 0.5
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.hr_data_yaml = hr_data_yaml
        self.lr_data_yaml = lr_data_yaml
        self.sr_output_dir = Path(sr_output_dir)
        self.yolo_weights_path = yolo_weights_path
        
        # Arch4 특화 threshold
        self.low_conf_threshold = low_conf_threshold
        self.high_conf_threshold = high_conf_threshold
        self.merge_iou_threshold = merge_iou_threshold
        
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
        print(f"📊 Arch4 Adaptive 2-Pass 평가")
        print(f"{'='*70}")
        print(f"HR images: {self.hr_val_images_dir}")
        print(f"LR images: {self.lr_val_images_dir}")
        print(f"Output: {self.sr_output_dir}")
        print(f"\n[Arch4 Threshold 설정]")
        print(f"  low_conf:  {self.low_conf_threshold}")
        print(f"  high_conf: {self.high_conf_threshold}")
        print(f"  merge_iou: {self.merge_iou_threshold}")
        
        # YOLO 로드 (baseline 평가용)
        print(f"\n[YOLO 로드]: {yolo_weights_path}")
        self.yolo = YOLO(yolo_weights_path, verbose=False)
        
        # Arch4 로드
        print(f"[Arch4 로드]")
        config = load_config(arch4_config_path)
        self.arch4 = Arch4Adaptive(config)
        
        # Threshold 업데이트
        self.arch4.set_thresholds(
            low_conf=self.low_conf_threshold,
            high_conf=self.high_conf_threshold,
            merge_iou=self.merge_iou_threshold,
            final_conf=self.conf_threshold
        )
        
        if arch4_weights_path and Path(arch4_weights_path).exists():
            checkpoint = torch.load(arch4_weights_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.arch4.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.arch4.load_state_dict(checkpoint)
            print(f"  ✓ Arch4 weights 로드 완료")
        else:
            print(f"  ⚠️ Arch4 전체 weights 없음 - 개별 모듈 weights 사용")
        
        self.arch4 = self.arch4.to(device)
        self.arch4.eval()
        
        self.transform = T.ToTensor()
    
    def generate_arch4_images(self, max_images: int = None) -> Path:
        """LR → Arch4 (2-Pass Adaptive) 이미지 생성"""
        print(f"\n{'='*70}")
        print(f"[Arch4 이미지 생성 - 2-Pass Adaptive]")
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
        pass2_triggered_list = []
        
        # SR 적용된 이미지의 PSNR/SSIM (pass2 트리거된 것만)
        psnr_sr_only = []
        ssim_sr_only = []
        
        for img_path in tqdm(lr_images, desc="Arch4 생성"):
            # LR 이미지 로드 (0-1 범위)
            lr_img = Image.open(img_path).convert('RGB')
            lr_tensor = self.transform(lr_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                result = self.arch4.forward(lr_tensor, return_intermediate=True)
            
            # Pass2 트리거 여부
            pass2_triggered = result['pass2_triggered'][0]
            pass2_triggered_list.append(pass2_triggered)
            
            # 최종 이미지 결정
            if pass2_triggered and result['hr_image'] is not None:
                # SR 이미지 사용
                final_image = result['hr_image'][0]
            else:
                # LR upsampled 사용
                final_image = result['lr_upsampled'][0]
            
            # 이미지 정규화 (0-1 범위로)
            if final_image.max() > 1.0:
                final_image_01 = final_image / 255.0
            else:
                final_image_01 = final_image
            final_image_01 = torch.clamp(final_image_01, 0, 1)
            
            # 이미지 저장
            final_pil = T.ToPILImage()(final_image_01.cpu())
            final_pil.save(sr_images_dir / img_path.name)
            
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
                
                final_for_metric = final_image_01.unsqueeze(0).cpu()
                if final_for_metric.shape[-2:] != hr_gt_tensor.shape[-2:]:
                    final_for_metric = F.interpolate(
                        final_for_metric, size=hr_gt_tensor.shape[-2:], 
                        mode='bilinear', align_corners=False
                    )
                
                psnr_val = calculate_psnr(final_for_metric, hr_gt_tensor)
                ssim_val = calculate_ssim(final_for_metric, hr_gt_tensor)
                
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
                
                # SR 적용된 경우만 별도 기록
                if pass2_triggered:
                    psnr_sr_only.append(psnr_val)
                    ssim_sr_only.append(ssim_val)
        
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
        
        # 품질 및 Pass2 통계 저장
        pass2_ratio = np.mean(pass2_triggered_list) if pass2_triggered_list else 0.0
        
        self.sr_quality = {
            'psnr_all': np.mean(psnr_values) if psnr_values else 0.0,
            'ssim_all': np.mean(ssim_values) if ssim_values else 0.0,
            'psnr_sr_only': np.mean(psnr_sr_only) if psnr_sr_only else 0.0,
            'ssim_sr_only': np.mean(ssim_sr_only) if ssim_sr_only else 0.0,
        }
        
        self.pass2_stats = {
            'total_images': len(pass2_triggered_list),
            'pass2_triggered_count': sum(pass2_triggered_list),
            'pass2_ratio': pass2_ratio,
            'pass1_only_count': len(pass2_triggered_list) - sum(pass2_triggered_list),
        }
        
        print(f"\n  ✓ Arch4 이미지 생성 완료")
        print(f"\n  [전체 품질]")
        print(f"  PSNR (전체): {self.sr_quality['psnr_all']:.2f} dB")
        print(f"  SSIM (전체): {self.sr_quality['ssim_all']:.4f}")
        print(f"\n  [SR 적용된 이미지만]")
        print(f"  PSNR (SR만): {self.sr_quality['psnr_sr_only']:.2f} dB")
        print(f"  SSIM (SR만): {self.sr_quality['ssim_sr_only']:.4f}")
        print(f"\n  [Pass2 (SR) 트리거 통계]")
        print(f"  총 이미지: {self.pass2_stats['total_images']}")
        print(f"  Pass2 트리거: {self.pass2_stats['pass2_triggered_count']} ({pass2_ratio*100:.1f}%)")
        print(f"  Pass1만 사용: {self.pass2_stats['pass1_only_count']} ({(1-pass2_ratio)*100:.1f}%)")
        
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
        output_path: str = 'results/arch4_evaluation.json',
        max_images: int = None
    ):
        """전체 3-way 비교 (LR, Arch4, HR) + Pass2 분석"""
        
        # 1. Arch4 이미지 생성
        arch4_yaml_path = self.generate_arch4_images(max_images=max_images)
        
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
        
        # 4. Arch4: YOLO on Adaptive 2-Pass
        print(f"\n{'='*70}")
        print(f"[Arch4] YOLO on Adaptive 2-Pass(LR)")
        print(f"{'='*70}")
        arch4_metrics = self.evaluate_with_ultralytics(str(arch4_yaml_path), "Arch4")
        
        # SR 품질 및 Pass2 통계 추가
        arch4_metrics.update({
            'psnr_all': self.sr_quality['psnr_all'],
            'ssim_all': self.sr_quality['ssim_all'],
            'psnr_sr_only': self.sr_quality['psnr_sr_only'],
            'ssim_sr_only': self.sr_quality['ssim_sr_only'],
            'pass2_ratio': self.pass2_stats['pass2_ratio'],
        })
        
        # 결과 정리
        comparison = {
            'upper_bound_hr': hr_metrics,
            'lower_bound_lr': lr_metrics,
            'arch4_adaptive': arch4_metrics,
            'pass2_stats': self.pass2_stats,
            'threshold_settings': {
                'low_conf': self.low_conf_threshold,
                'high_conf': self.high_conf_threshold,
                'merge_iou': self.merge_iou_threshold,
                'final_conf': self.conf_threshold,
            },
            'improvement_over_lr': {
                key: arch4_metrics[key] - lr_metrics[key]
                for key in ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']
            },
            'gap_to_hr': {
                key: hr_metrics[key] - arch4_metrics[key]
                for key in ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']
            }
        }
        
        # 결과 출력
        print(f"\n{'='*70}")
        print(f"📊 3-Way 비교 결과")
        print(f"{'='*70}")
        
        print(f"\n{'Metric':<15} {'LR(Lower)':>12} {'Arch4(2Pass)':>12} {'HR(Upper)':>12}")
        print("-" * 55)
        
        for metric in ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']:
            lr_val = lr_metrics[metric]
            arch4_val = arch4_metrics[metric]
            hr_val = hr_metrics[metric]
            print(f"{metric:<15} {lr_val:>12.4f} {arch4_val:>12.4f} {hr_val:>12.4f}")
        
        print(f"\n[Arch4 vs LR 개선]")
        for metric in ['mAP50', 'mAP50-95', 'f1']:
            delta = comparison['improvement_over_lr'][metric]
            sign = '+' if delta > 0 else ''
            pct = delta / (lr_metrics[metric] + 1e-10) * 100
            print(f"  {metric}: {sign}{delta:.4f} ({sign}{pct:.1f}%)")
        
        print(f"\n[HR 대비 회복률]")
        for metric in ['mAP50', 'mAP50-95', 'f1']:
            recovery = arch4_metrics[metric] / (hr_metrics[metric] + 1e-10) * 100
            print(f"  {metric}: {recovery:.1f}%")
        
        print(f"\n[Arch4 품질]")
        print(f"  PSNR (전체): {arch4_metrics['psnr_all']:.2f} dB")
        print(f"  SSIM (전체): {arch4_metrics['ssim_all']:.4f}")
        print(f"  PSNR (SR만): {arch4_metrics['psnr_sr_only']:.2f} dB")
        print(f"  SSIM (SR만): {arch4_metrics['ssim_sr_only']:.4f}")
        
        print(f"\n[Pass2 (SR) 트리거 분석]")
        print(f"  Pass2 트리거 비율: {self.pass2_stats['pass2_ratio']*100:.1f}%")
        print(f"  Pass2 트리거 수: {self.pass2_stats['pass2_triggered_count']}/{self.pass2_stats['total_images']}")
        print(f"  → SR이 필요하다고 판단된 이미지 비율")
        
        print(f"\n[Threshold 설정]")
        print(f"  low_conf:  {self.low_conf_threshold} (이 이상 ~ high 이하면 uncertain)")
        print(f"  high_conf: {self.high_conf_threshold} (이 이상이면 confident)")
        print(f"  merge_iou: {self.merge_iou_threshold}")
        
        # 저장
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n결과 저장: {output_path}")
        
        return comparison


def main():
    parser = argparse.ArgumentParser(description='Arch4 Adaptive 2-Pass 평가')
    
    parser.add_argument('--config', type=str, 
                        default='configs/experiment/arch4_adaptive.yaml')
    parser.add_argument('--weights', type=str, default=None,
                        help='Arch4 전체 weights (optional)')
    parser.add_argument('--yolo_weights', type=str, 
                        default='weights/yolohr/8s/best.pt')
    parser.add_argument('--hr_data_yaml', type=str,
                        default='/home/changmin/smart_airbus_data/data.yaml')
    parser.add_argument('--lr_data_yaml', type=str,
                        default='/home/changmin/smart_airbus_data_lr/data.yaml')
    parser.add_argument('--sr_output_dir', type=str,
                        default='/tmp/arch4_images_eval')
    parser.add_argument('--output', type=str,
                        default='results/arch4_evaluation.json')
    parser.add_argument('--max_images', type=int, default=None,
                        help='평가할 최대 이미지 수 (None=전체)')
    
    # Detection thresholds
    parser.add_argument('--conf', type=float, default=0.25,
                        help='최종 detection confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold')
    
    # Arch4 특화 thresholds (합리적 튜닝)
    parser.add_argument('--low_conf', type=float, default=0.15,
                        help='Pass2 트리거를 위한 low confidence threshold')
    parser.add_argument('--high_conf', type=float, default=0.45,
                        help='Confident detection threshold (이 이상이면 SR 불필요)')
    parser.add_argument('--merge_iou', type=float, default=0.5,
                        help='Pass1/Pass2 결과 병합 시 IoU threshold')
    
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    evaluator = Arch4Evaluator(
        arch4_config_path=args.config,
        arch4_weights_path=args.weights,
        yolo_weights_path=args.yolo_weights,
        hr_data_yaml=args.hr_data_yaml,
        lr_data_yaml=args.lr_data_yaml,
        sr_output_dir=args.sr_output_dir,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        low_conf_threshold=args.low_conf,
        high_conf_threshold=args.high_conf,
        merge_iou_threshold=args.merge_iou
    )
    
    evaluator.compare_and_save(
        output_path=args.output,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()