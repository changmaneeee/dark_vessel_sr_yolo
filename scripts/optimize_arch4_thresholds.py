#!/usr/bin/env python
"""
=============================================================================
optimize_arch4_thresholds.py - Arch4 Threshold Grid Search 최적화
=============================================================================
- Upper/Lower bound 평가 생략 (이미 알려진 값 사용)
- low_conf, high_conf 조합을 Grid Search
- 최적 threshold 조합 탐색
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
from typing import Dict, Any, List, Tuple
from types import SimpleNamespace
from datetime import datetime
import itertools
import gc

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
    """PSNR 계산"""
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


class Arch4ThresholdOptimizer:
    """Arch4 Threshold Grid Search 최적화"""
    
    # 이미 알려진 baseline 값들 (전체 데이터셋 기준)
    KNOWN_BASELINES = {
        'hr': {
            'mAP50': 0.780,
            'mAP50-95': 0.633,
            'precision': 0.719,
            'recall': 0.766,
            'f1': 0.742
        },
        'lr': {
            'mAP50': 0.693,
            'mAP50-95': 0.550,
            'precision': 0.724,
            'recall': 0.622,
            'f1': 0.669
        }
    }
    
    def __init__(
        self,
        arch4_config_path: str,
        yolo_hr_weights_path: str,
        yolo_lr_weights_path: str = None,  # None이면 HR weights 사용
        hr_data_yaml: str = '/home/changmin/smart_airbus_data/data.yaml',
        lr_data_yaml: str = '/home/changmin/smart_airbus_data_lr/data.yaml',
        output_dir: str = '/tmp/arch4_optimize',
        device: str = 'cuda',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.hr_data_yaml = hr_data_yaml
        self.lr_data_yaml = lr_data_yaml
        self.output_dir = Path(output_dir)
        self.yolo_hr_weights_path = yolo_hr_weights_path
        self.yolo_lr_weights_path = yolo_lr_weights_path or yolo_hr_weights_path
        
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
        print(f"📊 Arch4 Threshold 최적화")
        print(f"{'='*70}")
        print(f"HR images: {self.hr_val_images_dir}")
        print(f"LR images: {self.lr_val_images_dir}")
        print(f"YOLO HR weights: {self.yolo_hr_weights_path}")
        print(f"YOLO LR weights: {self.yolo_lr_weights_path}")
        
        # 평가용 YOLO (HR weights 사용)
        print(f"\n[평가용 YOLO 로드]")
        self.yolo_eval = YOLO(yolo_hr_weights_path, verbose=False)
        
        # Arch4 로드
        print(f"[Arch4 로드]")
        self.config = load_config(arch4_config_path)
        self.arch4_config_path = arch4_config_path
        
        self.transform = T.ToTensor()
        
        # 결과 저장
        self.results = []
    
    def _create_arch4_with_thresholds(
        self, 
        low_conf: float, 
        high_conf: float,
        merge_iou: float = 0.5
    ) -> Arch4Adaptive:
        """특정 threshold로 Arch4 생성"""
        arch4 = Arch4Adaptive(self.config)
        arch4.set_thresholds(
            low_conf=low_conf,
            high_conf=high_conf,
            merge_iou=merge_iou,
            final_conf=self.conf_threshold
        )
        arch4 = arch4.to(self.device)
        arch4.eval()
        arch4.reset_stats()
        return arch4
    
    def run_single_evaluation(
        self,
        low_conf: float,
        high_conf: float,
        merge_iou: float = 0.5,
        max_images: int = None,
        run_id: int = 0
    ) -> Dict[str, Any]:
        """단일 threshold 조합 평가"""
        
        print(f"\n{'='*70}")
        print(f"[Run {run_id}] low_conf={low_conf:.2f}, high_conf={high_conf:.2f}")
        print(f"{'='*70}")
        
        # Arch4 생성
        arch4 = self._create_arch4_with_thresholds(low_conf, high_conf, merge_iou)
        
        # 출력 디렉토리
        run_output_dir = self.output_dir / f'run_{run_id}'
        if run_output_dir.exists():
            shutil.rmtree(run_output_dir)
        
        sr_images_dir = run_output_dir / 'images' / 'val'
        sr_labels_dir = run_output_dir / 'labels' / 'val'
        sr_images_dir.mkdir(parents=True, exist_ok=True)
        sr_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # LR 이미지 목록
        lr_images = sorted(self.lr_val_images_dir.glob('*.jpg'))
        if max_images:
            lr_images = lr_images[:max_images]
        
        # 통계
        psnr_values = []
        ssim_values = []
        pass2_triggered_list = []
        
        for img_path in tqdm(lr_images, desc=f"Run {run_id}", leave=False):
            lr_img = Image.open(img_path).convert('RGB')
            lr_tensor = self.transform(lr_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                result = arch4.forward(lr_tensor, return_intermediate=True)
            
            pass2_triggered = result['pass2_triggered'][0]
            pass2_triggered_list.append(pass2_triggered)
            
            # 최종 이미지 결정
            if pass2_triggered and result['hr_image'] is not None:
                final_image = result['hr_image'][0]
            else:
                final_image = result['lr_upsampled'][0]
            
            # 정규화
            if final_image.max() > 1.0:
                final_image_01 = final_image / 255.0
            else:
                final_image_01 = final_image
            final_image_01 = torch.clamp(final_image_01, 0, 1)
            
            # 저장
            final_pil = T.ToPILImage()(final_image_01.cpu())
            final_pil.save(sr_images_dir / img_path.name)
            
            # 라벨 복사
            label_src = self.hr_val_labels_dir / f'{img_path.stem}.txt'
            label_dst = sr_labels_dir / f'{img_path.stem}.txt'
            if label_src.exists() and not label_dst.exists():
                shutil.copy(label_src, label_dst)
            
            # PSNR/SSIM (SR 적용된 경우만)
            if pass2_triggered and result['hr_image'] is not None:
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
                    
                    psnr_values.append(calculate_psnr(final_for_metric, hr_gt_tensor))
                    ssim_values.append(calculate_ssim(final_for_metric, hr_gt_tensor))
        
        # data.yaml 생성
        sr_data_yaml = {
            'path': str(run_output_dir),
            'train': 'images/val',
            'val': 'images/val',
            'names': {0: 'ship'}
        }
        sr_yaml_path = run_output_dir / 'data.yaml'
        with open(sr_yaml_path, 'w') as f:
            yaml.dump(sr_data_yaml, f)
        
        # Ultralytics 평가 (메모리 절약: workers=2, batch=8)
        results = self.yolo_eval.val(
            data=str(sr_yaml_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            workers=2,      # 메모리 절약
            batch=8         # 배치 크기 줄이기
        )
        
        # 결과 정리
        pass2_ratio = np.mean(pass2_triggered_list) if pass2_triggered_list else 0.0
        
        metrics = {
            'low_conf': low_conf,
            'high_conf': high_conf,
            'merge_iou': merge_iou,
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1': float(2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-10)),
            'pass2_ratio': pass2_ratio,
            'pass2_count': sum(pass2_triggered_list),
            'total_images': len(pass2_triggered_list),
            'psnr': np.mean(psnr_values) if psnr_values else 0.0,
            'ssim': np.mean(ssim_values) if ssim_values else 0.0,
        }
        
        # HR 대비 회복률
        metrics['recovery_mAP50'] = metrics['mAP50'] / self.KNOWN_BASELINES['hr']['mAP50'] * 100
        metrics['recovery_f1'] = metrics['f1'] / self.KNOWN_BASELINES['hr']['f1'] * 100
        
        # LR 대비 개선
        metrics['improvement_mAP50'] = metrics['mAP50'] - self.KNOWN_BASELINES['lr']['mAP50']
        metrics['improvement_f1'] = metrics['f1'] - self.KNOWN_BASELINES['lr']['f1']
        
        # 효율성 점수 (성능 향상 / SR 사용률)
        # SR을 적게 쓰면서 성능이 좋으면 높은 점수
        if pass2_ratio > 0:
            metrics['efficiency_score'] = metrics['improvement_mAP50'] / pass2_ratio
        else:
            metrics['efficiency_score'] = 0.0
        
        print(f"  mAP50: {metrics['mAP50']:.4f} (HR의 {metrics['recovery_mAP50']:.1f}%)")
        print(f"  F1: {metrics['f1']:.4f} (HR의 {metrics['recovery_f1']:.1f}%)")
        print(f"  Pass2 트리거: {metrics['pass2_ratio']*100:.1f}%")
        print(f"  효율성 점수: {metrics['efficiency_score']:.4f}")
        
        # 정리 (메모리 해제)
        del arch4
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        return metrics
    
    def grid_search(
        self,
        low_conf_values: List[float],
        high_conf_values: List[float],
        merge_iou: float = 0.5,
        max_images: int = None
    ) -> List[Dict[str, Any]]:
        """Grid Search 실행"""
        
        # 유효한 조합만 필터링 (low_conf < high_conf)
        combinations = [
            (low, high) for low, high in itertools.product(low_conf_values, high_conf_values)
            if low < high
        ]
        
        print(f"\n{'='*70}")
        print(f"📊 Grid Search 시작")
        print(f"{'='*70}")
        print(f"low_conf 범위: {low_conf_values}")
        print(f"high_conf 범위: {high_conf_values}")
        print(f"총 조합 수: {len(combinations)}")
        print(f"이미지 수: {max_images or '전체'}")
        
        results = []
        
        for run_id, (low_conf, high_conf) in enumerate(combinations):
            metrics = self.run_single_evaluation(
                low_conf=low_conf,
                high_conf=high_conf,
                merge_iou=merge_iou,
                max_images=max_images,
                run_id=run_id
            )
            results.append(metrics)
        
        self.results = results
        return results
    
    def analyze_and_save(self, output_path: str = 'results/arch4_threshold_optimization.json'):
        """결과 분석 및 저장"""
        
        if not self.results:
            print("결과가 없습니다. grid_search를 먼저 실행하세요.")
            return
        
        # 정렬 기준별 Best 찾기
        best_by_mAP50 = max(self.results, key=lambda x: x['mAP50'])
        best_by_f1 = max(self.results, key=lambda x: x['f1'])
        best_by_efficiency = max(self.results, key=lambda x: x['efficiency_score'])
        
        # 균형점 찾기 (mAP50 * (1 - pass2_ratio * 0.3))
        # pass2_ratio가 낮으면서 mAP50이 높은 것을 선호
        for r in self.results:
            r['balanced_score'] = r['mAP50'] * (1 - r['pass2_ratio'] * 0.3)
        best_balanced = max(self.results, key=lambda x: x['balanced_score'])
        
        print(f"\n{'='*70}")
        print(f"📊 최적화 결과 분석")
        print(f"{'='*70}")
        
        print(f"\n[Baseline 참고]")
        print(f"  HR: mAP50={self.KNOWN_BASELINES['hr']['mAP50']:.4f}, F1={self.KNOWN_BASELINES['hr']['f1']:.4f}")
        print(f"  LR: mAP50={self.KNOWN_BASELINES['lr']['mAP50']:.4f}, F1={self.KNOWN_BASELINES['lr']['f1']:.4f}")
        
        print(f"\n[Best by mAP50]")
        print(f"  low_conf={best_by_mAP50['low_conf']:.2f}, high_conf={best_by_mAP50['high_conf']:.2f}")
        print(f"  mAP50: {best_by_mAP50['mAP50']:.4f}, F1: {best_by_mAP50['f1']:.4f}")
        print(f"  Pass2 트리거: {best_by_mAP50['pass2_ratio']*100:.1f}%")
        
        print(f"\n[Best by F1]")
        print(f"  low_conf={best_by_f1['low_conf']:.2f}, high_conf={best_by_f1['high_conf']:.2f}")
        print(f"  mAP50: {best_by_f1['mAP50']:.4f}, F1: {best_by_f1['f1']:.4f}")
        print(f"  Pass2 트리거: {best_by_f1['pass2_ratio']*100:.1f}%")
        
        print(f"\n[Best by Efficiency (mAP향상/SR사용률)]")
        print(f"  low_conf={best_by_efficiency['low_conf']:.2f}, high_conf={best_by_efficiency['high_conf']:.2f}")
        print(f"  mAP50: {best_by_efficiency['mAP50']:.4f}, F1: {best_by_efficiency['f1']:.4f}")
        print(f"  Pass2 트리거: {best_by_efficiency['pass2_ratio']*100:.1f}%")
        print(f"  효율성 점수: {best_by_efficiency['efficiency_score']:.4f}")
        
        print(f"\n[Best Balanced (성능 * 효율)]")
        print(f"  low_conf={best_balanced['low_conf']:.2f}, high_conf={best_balanced['high_conf']:.2f}")
        print(f"  mAP50: {best_balanced['mAP50']:.4f}, F1: {best_balanced['f1']:.4f}")
        print(f"  Pass2 트리거: {best_balanced['pass2_ratio']*100:.1f}%")
        print(f"  균형 점수: {best_balanced['balanced_score']:.4f}")
        
        # 전체 결과 테이블
        print(f"\n[전체 결과 요약]")
        print(f"{'low':>6} {'high':>6} {'mAP50':>8} {'F1':>8} {'Pass2%':>8} {'Eff':>8}")
        print("-" * 50)
        
        sorted_results = sorted(self.results, key=lambda x: x['mAP50'], reverse=True)
        for r in sorted_results:
            print(f"{r['low_conf']:>6.2f} {r['high_conf']:>6.2f} {r['mAP50']:>8.4f} {r['f1']:>8.4f} {r['pass2_ratio']*100:>7.1f}% {r['efficiency_score']:>8.4f}")
        
        # 저장
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'baselines': self.KNOWN_BASELINES,
            'best_by_mAP50': best_by_mAP50,
            'best_by_f1': best_by_f1,
            'best_by_efficiency': best_by_efficiency,
            'best_balanced': best_balanced,
            'all_results': self.results
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n결과 저장: {output_path}")
        
        return output_data


def main():
    parser = argparse.ArgumentParser(description='Arch4 Threshold Grid Search')
    
    parser.add_argument('--config', type=str, 
                        default='configs/experiment/arch4_adaptive.yaml')
    parser.add_argument('--yolo_hr_weights', type=str, 
                        default='weights/yolohr/8s/best.pt',
                        help='HR 이미지용 YOLO weights')
    parser.add_argument('--yolo_lr_weights', type=str, 
                        default=None,
                        help='LR 이미지용 YOLO weights (None이면 HR weights 사용)')
    parser.add_argument('--hr_data_yaml', type=str,
                        default='/home/changmin/smart_airbus_data/data.yaml')
    parser.add_argument('--lr_data_yaml', type=str,
                        default='/home/changmin/smart_airbus_data_lr/data.yaml')
    parser.add_argument('--output_dir', type=str,
                        default='/tmp/arch4_optimize')
    parser.add_argument('--output', type=str,
                        default='results/arch4_threshold_optimization.json')
    parser.add_argument('--max_images', type=int, default=None,
                        help='평가할 최대 이미지 수 (None=전체 val)')
    
    # Grid Search 범위 (정밀 탐색)
    parser.add_argument('--low_conf_min', type=float, default=0.05)
    parser.add_argument('--low_conf_max', type=float, default=0.35)
    parser.add_argument('--low_conf_step', type=float, default=0.03)
    parser.add_argument('--high_conf_min', type=float, default=0.15)
    parser.add_argument('--high_conf_max', type=float, default=0.55)
    parser.add_argument('--high_conf_step', type=float, default=0.03)
    
    parser.add_argument('--merge_iou', type=float, default=0.5)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Grid 생성
    low_conf_values = np.arange(
        args.low_conf_min, 
        args.low_conf_max + args.low_conf_step/2, 
        args.low_conf_step
    ).tolist()
    high_conf_values = np.arange(
        args.high_conf_min, 
        args.high_conf_max + args.high_conf_step/2, 
        args.high_conf_step
    ).tolist()
    
    optimizer = Arch4ThresholdOptimizer(
        arch4_config_path=args.config,
        yolo_hr_weights_path=args.yolo_hr_weights,
        yolo_lr_weights_path=args.yolo_lr_weights,
        hr_data_yaml=args.hr_data_yaml,
        lr_data_yaml=args.lr_data_yaml,
        output_dir=args.output_dir,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    optimizer.grid_search(
        low_conf_values=low_conf_values,
        high_conf_values=high_conf_values,
        merge_iou=args.merge_iou,
        max_images=args.max_images 
    )
    
    optimizer.analyze_and_save(output_path=args.output)


if __name__ == '__main__':
    main()