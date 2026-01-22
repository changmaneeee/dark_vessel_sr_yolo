#!/usr/bin/env python
"""
=============================================================================
optimize_arch4_v2.py - Arch4 Threshold Grid Search 최적화 (로깅 강화 버전)
=============================================================================
- YOLO 스타일 실시간 로깅
- 각 run마다 즉시 결과 출력
- 최종 JSON 저장 보장 (중간 저장 포함)
- Upper/Lower bound 평가 생략 (이미 알려진 값 사용)
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
from datetime import datetime, timedelta
import itertools
import gc
import time

from ultralytics import YOLO
from src.models.pipelines.arch4_adaptive import Arch4Adaptive


# =============================================================================
# YOLO 스타일 로깅 함수들
# =============================================================================
class Colors:
    """ANSI 색상 코드"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def log_info(msg: str):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"{Colors.CYAN}[{timestamp}]{Colors.END} {msg}")


def log_success(msg: str):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"{Colors.GREEN}[{timestamp}] ✓{Colors.END} {msg}")


def log_warning(msg: str):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"{Colors.YELLOW}[{timestamp}] ⚠{Colors.END} {msg}")


def log_error(msg: str):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"{Colors.RED}[{timestamp}] ✗{Colors.END} {msg}")


def print_run_result(run_id: int, total: int, metrics: Dict, elapsed: float):
    """단일 run 결과를 YOLO 스타일로 출력"""
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}Run {run_id}/{total}{Colors.END} | "
          f"low_conf={Colors.CYAN}{metrics['low_conf']:.2f}{Colors.END}, "
          f"high_conf={Colors.CYAN}{metrics['high_conf']:.2f}{Colors.END} | "
          f"Time: {elapsed:.1f}s")
    print(f"{'='*80}")
    print(f"  {'Metric':<20} {'Value':>12} {'vs HR':>12} {'vs LR':>12}")
    print(f"  {'-'*56}")
    print(f"  {'mAP50':<20} {Colors.GREEN}{metrics['mAP50']:>12.4f}{Colors.END} "
          f"{metrics['recovery_mAP50']:>11.1f}% {metrics['improvement_mAP50']:>+11.4f}")
    print(f"  {'mAP50-95':<20} {metrics['mAP50-95']:>12.4f}")
    print(f"  {'Precision':<20} {metrics['precision']:>12.4f}")
    print(f"  {'Recall':<20} {metrics['recall']:>12.4f}")
    print(f"  {'F1':<20} {Colors.BLUE}{metrics['f1']:>12.4f}{Colors.END} "
          f"{metrics['recovery_f1']:>11.1f}% {metrics['improvement_f1']:>+11.4f}")
    print(f"  {'-'*56}")
    print(f"  {'Pass2 Ratio':<20} {Colors.YELLOW}{metrics['pass2_ratio']*100:>11.1f}%{Colors.END}")
    print(f"  {'Efficiency Score':<20} {metrics['efficiency_score']:>12.4f}")
    if metrics.get('psnr', 0) > 0:
        print(f"  {'PSNR (SR only)':<20} {metrics['psnr']:>12.2f} dB")
        print(f"  {'SSIM (SR only)':<20} {metrics['ssim']:>12.4f}")
    print(f"{'='*80}\n")


def print_progress(current: int, total: int, start_time: float):
    """진행 상황 출력"""
    elapsed = time.time() - start_time
    avg_per_run = elapsed / current if current > 0 else 0
    remaining = (total - current) * avg_per_run
    
    print(f"{Colors.CYAN}📊 Progress: {current}/{total} ({100*current/total:.1f}%) | "
          f"Elapsed: {timedelta(seconds=int(elapsed))} | "
          f"ETA: {timedelta(seconds=int(remaining))}{Colors.END}\n")


# =============================================================================
# 유틸리티 함수들
# =============================================================================
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


# =============================================================================
# 메인 클래스
# =============================================================================
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
        yolo_lr_weights_path: str = None,
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
        
        print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"{Colors.HEADER}📊 Arch4 Threshold 최적화 v2{Colors.END}")
        print(f"{'='*80}")
        log_info(f"HR images: {self.hr_val_images_dir}")
        log_info(f"LR images: {self.lr_val_images_dir}")
        log_info(f"YOLO HR weights: {self.yolo_hr_weights_path}")
        log_info(f"YOLO LR weights: {self.yolo_lr_weights_path}")
        
        # 평가용 YOLO
        log_info("평가용 YOLO 로드 중...")
        self.yolo_eval = YOLO(yolo_hr_weights_path, verbose=False)
        log_success("YOLO 로드 완료")
        
        # Arch4 config 로드
        log_info("Arch4 config 로드 중...")
        self.config = load_config(arch4_config_path)
        self.arch4_config_path = arch4_config_path
        log_success("Arch4 config 로드 완료")
        
        self.transform = T.ToTensor()
        
        # 결과 저장
        self.results = []
        self.start_time = None
    
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
        run_id: int = 0,
        total_runs: int = 1
    ) -> Dict[str, Any]:
        """단일 threshold 조합 평가"""
        
        run_start = time.time()
        
        # Arch4 생성
        arch4 = self._create_arch4_with_thresholds(low_conf, high_conf, merge_iou)
        
        # 출력 디렉토리
        run_output_dir = self.output_dir / f'run_{run_id:03d}'
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
        
        # Progress bar
        pbar = tqdm(lr_images, 
                    desc=f"Run {run_id}/{total_runs} (low={low_conf:.2f}, high={high_conf:.2f})",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                    leave=False)
        
        for img_path in pbar:
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
        
        pbar.close()
        
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
        
        # Ultralytics 평가
        results = self.yolo_eval.val(
            data=str(sr_yaml_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            workers=0,  # OOM 방지
            batch=4
        )
        
        # 결과 정리
        pass2_ratio = np.mean(pass2_triggered_list) if pass2_triggered_list else 0.0
        
        metrics = {
            'run_id': run_id,
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
        
        # 효율성 점수
        if pass2_ratio > 0:
            metrics['efficiency_score'] = metrics['improvement_mAP50'] / pass2_ratio
        else:
            metrics['efficiency_score'] = 0.0
        
        elapsed = time.time() - run_start
        metrics['elapsed_seconds'] = elapsed
        
        # 결과 출력
        print_run_result(run_id, total_runs, metrics, elapsed)
        
        # 메모리 정리
        del arch4
        torch.cuda.empty_cache()
        gc.collect()
        
        return metrics
    
    def grid_search(
        self,
        low_conf_values: List[float],
        high_conf_values: List[float],
        merge_iou: float = 0.5,
        max_images: int = None,
        output_path: str = 'results/arch4_threshold_optimization_v2.json'
    ) -> List[Dict[str, Any]]:
        """Grid Search 실행"""
        
        self.start_time = time.time()
        
        # 유효한 조합만 필터링 (low_conf < high_conf)
        combinations = [
            (low, high) for low, high in itertools.product(low_conf_values, high_conf_values)
            if low < high
        ]
        total_runs = len(combinations)
        
        # 이미지 수 확인
        lr_images = list(self.lr_val_images_dir.glob('*.jpg'))
        num_images = min(len(lr_images), max_images) if max_images else len(lr_images)
        
        print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"{Colors.HEADER}📊 Grid Search 시작{Colors.END}")
        print(f"{'='*80}")
        print(f"  Low conf range:  {min(low_conf_values):.2f} ~ {max(low_conf_values):.2f}")
        print(f"  High conf range: {min(high_conf_values):.2f} ~ {max(high_conf_values):.2f}")
        print(f"  Total combinations: {Colors.CYAN}{total_runs}{Colors.END}")
        print(f"  Images per run: {Colors.GREEN}{num_images}{Colors.END}")
        
        # 예상 시간 (run당 약 35분 기준)
        est_seconds = total_runs * 35 * 60 * (num_images / 28884) if num_images < 28884 else total_runs * 35 * 60
        print(f"  Estimated time: ~{est_seconds/3600:.1f} hours")
        print(f"{'='*80}\n")
        
        results = []
        
        for run_id, (low_conf, high_conf) in enumerate(combinations, 1):
            try:
                metrics = self.run_single_evaluation(
                    low_conf=low_conf,
                    high_conf=high_conf,
                    merge_iou=merge_iou,
                    max_images=max_images,
                    run_id=run_id,
                    total_runs=total_runs
                )
                results.append(metrics)
                
                # 진행 상황 출력
                print_progress(run_id, total_runs, self.start_time)
                
                # ★ 중간 결과 저장 (매 run마다)
                self._save_intermediate_results(results, output_path)
                
            except Exception as e:
                log_error(f"Run {run_id} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.results = results
        log_success(f"Grid Search 완료! {len(results)}/{total_runs} successful runs")
        
        return results
    
    def _save_intermediate_results(self, results: List[Dict], output_path: str):
        """중간 결과 저장 (각 run마다 호출)"""
        if not results:
            return
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 현재까지 best 계산
        best_by_mAP50 = max(results, key=lambda x: x['mAP50'])
        best_by_f1 = max(results, key=lambda x: x['f1'])
        
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'status': 'in_progress',
            'completed_runs': len(results),
            'baselines': self.KNOWN_BASELINES,
            'current_best_mAP50': {
                'low_conf': best_by_mAP50['low_conf'],
                'high_conf': best_by_mAP50['high_conf'],
                'mAP50': best_by_mAP50['mAP50'],
                'f1': best_by_mAP50['f1'],
                'pass2_ratio': best_by_mAP50['pass2_ratio']
            },
            'current_best_f1': {
                'low_conf': best_by_f1['low_conf'],
                'high_conf': best_by_f1['high_conf'],
                'mAP50': best_by_f1['mAP50'],
                'f1': best_by_f1['f1'],
                'pass2_ratio': best_by_f1['pass2_ratio']
            },
            'all_results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def analyze_and_save(self, output_path: str = 'results/arch4_threshold_optimization_v2.json'):
        """최종 결과 분석 및 저장"""
        
        if not self.results:
            log_error("결과가 없습니다. grid_search를 먼저 실행하세요.")
            return
        
        # 정렬 기준별 Best 찾기
        best_by_mAP50 = max(self.results, key=lambda x: x['mAP50'])
        best_by_f1 = max(self.results, key=lambda x: x['f1'])
        best_by_efficiency = max(self.results, key=lambda x: x['efficiency_score'])
        
        # 균형점 찾기
        for r in self.results:
            r['balanced_score'] = r['mAP50'] * (1 - r['pass2_ratio'] * 0.3)
        best_balanced = max(self.results, key=lambda x: x['balanced_score'])
        
        print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"{Colors.HEADER}📊 최종 결과 분석{Colors.END}")
        print(f"{'='*80}")
        
        print(f"\n{Colors.CYAN}[Baseline 참고]{Colors.END}")
        print(f"  HR: mAP50={self.KNOWN_BASELINES['hr']['mAP50']:.4f}, F1={self.KNOWN_BASELINES['hr']['f1']:.4f}")
        print(f"  LR: mAP50={self.KNOWN_BASELINES['lr']['mAP50']:.4f}, F1={self.KNOWN_BASELINES['lr']['f1']:.4f}")
        
        print(f"\n{Colors.GREEN}[Best by mAP50]{Colors.END}")
        print(f"  low_conf={best_by_mAP50['low_conf']:.2f}, high_conf={best_by_mAP50['high_conf']:.2f}")
        print(f"  mAP50: {best_by_mAP50['mAP50']:.4f}, F1: {best_by_mAP50['f1']:.4f}")
        print(f"  Pass2 트리거: {best_by_mAP50['pass2_ratio']*100:.1f}%")
        
        print(f"\n{Colors.BLUE}[Best by F1]{Colors.END}")
        print(f"  low_conf={best_by_f1['low_conf']:.2f}, high_conf={best_by_f1['high_conf']:.2f}")
        print(f"  mAP50: {best_by_f1['mAP50']:.4f}, F1: {best_by_f1['f1']:.4f}")
        print(f"  Pass2 트리거: {best_by_f1['pass2_ratio']*100:.1f}%")
        
        print(f"\n{Colors.YELLOW}[Best by Efficiency]{Colors.END}")
        print(f"  low_conf={best_by_efficiency['low_conf']:.2f}, high_conf={best_by_efficiency['high_conf']:.2f}")
        print(f"  mAP50: {best_by_efficiency['mAP50']:.4f}, F1: {best_by_efficiency['f1']:.4f}")
        print(f"  Pass2 트리거: {best_by_efficiency['pass2_ratio']*100:.1f}%")
        print(f"  효율성 점수: {best_by_efficiency['efficiency_score']:.4f}")
        
        print(f"\n{Colors.HEADER}[Best Balanced]{Colors.END}")
        print(f"  low_conf={best_balanced['low_conf']:.2f}, high_conf={best_balanced['high_conf']:.2f}")
        print(f"  mAP50: {best_balanced['mAP50']:.4f}, F1: {best_balanced['f1']:.4f}")
        print(f"  Pass2 트리거: {best_balanced['pass2_ratio']*100:.1f}%")
        
        # 전체 결과 테이블
        print(f"\n{Colors.CYAN}[전체 결과 요약 (mAP50 순)]{Colors.END}")
        print(f"{'low':>6} {'high':>6} {'mAP50':>8} {'F1':>8} {'Pass2%':>8} {'Eff':>8}")
        print("-" * 50)
        
        sorted_results = sorted(self.results, key=lambda x: x['mAP50'], reverse=True)
        for r in sorted_results[:10]:  # Top 10만 출력
            print(f"{r['low_conf']:>6.2f} {r['high_conf']:>6.2f} {r['mAP50']:>8.4f} "
                  f"{r['f1']:>8.4f} {r['pass2_ratio']*100:>7.1f}% {r['efficiency_score']:>8.4f}")
        
        if len(sorted_results) > 10:
            print(f"  ... and {len(sorted_results) - 10} more results")
        
        # 최종 저장
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'status': 'completed',
            'total_runs': len(self.results),
            'baselines': self.KNOWN_BASELINES,
            'best_by_mAP50': {
                'low_conf': best_by_mAP50['low_conf'],
                'high_conf': best_by_mAP50['high_conf'],
                'metrics': {k: v for k, v in best_by_mAP50.items() if k not in ['run_id']}
            },
            'best_by_f1': {
                'low_conf': best_by_f1['low_conf'],
                'high_conf': best_by_f1['high_conf'],
                'metrics': {k: v for k, v in best_by_f1.items() if k not in ['run_id']}
            },
            'best_by_efficiency': {
                'low_conf': best_by_efficiency['low_conf'],
                'high_conf': best_by_efficiency['high_conf'],
                'metrics': {k: v for k, v in best_by_efficiency.items() if k not in ['run_id']}
            },
            'best_balanced': {
                'low_conf': best_balanced['low_conf'],
                'high_conf': best_balanced['high_conf'],
                'metrics': {k: v for k, v in best_balanced.items() if k not in ['run_id']}
            },
            'all_results': self.results
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        log_success(f"최종 결과 저장: {output_path}")
        
        return output_data


def main():
    parser = argparse.ArgumentParser(description='Arch4 Threshold Grid Search v2')
    
    parser.add_argument('--config', type=str, 
                        default='configs/experiment/arch4_adaptive.yaml')
    parser.add_argument('--yolo_hr_weights', type=str, 
                        default='weights/yolohr/8s/best.pt')
    parser.add_argument('--yolo_lr_weights', type=str, 
                        default=None)
    parser.add_argument('--hr_data_yaml', type=str,
                        default='/home/changmin/smart_airbus_data/data.yaml')
    parser.add_argument('--lr_data_yaml', type=str,
                        default='/home/changmin/smart_airbus_data_lr/data.yaml')
    parser.add_argument('--output_dir', type=str,
                        default='/tmp/arch4_optimize')
    parser.add_argument('--output', type=str,
                        default='results/arch4_threshold_optimization_v2.json')
    parser.add_argument('--max_images', type=int, default=None,
                        help='평가할 최대 이미지 수 (None=전체 val)')
    
    # Grid Search 범위
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
        max_images=args.max_images,
        output_path=args.output
    )
    
    optimizer.analyze_and_save(output_path=args.output)


if __name__ == '__main__':
    main()