#!/usr/bin/env python
"""
=============================================================================
optimize_arch4_thresholds.py - Direct Evaluation using TorchMetrics
=============================================================================

[핵심 변경 사항]
1. Ultralytics YOLO.val() 제거 -> Arch4의 예측값 직접 사용
2. TorchMetrics 사용 -> mAP 직접 계산
3. 불필요한 이미지 저장/로드 과정 삭제 (속도 향상)
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
import yaml
from typing import Dict, Any, List
from types import SimpleNamespace
from datetime import datetime
import gc

# ★ TorchMetrics: 정확한 mAP 계산을 위한 라이브러리
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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

class Arch4ThresholdOptimizer:
    
    # 비교를 위한 Baseline 점수 (참고용)
    KNOWN_BASELINES = {
        'hr': {'mAP50': 0.780},
        'lr': {'mAP50': 0.693},
        'arch0': {'mAP50': 0.731}
    }
    
    def __init__(self, args):
        self.device = args.device
        self.final_conf_threshold = args.final_conf
        self.iou_threshold = args.iou
        self.output_dir = Path(args.output_dir)
        self.config = load_config(args.config)
        
        # 데이터 경로 로드
        with open(args.lr_data_yaml, 'r') as f:
            lr_config = yaml.safe_load(f)
        with open(args.hr_data_yaml, 'r') as f:
            hr_config = yaml.safe_load(f) # GT 라벨은 HR 기준
            
        lr_path = Path(lr_config.get('path', ''))
        hr_path = Path(hr_config.get('path', ''))
        
        self.lr_val_images_dir = lr_path / 'images' / 'val'
        self.hr_val_labels_dir = hr_path / 'labels' / 'val'
        
        print(f"\n{'='*70}")
        print(f"📊 평가 데이터 경로 확인")
        print(f"{'='*70}")
        print(f"Input Images (LR): {self.lr_val_images_dir}")
        print(f"GT Labels (HR):    {self.hr_val_labels_dir}")

        self.transform = T.ToTensor()
        self.results = []

    def _load_gt_targets(self, img_name: str, img_w: int, img_h: int) -> Dict[str, torch.Tensor]:
        """
        YOLO 포맷(.txt) 라벨을 로드하여 절대 좌표(Pixel)로 변환
        Args:
            img_w, img_h: 기준이 되는 HR 이미지의 크기
        """
        label_path = self.hr_val_labels_dir / f"{Path(img_name).stem}.txt"
        
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, w, h = parts[1:5]
                        
                        # YOLO Normalized (0~1) -> Pixel Coordinates (Absolute)
                        x1 = (cx - w/2) * img_w
                        y1 = (cy - h/2) * img_h
                        x2 = (cx + w/2) * img_w
                        y2 = (cy + h/2) * img_h
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls)
        
        if not boxes:
            return {
                'boxes': torch.tensor([], device=self.device),
                'labels': torch.tensor([], device=self.device)
            }
            
        return {
            'boxes': torch.tensor(boxes, device=self.device),
            'labels': torch.tensor(labels, device=self.device)
        }

    def run_single_evaluation(self, high_conf: float, max_images: int = None, run_id: int = 0):
        print(f"\n>>> [Run {run_id}] high_conf={high_conf:.2f} 평가 시작...")
        
        # Arch4 모델 생성 및 초기화
        arch4 = Arch4Adaptive(self.config)
        arch4.set_thresholds(
            pass1_conf=0.01,
            high_conf=high_conf,
            final_conf=self.final_conf_threshold,
            nms_iou=self.iou_threshold,
            sr_on_zero=False
        )
        arch4 = arch4.to(self.device)
        arch4.eval()
        
        # Metric 초기화 (mAP 계산기)
        metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox').to(self.device)
        
        lr_images = sorted(list(self.lr_val_images_dir.glob('*.jpg')))
        if max_images:
            lr_images = lr_images[:max_images]

        action_counts = {'confirmed': 0, 'need_sr': 0, 'zero_detection': 0}
        
        start_time = datetime.now()
        
        # === 배치 처리 루프 ===
        for img_path in tqdm(lr_images, desc=f"Eval Conf {high_conf:.2f}"):
            # 1. 이미지 로드
            img = Image.open(img_path).convert('RGB')
            w_lr, h_lr = img.size
            
            # Arch4는 내부적으로 4배 Upscaling을 수행하므로, 
            # GT 좌표도 4배 큰 HR 기준(w_lr * 4)으로 로드해야 매칭이 됨.
            w_hr, h_hr = w_lr * 4, h_lr * 4 
            
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # 2. GT 라벨 로드 (HR 스케일)
            target = self._load_gt_targets(img_path.name, w_hr, h_hr)

            # 3. Arch4 추론 (여기서 result['detections']에 정답이 들어있음)
            with torch.no_grad():
                result = arch4.forward(img_tensor)
            
            # 4. 통계 집계
            action = result['actions'][0]
            action_counts[action] += 1
            
            # 5. 예측 결과 포맷 변환 (Metric 입력용)
            det = result['detections'][0]
            
            preds = [
                {
                    'boxes': det['boxes'],    # Arch4가 찾은 박스 (HR 스케일)
                    'scores': det['scores'],  # Arch4가 확신한 점수
                    'labels': det['classes'].long()
                }
            ]
            
            # 6. Metric 업데이트 (Arch4 예측값 vs 실제 정답)
            # ★ 핵심: 이미지를 다시 YOLO에 넣는게 아니라, Arch4가 찾은 값을 그대로 채점함!
            metric.update(preds, [target])

        # === 결과 계산 ===
        mAP_result = metric.compute()
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # 수치 정리
        mAP50 = mAP_result['map_50'].item()
        mAP50_95 = mAP_result['map'].item()
        
        # 통계 계산
        total = len(lr_images)
        sr_saved = action_counts['confirmed'] + action_counts['zero_detection']
        recovery = (mAP50 / self.KNOWN_BASELINES['arch0']['mAP50']) * 100
        
        print(f"  [Result] mAP50: {mAP50:.4f}")
        print(f"  [Result] SR Saved: {sr_saved/total*100:.1f}% (SR 수행: {action_counts['need_sr']}장)")

        metrics = {
            'high_conf': high_conf,
            'mAP50': mAP50,
            'mAP50-95': mAP50_95,
            'full_sr_ratio': action_counts['need_sr'] / total * 100,
            'sr_saved_ratio': sr_saved / total * 100,
            'recovery_arch0': recovery,
            'efficiency_score': recovery * (sr_saved / total),
            'elapsed_seconds': elapsed
        }
        
        # 메모리 정리
        del arch4, metric
        torch.cuda.empty_cache()
        gc.collect()
        
        return metrics

    def grid_search(self, high_conf_values, max_images):
        results = []
        for i, val in enumerate(high_conf_values):
            res = self.run_single_evaluation(val, max_images, i)
            results.append(res)
            
            # 중간 저장
            self._save_results(results)
        
        self._print_final_summary(results)

    def _save_results(self, results):
        output_path = self.output_dir / 'arch4_optimization_results.json'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({'results': results}, f, indent=2)

    def _print_final_summary(self, results):
        print("\n" + "="*60)
        print("🚀 최종 최적화 결과 요약")
        print("="*60)
        print(f"{'Conf':<8} | {'mAP50':<8} | {'Arch0 대비':<12} | {'SR 절약률':<10}")
        print("-" * 60)
        
        sorted_res = sorted(results, key=lambda x: x['high_conf'])
        for r in sorted_res:
            print(f"{r['high_conf']:<8.2f} | {r['mAP50']:<8.4f} | {r['recovery_arch0']:<11.1f}% | {r['sr_saved_ratio']:<10.1f}%")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Arch4 Direct Evaluation')
    
    # 설정 파일 경로
    parser.add_argument('--config', type=str, default='configs/experiment/arch4_adaptive.yaml')
    parser.add_argument('--hr_data_yaml', type=str, required=True, help='HR 데이터셋(GT) 경로')
    parser.add_argument('--lr_data_yaml', type=str, required=True, help='LR 데이터셋(Input) 경로')
    
    # 평가 옵션
    parser.add_argument('--output_dir', type=str, default='results/arch4_eval')
    parser.add_argument('--final_conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_images', type=int, default=None, help='테스트할 이미지 수 (None이면 전체)')
    
    # Grid Search 범위
    parser.add_argument('--conf_min', type=float, default=0.3)
    parser.add_argument('--conf_max', type=float, default=0.7)
    parser.add_argument('--conf_step', type=float, default=0.1)

    args = parser.parse_args()
    
    optimizer = Arch4ThresholdOptimizer(args)
    
    # Grid Search 실행
    high_confs = np.arange(args.conf_min, args.conf_max + 0.001, args.conf_step).tolist()
    optimizer.grid_search(high_confs, args.max_images)

if __name__ == '__main__':
    main()