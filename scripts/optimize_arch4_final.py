#!/usr/bin/env python
"""
=============================================================================
optimize_arch4_final.py - Upsampling Strategy (Recall Booster)
=============================================================================
[핵심 전략]
1. 문제: LR 모델의 Recall이 0.57로 낮음 -> 작은 배를 놓침
2. 해결: Pass 1 입력으로 'Upsampled Image (640px)'를 강제로 주입
3. 효과: 흐릿하지만 객체가 커져서 Recall이 상승함 (Inference.py 성공 요인)
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
import yaml
from typing import Dict, Any, List
import gc
import itertools

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.models.pipelines.arch4_adaptive import Arch4Adaptive

def load_config(config_path: str) -> Any:
    with open(config_path, 'r') as f: config_dict = yaml.safe_load(f)
    from types import SimpleNamespace
    def dict_to_namespace(d):
        if isinstance(d, dict):
            for k, v in d.items(): d[k] = dict_to_namespace(v)
            return SimpleNamespace(**d)
        return d
    return dict_to_namespace(config_dict)

# ============================================================================
# ★ Upsampling Logic Wrapper (Inference.py와 동일한 로직)
# ============================================================================
class Arch4WinningLogic(Arch4Adaptive):
    
    def _classify_image_dynamic(self, detections: Dict, low: float, high: float) -> str:
        scores = detections.get('scores', torch.tensor([]))
        if len(scores) == 0: return 'zero_detection'
        
        # low보다 크고 high보다 작은 '애매한' 녀석이 있으면 SR
        has_uncertain = ((scores >= low) & (scores < high)).any()
        
        if has_uncertain:
            return 'need_sr'
        else:
            return 'confirmed'

    @torch.no_grad()
    def forward_optimized(self, lr_image: torch.Tensor, low_conf: float, high_conf: float, sr_on_zero: bool) -> Dict[str, Any]:
        """
        [성공 전략]
        1. 160px LR 이미지를 640px로 Upsampling
        2. 640px 이미지를 LR 모델(192px 학습)에 입력 -> Ultralytics가 알아서 처리하며 Recall 상승
        """
        self.eval()
        B = lr_image.size(0)
        
        # 1. Upsampling (160 -> 640)
        lr_upsampled = F.interpolate(lr_image, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)

        # 2. Pass 1 Detect (Upsampled Image 사용!)
        pass1_detections = self.detector_lr.predict(
            lr_upsampled, # ★ 핵심: 큰 이미지를 넣어서 작은 배도 찾게 만듦
            conf=low_conf, 
            iou=self.nms_iou_threshold
        )

        # 3. 좌표 스케일링 삭제 (입력이 이미 640px이므로 좌표도 640px 기준임)

        final_detections = []
        action_counts = {'confirmed': 0, 'need_sr': 0, 'zero_detection': 0}

        for i in range(B):
            det = pass1_detections[i]
            action = self._classify_image_dynamic(det, low_conf, high_conf)
            action_counts[action] += 1
            
            if action == 'confirmed':
                scores = det['scores']
                if len(scores) > 0:
                    mask = scores >= high_conf
                    final_detections.append({'boxes': det['boxes'][mask], 'scores': det['scores'][mask], 'classes': det['classes'][mask]})
                else: final_detections.append(det)
            
            elif action == 'need_sr':
                hr_image = self._apply_full_sr(lr_image[i:i+1])
                pass2_result = self.detector_hr.predict(hr_image, conf=self.final_conf_threshold, iou=self.nms_iou_threshold)[0]
                final_detections.append(pass2_result)
            
            else: # zero_detection
                if sr_on_zero: # True면 묻지도 따지지도 않고 SR
                    hr_image = self._apply_full_sr(lr_image[i:i+1])
                    pass2_result = self.detector_hr.predict(hr_image, conf=self.final_conf_threshold, iou=self.nms_iou_threshold)[0]
                    final_detections.append(pass2_result)
                else: # False면 생략 (Efficiency Mode)
                    final_detections.append({'boxes': torch.tensor([], device=self.device), 'scores': torch.tensor([], device=self.device), 'classes': torch.tensor([], device=self.device)})

        return {'detections': final_detections, 'action_counts': action_counts}

class Arch4ThresholdOptimizer:
    def __init__(self, args):
        self.device = args.device
        self.final_conf_threshold = args.final_conf
        self.iou_threshold = args.iou
        self.output_dir = Path(args.output_dir)
        self.config = load_config(args.config)
        with open(args.lr_data_yaml, 'r') as f: lr_config = yaml.safe_load(f)
        with open(args.hr_data_yaml, 'r') as f: hr_config = yaml.safe_load(f)
        self.lr_val_images_dir = Path(lr_config.get('path', '')) / 'images' / 'val'
        self.hr_val_labels_dir = Path(hr_config.get('path', '')) / 'labels' / 'val'
        self.transform = T.ToTensor()

    def _load_gt_targets(self, img_name, img_w, img_h):
        label_path = self.hr_val_labels_dir / f"{Path(img_name).stem}.txt"
        boxes, labels = [], []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 5:
                        cls, cx, cy, w, h = parts[0:5]
                        x1, y1, x2, y2 = (cx-w/2)*img_w, (cy-h/2)*img_h, (cx+w/2)*img_w, (cy+h/2)*img_h
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(cls))
        return {'boxes': torch.tensor(boxes, device=self.device) if boxes else torch.tensor([], device=self.device), 'labels': torch.tensor(labels, device=self.device) if labels else torch.tensor([], device=self.device)}

    def run_single_evaluation(self, low_conf, high_conf, sr_on_zero, max_images, run_id):
        if low_conf >= high_conf: return None
        mode_str = "SR_ON_ZERO" if sr_on_zero else "SKIP_ZERO"
        print(f"\n>>> [Run {run_id}] Low={low_conf:.4f}, High={high_conf:.2f}, Mode={mode_str}")
        
        arch4 = Arch4WinningLogic(self.config).to(self.device)
        arch4.eval()
        metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox').to(self.device)
        lr_images = sorted(list(self.lr_val_images_dir.glob('*.jpg')))
        if max_images: lr_images = lr_images[:max_images]
        
        action_counts = {'confirmed': 0, 'need_sr': 0, 'zero_detection': 0}
        
        for img_path in tqdm(lr_images, desc="Evaluating", leave=False):
            img = Image.open(img_path).convert('RGB')
            w_lr, h_lr = img.size
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            target = self._load_gt_targets(img_path.name, w_lr * 4, h_lr * 4)

            with torch.no_grad():
                result = arch4.forward_optimized(img_tensor, low_conf, high_conf, sr_on_zero)
            
            counts = result['action_counts']
            for k in action_counts: action_counts[k] += counts[k]
            
            det = result['detections'][0]
            preds = [{'boxes': det['boxes'], 'scores': det['scores'], 'labels': det['classes'].long()}]
            metric.update(preds, [target])

        mAP_result = metric.compute()
        mAP50 = mAP_result['map_50'].item()
        total = len(lr_images)
        
        # SR Saved: Confirmed는 무조건 절약, Zero Det는 sr_on_zero=False일 때만 절약
        sr_saved = action_counts['confirmed']
        if not sr_on_zero: sr_saved += action_counts['zero_detection']
            
        print(f"  [Result] mAP50: {mAP50:.4f}, Saved: {sr_saved/total*100:.1f}%")
        
        return {
            'low_conf': low_conf, 'high_conf': high_conf, 'sr_on_zero': sr_on_zero,
            'mAP50': mAP50, 'sr_saved_ratio': sr_saved / total * 100
        }

    def grid_search(self, low_confs, high_confs, max_images):
        results = []
        run_id = 0
        for sr_zero in [False, True]: # Full Spectrum 탐색
            for low, high in itertools.product(low_confs, high_confs):
                if low >= high: continue
                res = self.run_single_evaluation(low, high, sr_zero, max_images, run_id)
                if res:
                    results.append(res)
                    self._save_results(results)
                run_id += 1
        self._print_final_summary(results)

    def _save_results(self, results):
        output_path = self.output_dir / 'arch4_optimization_results.json'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f: json.dump({'results': results}, f, indent=2)

    def _print_final_summary(self, results):
        print("\n" + "="*80)
        print("🚀 Final Optimization Summary (Full Spectrum)")
        print("="*80)
        print(f"{'Mode':<10} {'Low':<6} {'High':<6} | {'mAP50':<8} | {'SR Saved':<10}")
        print("-" * 80)
        sorted_res = sorted(results, key=lambda x: x['mAP50'], reverse=True)
        for r in sorted_res:
            mode = "SR_ZERO" if r['sr_on_zero'] else "SKIP"
            print(f"{mode:<10} {r['low_conf']:<6.4f} {r['high_conf']:<6.2f} | {r['mAP50']:<8.4f} | {r['sr_saved_ratio']:<9.1f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/experiment/arch4_adaptive.yaml')
    parser.add_argument('--hr_data_yaml', type=str, required=True)
    parser.add_argument('--lr_data_yaml', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/arch4_eval')
    parser.add_argument('--final_conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_images', type=int, default=None)
    
    # Grid Search 범위
    parser.add_argument('--low_min', type=float, default=0.001)
    parser.add_argument('--low_max', type=float, default=0.05)
    parser.add_argument('--low_step', type=float, default=0.02)
    
    parser.add_argument('--high_min', type=float, default=0.3)
    parser.add_argument('--high_max', type=float, default=0.6)
    parser.add_argument('--high_step', type=float, default=0.1)

    args = parser.parse_args()
    optimizer = Arch4ThresholdOptimizer(args)
    low_confs = np.arange(args.low_min, args.low_max + 0.0001, args.low_step).tolist()
    high_confs = np.arange(args.high_min, args.high_max + 0.001, args.high_step).tolist()
    print(f"Grid Search with SR_ON_ZERO Toggle")
    optimizer.grid_search(low_confs, high_confs, args.max_images)

if __name__ == '__main__':
    main()