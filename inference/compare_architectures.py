#!/usr/bin/env python
"""
=============================================================================
compare_architectures.py - Arch 0, 2, 4 비교 실험
=============================================================================
동일한 데이터셋에서 3개 아키텍처 성능 비교

[비교 메트릭]
- Detection: mAP@0.5, Precision, Recall
- Efficiency: Inference time, SR usage ratio
- Quality: PSNR (SR 적용된 경우)

사용법:
    python compare_architectures.py \
        --sr_type mamba \
        --sr_weights /path/to/sr.pth \
        --yolo_weights /path/to/yolo.pt \
        --hr_root /path/to/hr_dataset \
        --lr_root /path/to/lr_dataset \
        --output ./comparison_results
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import time

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.inference import Arch0Inference, Arch2Inference, Arch4Inference, Arch5BInference


def load_labels(label_path: Path) -> np.ndarray:
    """YOLO format label 로드"""
    if not label_path.exists():
        return np.zeros((0, 5))
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append([float(x) for x in parts[:5]])
    
    return np.array(labels) if labels else np.zeros((0, 5))


def xywh_to_xyxy(boxes: np.ndarray, img_size: int) -> np.ndarray:
    """normalized xywh → xyxy"""
    if len(boxes) == 0:
        return np.zeros((0, 4))
    
    x, y, w, h = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    x1 = (x - w / 2) * img_size
    y1 = (y - h / 2) * img_size
    x2 = (x + w / 2) * img_size
    y2 = (y + h / 2) * img_size
    
    return np.stack([x1, y1, x2, y2], axis=1)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def calculate_metrics(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """Detection 메트릭 계산"""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for preds, scores, gts in zip(pred_boxes, pred_scores, gt_boxes):
        if len(gts) == 0:
            total_fp += len(preds)
            continue
        
        if len(preds) == 0:
            total_fn += len(gts)
            continue
        
        # Sort by confidence
        sorted_idx = np.argsort(-scores)
        preds = preds[sorted_idx]
        
        gt_matched = np.zeros(len(gts), dtype=bool)
        
        for pred in preds:
            best_iou = 0
            best_idx = -1
            
            for i, gt in enumerate(gts):
                if gt_matched[i]:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_iou >= iou_threshold:
                total_tp += 1
                gt_matched[best_idx] = True
            else:
                total_fp += 1
        
        total_fn += np.sum(~gt_matched)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


def calculate_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """PSNR 계산"""
    mse = np.mean((pred.astype(float) - target.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255 ** 2 / mse)


def run_comparison(
    engines: Dict[str, Any],
    hr_root: str,
    lr_root: str,
    split: str = 'val',
    max_images: int = None
) -> Dict[str, Dict[str, Any]]:
    """
    아키텍처 비교 실험 실행
    
    Args:
        engines: {'arch0': engine, 'arch2': engine, 'arch4': engine}
        hr_root: HR 데이터셋 경로
        lr_root: LR 데이터셋 경로
        split: 'train' or 'val'
        max_images: 최대 이미지 수 (None이면 전체)
    
    Returns:
        아키텍처별 결과
    """
    hr_img_dir = Path(hr_root) / 'images' / split
    lr_img_dir = Path(lr_root) / 'images' / split
    label_dir = Path(hr_root) / 'labels' / split
    
    # 이미지 목록
    image_files = sorted(list(lr_img_dir.glob('*.jpg')) + list(lr_img_dir.glob('*.png')))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n[Comparison] {len(image_files)} images from {split} split")
    
    # 결과 저장
    results = {name: {
        'pred_boxes': [],
        'pred_scores': [],
        'gt_boxes': [],
        'sr_applied': 0,
        'total_time': 0.0,
        'psnr_sum': 0.0,
        'psnr_count': 0
    } for name in engines.keys()}
    
    for img_path in tqdm(image_files, desc="Comparing"):
        stem = img_path.stem
        
        # 이미지 로드
        lr_img = cv2.imread(str(img_path))
        
        hr_path = hr_img_dir / f"{stem}.jpg"
        if not hr_path.exists():
            hr_path = hr_img_dir / f"{stem}.png"
        hr_img = cv2.imread(str(hr_path)) if hr_path.exists() else None
        
        # Label 로드
        label_path = label_dir / f"{stem}.txt"
        gt_labels = load_labels(label_path)
        
        # HR 크기 기준 GT boxes
        if hr_img is not None:
            hr_size = hr_img.shape[0]
        else:
            hr_size = lr_img.shape[0] * 4  # 기본 upscale factor
        
        gt_xyxy = xywh_to_xyxy(gt_labels, hr_size)
        
        # 각 아키텍처 실행
        for arch_name, engine in engines.items():
            result = engine.inference(lr_img)
            
            # 결과 수집
            results[arch_name]['total_time'] += result['inference_time']
            
            if result['sr_applied']:
                results[arch_name]['sr_applied'] += 1
            
            # Detection 결과
            detections = result['detections']
            
            # Arch5B는 dict 형식, 나머지는 ultralytics 객체
            if isinstance(detections, dict):
                # Arch5B
                pred_xyxy = detections.get('boxes', np.zeros((0, 4)))
                pred_conf = detections.get('scores', np.zeros(0))
            elif detections and hasattr(detections, 'boxes') and len(detections.boxes) > 0:
                # Arch0/2/4 (ultralytics)
                pred_xyxy = detections.boxes.xyxy.cpu().numpy()
                pred_conf = detections.boxes.conf.cpu().numpy()
            else:
                pred_xyxy = np.zeros((0, 4))
                pred_conf = np.zeros(0)
            
            results[arch_name]['pred_boxes'].append(pred_xyxy)
            results[arch_name]['pred_scores'].append(pred_conf)
            results[arch_name]['gt_boxes'].append(gt_xyxy)
            
            # PSNR (SR 적용된 경우)
            if result['sr_applied'] and hr_img is not None:
                sr_img = result['sr_image']
                if sr_img.shape[:2] != hr_img.shape[:2]:
                    sr_img = cv2.resize(sr_img, (hr_img.shape[1], hr_img.shape[0]))
                psnr = calculate_psnr(sr_img, hr_img)
                results[arch_name]['psnr_sum'] += psnr
                results[arch_name]['psnr_count'] += 1
    
    # 메트릭 계산
    final_results = {}
    
    for arch_name, data in results.items():
        metrics = calculate_metrics(
            data['pred_boxes'],
            data['pred_scores'],
            data['gt_boxes']
        )
        
        n_images = len(image_files)
        
        final_results[arch_name] = {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'sr_ratio': data['sr_applied'] / n_images,
            'avg_time_ms': (data['total_time'] / n_images) * 1000,
            'total_time': data['total_time'],
            'avg_psnr': data['psnr_sum'] / data['psnr_count'] if data['psnr_count'] > 0 else 0
        }
    
    return final_results


def print_comparison(results: Dict[str, Dict[str, Any]]):
    """비교 결과 출력"""
    print("\n" + "=" * 80)
    print("📊 Architecture Comparison Results")
    print("=" * 80)
    
    # 아키텍처 목록 (존재하는 것만)
    archs = [a for a in ['arch0', 'arch2', 'arch4', 'arch5b'] if a in results]
    
    # 테이블 헤더
    header = f"{'Metric':<20}"
    for arch in archs:
        header += f" {arch.upper():<15}"
    print(f"\n{header}")
    print("-" * (20 + 16 * len(archs)))
    
    metrics = ['precision', 'recall', 'f1', 'sr_ratio', 'avg_time_ms', 'avg_psnr']
    labels = {
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score',
        'sr_ratio': 'SR Usage',
        'avg_time_ms': 'Avg Time (ms)',
        'avg_psnr': 'Avg PSNR (dB)'
    }
    
    for metric in metrics:
        label = labels.get(metric, metric)
        row = f"{label:<20}"
        
        for arch in archs:
            val = results.get(arch, {}).get(metric, 0)
            if metric == 'sr_ratio':
                row += f" {val*100:.1f}%{'':<10}"
            elif metric == 'avg_time_ms':
                row += f" {val:.1f}{'':<11}"
            else:
                row += f" {val:.4f}{'':<9}"
        
        print(row)
    
    print("=" * (20 + 16 * len(archs)))
    
    # 최고 성능 표시
    print("\n🏆 Best Performance:")
    
    best_f1 = max(results.items(), key=lambda x: x[1].get('f1', 0))
    best_time = min(results.items(), key=lambda x: x[1].get('avg_time_ms', float('inf')))
    best_efficiency = max(results.items(), key=lambda x: x[1].get('f1', 0) / max(x[1].get('avg_time_ms', 1), 1))
    
    print(f"  - Best F1: {best_f1[0].upper()} ({best_f1[1]['f1']:.4f})")
    print(f"  - Fastest: {best_time[0].upper()} ({best_time[1]['avg_time_ms']:.1f} ms)")
    print(f"  - Best Efficiency (F1/time): {best_efficiency[0].upper()}")


def main():
    parser = argparse.ArgumentParser(description='Architecture Comparison')
    
    # Models
    parser.add_argument('--sr_type', type=str, default='mamba',
                        choices=['rfdn', 'mamba'], help='SR model type')
    parser.add_argument('--sr_weights', type=str, default=None,
                        help='SR model weights path')
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.pt',
                        help='YOLO model weights path')
    parser.add_argument('--gate_weights', type=str, default=None,
                        help='Gate model weights (Arch2)')
    parser.add_argument('--arch5b_checkpoint', type=str, default=None,
                        help='Arch5B checkpoint path (for including Arch5B in comparison)')
    
    # Data
    parser.add_argument('--hr_root', type=str, required=True,
                        help='HR dataset root')
    parser.add_argument('--lr_root', type=str, required=True,
                        help='LR dataset root')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'], help='Dataset split')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Max images to process (None for all)')
    
    # Output
    parser.add_argument('--output', type=str, default='./comparison_results',
                        help='Output directory')
    
    # Settings
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                        help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    # 출력 디렉토리
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # Create engines
    # ==========================================================================
    print("\n[Initializing] Creating inference engines...")
    
    common_kwargs = {
        'sr_type': args.sr_type,
        'sr_weights': args.sr_weights,
        'yolo_weights': args.yolo_weights,
        'device': args.device,
        'conf_threshold': args.conf_threshold
    }
    
    engines = {
        'arch0': Arch0Inference(**common_kwargs),
        'arch2': Arch2Inference(**common_kwargs, gate_weights=args.gate_weights),
        'arch4': Arch4Inference(**common_kwargs)
    }
    
    # Arch5B 추가 (checkpoint 있는 경우만)
    if args.arch5b_checkpoint:
        engines['arch5b'] = Arch5BInference(
            checkpoint_path=args.arch5b_checkpoint,
            device=args.device,
            conf_threshold=args.conf_threshold
        )
        print(f"  + Arch5B added to comparison")
    
    # ==========================================================================
    # Run comparison
    # ==========================================================================
    results = run_comparison(
        engines,
        args.hr_root,
        args.lr_root,
        args.split,
        args.max_images
    )
    
    # 결과 출력
    print_comparison(results)
    
    # JSON 저장
    json_path = output_dir / 'comparison_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {json_path}")
    
    print("\n✓ Comparison completed!")


if __name__ == '__main__':
    main()