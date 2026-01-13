#!/usr/bin/env python
"""
YOLO 단독 추론 테스트 - Baseline 비교용
LR 이미지 직접 / SR 이미지 / HR 이미지 각각 테스트
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def xywh_to_xyxy(box, img_w, img_h):
    x_center, y_center, w, h = box
    x1 = (x_center - w / 2) * img_w
    y1 = (y_center - h / 2) * img_h
    x2 = (x_center + w / 2) * img_w
    y2 = (y_center + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_gt_labels(label_path, img_w, img_h):
    boxes = []
    if label_path.exists() and label_path.stat().st_size > 0:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    xywh = [float(x) for x in parts[1:5]]
                    xyxy = xywh_to_xyxy(xywh, img_w, img_h)
                    boxes.append({'class': cls, 'box': xyxy})
    return boxes


def calculate_ap(precisions, recalls):
    """COCO-style AP (All-point interpolation) - Ultralytics와 동일"""
    if not precisions or not recalls:
        return 0
    
    # Prepend sentinel values
    precisions = [0] + list(precisions) + [0]
    recalls = [0] + list(recalls) + [1]
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Find points where recall changes
    recall_changes = []
    for i in range(1, len(recalls)):
        if recalls[i] != recalls[i - 1]:
            recall_changes.append(i)
    
    # Sum (recall[i] - recall[i-1]) * precision[i]
    ap = 0
    for i in recall_changes:
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    
    return ap


def evaluate_yolo(yolo, img_dir, label_dir, max_samples=None, desc="YOLO"):
    """YOLO 단독 추론 및 평가"""
    
    img_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    if max_samples:
        img_files = img_files[:max_samples]
    
    all_detections = []
    all_gt_count = 0
    all_tp = 0
    all_fp = 0
    all_fn = 0
    
    for img_path in tqdm(img_files, desc=desc):
        img_name = img_path.stem
        label_path = label_dir / f"{img_name}.txt"
        
        # 이미지 로드
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        
        # GT 로드
        gt_boxes = load_gt_labels(label_path, img_w, img_h)
        all_gt_count += len(gt_boxes)
        gt_matched = [False] * len(gt_boxes)
        
        # YOLO 추론
        results = yolo(img, verbose=False)
        boxes = results[0].boxes
        
        preds = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                preds.append({'box': [x1, y1, x2, y2], 'conf': conf})
        
        preds = sorted(preds, key=lambda x: x['conf'], reverse=True)
        
        # 매칭
        for pred in preds:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                iou = calculate_iou(pred['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= 0.5 and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                all_detections.append({'conf': pred['conf'], 'tp': 1})
                all_tp += 1
            else:
                all_detections.append({'conf': pred['conf'], 'tp': 0})
                all_fp += 1
        
        all_fn += sum(1 for m in gt_matched if not m)
    
    # PR curve
    all_detections = sorted(all_detections, key=lambda x: x['conf'], reverse=True)
    precisions, recalls = [], []
    tp_cum, fp_cum = 0, 0
    
    for det in all_detections:
        if det['tp']:
            tp_cum += 1
        else:
            fp_cum += 1
        prec = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) > 0 else 0
        rec = tp_cum / all_gt_count if all_gt_count > 0 else 0
        precisions.append(prec)
        recalls.append(rec)
    
    ap = calculate_ap(precisions, recalls) if precisions else 0
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'mAP@0.5': ap,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': all_tp,
        'fp': all_fp,
        'fn': all_fn,
        'total_gt': all_gt_count,
        'total_pred': all_tp + all_fp
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_root', type=str, required=True)
    parser.add_argument('--hr_root', type=str, default=None)
    parser.add_argument('--yolo_weights', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Device] {device}")
    
    # YOLO 로드
    print(f"\nLoading YOLO from {args.yolo_weights}...")
    yolo = YOLO(args.yolo_weights)
    yolo.to(device)
    
    lr_root = Path(args.lr_root)
    hr_root = Path(args.hr_root) if args.hr_root else None
    
    print("\n" + "=" * 70)
    print("🔵 Test 1: YOLO on LR images (192x192) with LR labels")
    print("=" * 70)
    
    lr_img_dir = lr_root / 'images' / 'val'
    lr_label_dir = lr_root / 'labels' / 'val'
    
    lr_metrics = evaluate_yolo(yolo, lr_img_dir, lr_label_dir, args.max_samples, "YOLO on LR")
    
    print(f"\n[YOLO on LR]")
    print(f"  mAP@0.5:    {lr_metrics['mAP@0.5']:.4f}")
    print(f"  Precision:  {lr_metrics['precision']:.4f}")
    print(f"  Recall:     {lr_metrics['recall']:.4f}")
    print(f"  F1 Score:   {lr_metrics['f1']:.4f}")
    print(f"  TP/FP/FN:   {lr_metrics['tp']}/{lr_metrics['fp']}/{lr_metrics['fn']}")
    print(f"  Total GT:   {lr_metrics['total_gt']}")
    
    if hr_root:
        print("\n" + "=" * 70)
        print("🟢 Test 2: YOLO on HR images (768x768) with HR labels")
        print("=" * 70)
        
        hr_img_dir = hr_root / 'images' / 'val'
        hr_label_dir = hr_root / 'labels' / 'val'
        
        if hr_label_dir.exists():
            hr_metrics = evaluate_yolo(yolo, hr_img_dir, hr_label_dir, args.max_samples, "YOLO on HR")
            
            print(f"\n[YOLO on HR (HR labels)]")
            print(f"  mAP@0.5:    {hr_metrics['mAP@0.5']:.4f}")
            print(f"  Precision:  {hr_metrics['precision']:.4f}")
            print(f"  Recall:     {hr_metrics['recall']:.4f}")
            print(f"  F1 Score:   {hr_metrics['f1']:.4f}")
            print(f"  TP/FP/FN:   {hr_metrics['tp']}/{hr_metrics['fp']}/{hr_metrics['fn']}")
            print(f"  Total GT:   {hr_metrics['total_gt']}")
            
            # HR 이미지 + LR 라벨로도 테스트 (라벨 불일치 확인)
            print("\n" + "=" * 70)
            print("🟡 Test 3: YOLO on HR images with LR labels (mismatch check)")
            print("=" * 70)
            
            hr_lr_metrics = evaluate_yolo(yolo, hr_img_dir, lr_label_dir, args.max_samples, "YOLO on HR (LR labels)")
            
            print(f"\n[YOLO on HR (LR labels)]")
            print(f"  mAP@0.5:    {hr_lr_metrics['mAP@0.5']:.4f}")
            print(f"  Precision:  {hr_lr_metrics['precision']:.4f}")
            print(f"  Recall:     {hr_lr_metrics['recall']:.4f}")
            print(f"  F1 Score:   {hr_lr_metrics['f1']:.4f}")
            print(f"  TP/FP/FN:   {hr_lr_metrics['tp']}/{hr_lr_metrics['fp']}/{hr_lr_metrics['fn']}")
            
            # 비교 테이블
            print("\n" + "=" * 70)
            print("📊 Comparison Summary")
            print("=" * 70)
            print(f"\n{'Test':<35} {'mAP@0.5':<12} {'Precision':<12} {'Recall':<12} {'GT Count':<10}")
            print("-" * 80)
            print(f"{'YOLO on LR (LR labels)':<35} {lr_metrics['mAP@0.5']:.4f}{'':<6} {lr_metrics['precision']:.4f}{'':<6} {lr_metrics['recall']:.4f}{'':<6} {lr_metrics['total_gt']}")
            print(f"{'YOLO on HR (HR labels)':<35} {hr_metrics['mAP@0.5']:.4f}{'':<6} {hr_metrics['precision']:.4f}{'':<6} {hr_metrics['recall']:.4f}{'':<6} {hr_metrics['total_gt']}")
            print(f"{'YOLO on HR (LR labels)':<35} {hr_lr_metrics['mAP@0.5']:.4f}{'':<6} {hr_lr_metrics['precision']:.4f}{'':<6} {hr_lr_metrics['recall']:.4f}{'':<6} {lr_metrics['total_gt']}")
        else:
            print(f"  HR labels not found at {hr_label_dir}")
    
    # Ultralytics 내장 val 비교 안내
    print("\n" + "=" * 70)
    print("📋 Ultralytics 내장 val 명령어 (비교용)")
    print("=" * 70)
    print(f"""
# LR 데이터셋으로 공식 val
yolo val model={args.yolo_weights} data=/home/changmin/smart_airbus_data_lr/data.yaml split=val

# HR 데이터셋으로 공식 val (있다면)
yolo val model={args.yolo_weights} data=/home/changmin/smart_airbus_data/data.yaml split=val
""")
    
    print("\nDone!")


if __name__ == '__main__':
    main()