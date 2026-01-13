#!/usr/bin/env python
"""
전체 Validation 데이터로 YOLO 성능 측정
- Ultralytics 공식 val 결과와 비교용
"""

import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def load_labels(label_path):
    """YOLO 형식 라벨 로드"""
    boxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, x, y, w, h = map(float, parts[:5])
                    boxes.append([cls, x, y, w, h])
    return boxes


def xywh_to_xyxy(box, img_w, img_h):
    """YOLO normalized xywh → pixel xyxy"""
    x, y, w, h = box
    x1 = (x - w/2) * img_w
    y1 = (y - h/2) * img_h
    x2 = (x + w/2) * img_w
    y2 = (y + h/2) * img_h
    return [x1, y1, x2, y2]


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def calculate_ap_coco(precisions, recalls):
    """COCO-style AP (All-point interpolation)"""
    if len(precisions) == 0:
        return 0.0
    
    # Add sentinel values
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
    
    # Calculate area under curve
    ap = 0
    for i in recall_changes:
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    
    return ap


def evaluate_yolo(yolo, img_dir, label_dir, max_samples=None, desc="Evaluating"):
    """YOLO 평가 - COCO-style mAP"""
    
    img_paths = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    if max_samples:
        img_paths = img_paths[:max_samples]
    
    all_detections = []  # (confidence, is_tp, is_fp)
    total_gt = 0
    tp_count = 0
    fp_count = 0
    fn_count = 0
    
    for img_path in tqdm(img_paths, desc=desc):
        # Load image and get size
        from PIL import Image
        img = Image.open(img_path)
        img_w, img_h = img.size
        
        # Load GT
        label_path = label_dir / f"{img_path.stem}.txt"
        gt_boxes_norm = load_labels(label_path)
        gt_boxes = [xywh_to_xyxy(box[1:], img_w, img_h) for box in gt_boxes_norm]
        total_gt += len(gt_boxes)
        
        # YOLO inference
        results = yolo(img_path, verbose=False, conf=0.001)  # Low conf for AP calculation
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            pred_boxes = boxes.xyxy.cpu().numpy()
            pred_confs = boxes.conf.cpu().numpy()
        else:
            pred_boxes = []
            pred_confs = []
        
        # Match predictions to GT
        gt_matched = [False] * len(gt_boxes)
        
        # Sort by confidence
        if len(pred_confs) > 0:
            sorted_indices = np.argsort(-pred_confs)
            
            for idx in sorted_indices:
                pred_box = pred_boxes[idx]
                conf = pred_confs[idx]
                
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_matched[gt_idx]:
                        continue
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= 0.5 and best_gt_idx >= 0:
                    # True Positive
                    all_detections.append((conf, 1, 0))
                    gt_matched[best_gt_idx] = True
                    if conf >= 0.25:  # Standard threshold for counting
                        tp_count += 1
                else:
                    # False Positive
                    all_detections.append((conf, 0, 1))
                    if conf >= 0.25:
                        fp_count += 1
        
        # Count FN (unmatched GT)
        fn_count += sum(1 for m in gt_matched if not m)
    
    # Calculate mAP
    if len(all_detections) == 0:
        return {
            'mAP@0.5': 0, 'precision': 0, 'recall': 0, 'f1': 0,
            'tp': 0, 'fp': 0, 'fn': total_gt, 'total_gt': total_gt
        }
    
    # Sort by confidence
    all_detections.sort(key=lambda x: -x[0])
    
    # Calculate precision-recall curve
    precisions = []
    recalls = []
    cum_tp = 0
    cum_fp = 0
    
    for conf, is_tp, is_fp in all_detections:
        cum_tp += is_tp
        cum_fp += is_fp
        
        precision = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0
        recall = cum_tp / total_gt if total_gt > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP (COCO-style)
    ap = calculate_ap_coco(precisions, recalls)
    
    # Metrics at standard threshold
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'mAP@0.5': ap,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp_count,
        'fp': fp_count,
        'fn': fn_count,
        'total_gt': total_gt
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_root', type=str, required=True, help='HR dataset root')
    parser.add_argument('--lr_root', type=str, default=None, help='LR dataset root')
    parser.add_argument('--yolo_weights', type=str, required=True, help='YOLO weights')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples (None=all)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Device] {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
    
    # Load YOLO
    print(f"\n[Model] Loading YOLO from {args.yolo_weights}...")
    yolo = YOLO(args.yolo_weights)
    yolo.to(device)
    
    hr_root = Path(args.hr_root)
    
    # Test 1: HR images with HR labels
    print("\n" + "=" * 70)
    print("🔵 YOLO-HR on HR images (768×768) - Full Validation")
    print("=" * 70)
    
    hr_img_dir = hr_root / 'images' / 'val'
    hr_label_dir = hr_root / 'labels' / 'val'
    
    total_images = len(list(hr_img_dir.glob('*.jpg')) + list(hr_img_dir.glob('*.png')))
    test_images = args.max_samples if args.max_samples else total_images
    print(f"  Total images: {total_images}")
    print(f"  Testing: {test_images}")
    
    hr_metrics = evaluate_yolo(yolo, hr_img_dir, hr_label_dir, args.max_samples, "YOLO on HR")
    
    print(f"\n[YOLO-HR on HR]")
    print(f"  mAP@0.5:    {hr_metrics['mAP@0.5']:.4f}")
    print(f"  Precision:  {hr_metrics['precision']:.4f}")
    print(f"  Recall:     {hr_metrics['recall']:.4f}")
    print(f"  F1 Score:   {hr_metrics['f1']:.4f}")
    print(f"  TP/FP/FN:   {hr_metrics['tp']}/{hr_metrics['fp']}/{hr_metrics['fn']}")
    print(f"  Total GT:   {hr_metrics['total_gt']}")
    
    # Test 2: LR images (optional)
    if args.lr_root:
        lr_root = Path(args.lr_root)
        
        print("\n" + "=" * 70)
        print("🟢 YOLO-HR on LR images (192×192) - Full Validation")
        print("=" * 70)
        
        lr_img_dir = lr_root / 'images' / 'val'
        lr_label_dir = lr_root / 'labels' / 'val'
        
        lr_metrics = evaluate_yolo(yolo, lr_img_dir, lr_label_dir, args.max_samples, "YOLO on LR")
        
        print(f"\n[YOLO-HR on LR]")
        print(f"  mAP@0.5:    {lr_metrics['mAP@0.5']:.4f}")
        print(f"  Precision:  {lr_metrics['precision']:.4f}")
        print(f"  Recall:     {lr_metrics['recall']:.4f}")
        print(f"  F1 Score:   {lr_metrics['f1']:.4f}")
        print(f"  TP/FP/FN:   {lr_metrics['tp']}/{lr_metrics['fp']}/{lr_metrics['fn']}")
        print(f"  Total GT:   {lr_metrics['total_gt']}")
        
        # Comparison
        print("\n" + "=" * 70)
        print("📊 Comparison Summary")
        print("=" * 70)
        print(f"\n{'Test':<30} {'mAP@0.5':<12} {'Precision':<12} {'Recall':<12}")
        print("-" * 70)
        print(f"{'Ultralytics val (official)':<30} {'0.778':<12} {'0.72':<12} {'0.75':<12}")
        print(f"{'Our code: YOLO-HR on HR':<30} {hr_metrics['mAP@0.5']:<12.4f} {hr_metrics['precision']:<12.4f} {hr_metrics['recall']:<12.4f}")
        print(f"{'Our code: YOLO-HR on LR':<30} {lr_metrics['mAP@0.5']:<12.4f} {lr_metrics['precision']:<12.4f} {lr_metrics['recall']:<12.4f}")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()