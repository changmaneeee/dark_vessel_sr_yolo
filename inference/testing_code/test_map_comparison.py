#!/usr/bin/env python
"""
mAP 계산 방식 비교 테스트
- 우리 코드 (11-point interpolation)
- Ultralytics 내장 방식 (COCO-style AP)
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


def calculate_ap_11point(precisions, recalls):
    """11-point interpolation (Pascal VOC style)"""
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
        if prec_at_recall:
            ap += max(prec_at_recall) / 11
    return ap


def calculate_ap_coco(precisions, recalls):
    """All-point interpolation (COCO style) - Ultralytics 사용"""
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


def evaluate_with_both_ap(yolo, img_dir, label_dir, max_samples=None, imgsz=None):
    """YOLO 추론 후 두 가지 AP 계산 방식으로 비교"""
    
    img_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    if max_samples:
        img_files = img_files[:max_samples]
    
    all_detections = []
    all_gt_count = 0
    all_tp = 0
    all_fp = 0
    
    for img_path in tqdm(img_files, desc="Evaluating"):
        img_name = img_path.stem
        label_path = label_dir / f"{img_name}.txt"
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        
        gt_boxes = load_gt_labels(label_path, img_w, img_h)
        all_gt_count += len(gt_boxes)
        gt_matched = [False] * len(gt_boxes)
        
        # YOLO 추론 (imgsz 지정 가능)
        if imgsz:
            results = yolo(img, verbose=False, imgsz=imgsz)
        else:
            results = yolo(img, verbose=False)
        
        boxes = results[0].boxes
        
        preds = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                preds.append({'box': [x1, y1, x2, y2], 'conf': conf})
        
        preds = sorted(preds, key=lambda x: x['conf'], reverse=True)
        
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
    
    all_fn = all_gt_count - all_tp
    
    # Sort by confidence
    all_detections = sorted(all_detections, key=lambda x: x['conf'], reverse=True)
    
    # PR curve
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
    
    # 두 가지 AP 계산
    ap_11point = calculate_ap_11point(precisions, recalls)
    ap_coco = calculate_ap_coco(precisions, recalls)
    
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / all_gt_count if all_gt_count > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'mAP_11point': ap_11point,
        'mAP_COCO': ap_coco,
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
    parser.add_argument('--yolo_weights', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Device] {device}")
    
    print(f"\nLoading YOLO from {args.yolo_weights}...")
    yolo = YOLO(args.yolo_weights)
    yolo.to(device)
    
    lr_root = Path(args.lr_root)
    lr_img_dir = lr_root / 'images' / 'val'
    lr_label_dir = lr_root / 'labels' / 'val'
    
    # 전체 이미지 수 확인
    total_images = len(list(lr_img_dir.glob('*.jpg')))
    test_samples = args.max_samples if args.max_samples else total_images
    print(f"\nTotal val images: {total_images}")
    print(f"Testing on: {test_samples} images")
    
    print("\n" + "=" * 70)
    print("🔬 Test 1: Default imgsz")
    print("=" * 70)
    
    metrics_default = evaluate_with_both_ap(yolo, lr_img_dir, lr_label_dir, args.max_samples)
    
    print(f"\n[Results - Default imgsz]")
    print(f"  mAP@0.5 (11-point):  {metrics_default['mAP_11point']:.4f}")
    print(f"  mAP@0.5 (COCO):      {metrics_default['mAP_COCO']:.4f}")
    print(f"  Precision:           {metrics_default['precision']:.4f}")
    print(f"  Recall:              {metrics_default['recall']:.4f}")
    print(f"  F1:                  {metrics_default['f1']:.4f}")
    print(f"  TP/FP/FN:            {metrics_default['tp']}/{metrics_default['fp']}/{metrics_default['fn']}")
    print(f"  Total GT:            {metrics_default['total_gt']}")
    
    print("\n" + "=" * 70)
    print("🔬 Test 2: imgsz=640 (Ultralytics default)")
    print("=" * 70)
    
    metrics_640 = evaluate_with_both_ap(yolo, lr_img_dir, lr_label_dir, args.max_samples, imgsz=640)
    
    print(f"\n[Results - imgsz=640]")
    print(f"  mAP@0.5 (11-point):  {metrics_640['mAP_11point']:.4f}")
    print(f"  mAP@0.5 (COCO):      {metrics_640['mAP_COCO']:.4f}")
    print(f"  Precision:           {metrics_640['precision']:.4f}")
    print(f"  Recall:              {metrics_640['recall']:.4f}")
    print(f"  F1:                  {metrics_640['f1']:.4f}")
    print(f"  TP/FP/FN:            {metrics_640['tp']}/{metrics_640['fp']}/{metrics_640['fn']}")
    
    print("\n" + "=" * 70)
    print("📊 Comparison Summary")
    print("=" * 70)
    
    print(f"\n{'Setting':<20} {'mAP(11pt)':<12} {'mAP(COCO)':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 70)
    print(f"{'Default imgsz':<20} {metrics_default['mAP_11point']:.4f}{'':<6} {metrics_default['mAP_COCO']:.4f}{'':<6} {metrics_default['precision']:.4f}{'':<6} {metrics_default['recall']:.4f}")
    print(f"{'imgsz=640':<20} {metrics_640['mAP_11point']:.4f}{'':<6} {metrics_640['mAP_COCO']:.4f}{'':<6} {metrics_640['precision']:.4f}{'':<6} {metrics_640['recall']:.4f}")
    print(f"{'Training result':<20} {'N/A':<12} {'0.6576':<12} {'0.8000':<12} {'0.5700':<12}")
    
    print("\n" + "=" * 70)
    print("💡 Ultralytics val 직접 호출 (Python API)")
    print("=" * 70)
    
    # Ultralytics 내장 val 사용
    try:
        print("\nRunning Ultralytics built-in validation...")
        val_results = yolo.val(
            data=str(lr_root / 'data.yaml'),
            split='val',
            verbose=False
        )
        print(f"\n[Ultralytics Built-in Val]")
        print(f"  mAP@0.5:     {val_results.box.map50:.4f}")
        print(f"  mAP@0.5-95:  {val_results.box.map:.4f}")
        print(f"  Precision:   {val_results.box.p[0]:.4f}" if len(val_results.box.p) > 0 else "  Precision:   N/A")
        print(f"  Recall:      {val_results.box.r[0]:.4f}" if len(val_results.box.r) > 0 else "  Recall:      N/A")
    except Exception as e:
        print(f"  ⚠️ Ultralytics val failed: {e}")
        print("  Try: yolo val model=<weights> data=<data.yaml> split=val")
    
    print("\nDone!")


if __name__ == '__main__':
    main()