#!/usr/bin/env python
"""
=============================================================================
verify_detection_eval.py - Detection 평가 로직 검증
=============================================================================

단일 이미지로 step-by-step 확인:
1. GT 라벨 로드 확인
2. YOLO prediction 확인
3. IoU 계산 확인
4. TP/FP/FN 계산 확인

사용법:
    cd ~/dark_vessel_sr_yolo
    
    python inference/testing_code/verify_detection_eval.py \
        --hr_image /home/changmin/smart_airbus_data/images/val/0a7a0fa38.jpg \
        --label /home/changmin/smart_airbus_data/labels/val/0a7a0fa38.txt \
        --yolo_weights weights/yolohr/8s/best.pt
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """IoU 계산 (xyxy format)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def xywhn_to_xyxy(box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """YOLO format (xywh normalized) → xyxy pixel"""
    x_c, y_c, w, h = box
    x1 = (x_c - w/2) * img_w
    y1 = (y_c - h/2) * img_h
    x2 = (x_c + w/2) * img_w
    y2 = (y_c + h/2) * img_h
    return np.array([x1, y1, x2, y2])


def load_gt_boxes(label_path: Path, img_w: int, img_h: int):
    """GT 라벨 로드"""
    boxes = []
    raw_lines = []
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                raw_lines.append(line.strip())
                parts = line.strip().split()
                if len(parts) >= 5:
                    box_norm = np.array([float(x) for x in parts[1:5]])
                    box_pixel = xywhn_to_xyxy(box_norm, img_w, img_h)
                    boxes.append({
                        'class': int(parts[0]),
                        'box_norm': box_norm,
                        'box_pixel': box_pixel
                    })
    
    return boxes, raw_lines


def visualize_boxes(img_path, gt_boxes, pred_boxes, pred_scores, save_path=None):
    """GT와 Prediction 시각화"""
    img = Image.open(img_path)
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img)
    
    # GT boxes (초록색)
    for gt in gt_boxes:
        box = gt['box_pixel']
        rect = patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        ax.text(box[0], box[1]-5, 'GT', color='green', fontsize=10, fontweight='bold')
    
    # Prediction boxes (빨간색)
    for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
        rect = patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(box[0], box[1]-5, f'P:{score:.2f}', color='red', fontsize=10, fontweight='bold')
    
    ax.set_title(f'GT: {len(gt_boxes)} (green) | Pred: {len(pred_boxes)} (red)')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✅ 시각화 저장: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_image', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--yolo_weights', type=str, required=True)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--save_viz', type=str, default=None)
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"🔍 Detection 평가 검증")
    print(f"{'='*70}")
    
    # 1. 이미지 로드
    print(f"\n[1] 이미지 로드")
    img = Image.open(args.hr_image)
    img_w, img_h = img.size
    print(f"  이미지: {args.hr_image}")
    print(f"  크기: {img_w} x {img_h}")
    
    # 2. GT 라벨 로드
    print(f"\n[2] GT 라벨 로드")
    gt_boxes, raw_lines = load_gt_boxes(Path(args.label), img_w, img_h)
    print(f"  라벨 파일: {args.label}")
    print(f"  원본 라인:")
    for line in raw_lines:
        print(f"    {line}")
    print(f"\n  파싱된 GT boxes ({len(gt_boxes)}개):")
    for i, gt in enumerate(gt_boxes):
        print(f"    [{i}] class={gt['class']}, norm={gt['box_norm']}, pixel={gt['box_pixel']}")
    
    # 3. YOLO 예측
    print(f"\n[3] YOLO 예측")
    from ultralytics import YOLO
    
    model = YOLO(args.yolo_weights)
    results = model(args.hr_image, conf=args.conf, verbose=False)
    
    pred_boxes = []
    pred_scores = []
    pred_classes = []
    
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                pred_boxes.append(box)
                pred_scores.append(float(score))
                pred_classes.append(int(cls))
    
    print(f"  YOLO weights: {args.yolo_weights}")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  예측 결과 ({len(pred_boxes)}개):")
    for i, (box, score, cls) in enumerate(zip(pred_boxes, pred_scores, pred_classes)):
        print(f"    [{i}] class={cls}, score={score:.3f}, box={box}")
    
    # 4. IoU 계산
    print(f"\n[4] IoU 계산")
    if len(gt_boxes) > 0 and len(pred_boxes) > 0:
        print(f"  IoU Matrix:")
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt in enumerate(gt_boxes):
                iou = box_iou(pred_box, gt['box_pixel'])
                iou_matrix[i, j] = iou
                print(f"    Pred[{i}] vs GT[{j}]: IoU = {iou:.3f}")
    else:
        print(f"  GT 또는 Pred가 비어있음")
    
    # 5. TP/FP/FN 계산
    print(f"\n[5] TP/FP/FN 계산 (IoU threshold = 0.5)")
    iou_threshold = 0.5
    
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        tp, fp, fn = 0, 0, 0
        print(f"  둘 다 비어있음 → TP=0, FP=0, FN=0")
    elif len(gt_boxes) == 0:
        tp, fp, fn = 0, len(pred_boxes), 0
        print(f"  GT 없음, Pred {len(pred_boxes)}개 → 모두 FP")
    elif len(pred_boxes) == 0:
        tp, fp, fn = 0, 0, len(gt_boxes)
        print(f"  Pred 없음, GT {len(gt_boxes)}개 → 모두 FN")
    else:
        # Score 기준 정렬
        sorted_indices = np.argsort(pred_scores)[::-1]
        sorted_pred_boxes = [pred_boxes[i] for i in sorted_indices]
        sorted_pred_scores = [pred_scores[i] for i in sorted_indices]
        
        gt_matched = [False] * len(gt_boxes)
        tp, fp = 0, 0
        
        print(f"\n  매칭 과정:")
        for i, (pred_box, score) in enumerate(zip(sorted_pred_boxes, sorted_pred_scores)):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                iou = box_iou(pred_box, gt['box_pixel'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                gt_matched[best_gt_idx] = True
                print(f"    Pred[{sorted_indices[i]}] (score={score:.3f}) → GT[{best_gt_idx}] (IoU={best_iou:.3f}) → TP ✅")
            else:
                fp += 1
                if best_gt_idx >= 0:
                    print(f"    Pred[{sorted_indices[i]}] (score={score:.3f}) → GT[{best_gt_idx}] (IoU={best_iou:.3f} < 0.5) → FP ❌")
                else:
                    print(f"    Pred[{sorted_indices[i]}] (score={score:.3f}) → No match → FP ❌")
        
        fn = sum(1 for m in gt_matched if not m)
        print(f"\n  미매칭 GT: {fn}개 → FN")
    
    # 6. 최종 결과
    print(f"\n[6] 최종 결과")
    print(f"  TP: {tp}")
    print(f"  FP: {fp}")
    print(f"  FN: {fn}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1: {f1:.3f}")
    
    # 7. 시각화 (선택)
    if args.save_viz or True:  # 항상 시각화
        print(f"\n[7] 시각화")
        gt_pixel_boxes = [gt['box_pixel'] for gt in gt_boxes]
        visualize_boxes(args.hr_image, gt_boxes, pred_boxes, pred_scores, args.save_viz)
    
    print(f"\n{'='*70}")
    print(f"✅ 검증 완료!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()