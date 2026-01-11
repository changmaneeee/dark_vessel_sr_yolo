#!/usr/bin/env python
"""
=============================================================================
generate_gate_labels_gt.py - GT 기반 Gate 라벨 생성 (옵션 A)
=============================================================================
GT Label을 참조하여 "SR 필요 여부" 라벨 생성

[로직]
1. GT에 선박 있음 (label 파일에 객체 존재):
   - Student(YOLOv8n)가 LR에서 검출 실패 → label=1 (SR needed)
   - Student가 검출 성공 → label=0 (Bypass OK)
2. GT에 선박 없음 (Empty 이미지):
   - label=0 (SR 불필요, 검출할 게 없음)

[출력]
- gate_labels_gt_{split}.json: {image_name: label, ...}
- gate_labels_gt_{split}.csv: 상세 정보
- gate_stats_gt_{split}.json: 통계

사용법:
    python generate_gate_labels_gt.py \
        --lr_root /path/to/lr_dataset \
        --label_root /path/to/hr_dataset \
        --yolo_weights /path/to/yolo_ship.pt \
        --output ./gate_labels_gt \
        --conf_threshold 0.5 \
        --iou_threshold 0.5 \
        --split both
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import torch
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO


def load_gt_labels(label_path: Path) -> List[List[float]]:
    """
    YOLO format GT label 로드
    
    Returns:
        List of [class, x_center, y_center, width, height]
    """
    if not label_path.exists():
        return []
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append([float(x) for x in parts[:5]])
    
    return labels


def get_image_files(lr_root: Path, split: str = 'train') -> List[Path]:
    """이미지 파일 목록 반환"""
    img_dir = lr_root / 'images' / split
    
    if not img_dir.exists():
        raise ValueError(f"Directory not found: {img_dir}")
    
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        files.extend(img_dir.glob(ext))
    
    return sorted(files)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    IoU 계산 (xyxy format)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def xywh_to_xyxy(box: List[float], img_w: int, img_h: int) -> np.ndarray:
    """
    Normalized xywh → xyxy (pixel coords)
    """
    x_center, y_center, w, h = box[1], box[2], box[3], box[4]
    
    x1 = (x_center - w / 2) * img_w
    y1 = (y_center - h / 2) * img_h
    x2 = (x_center + w / 2) * img_w
    y2 = (y_center + h / 2) * img_h
    
    return np.array([x1, y1, x2, y2])


def check_detection_success(
    pred_boxes: np.ndarray,
    pred_confs: np.ndarray,
    gt_boxes: List[np.ndarray],
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5
) -> Tuple[bool, float, int]:
    """
    검출 성공 여부 판단
    
    Args:
        pred_boxes: 예측 박스 [N, 4] (xyxy)
        pred_confs: 예측 confidence [N]
        gt_boxes: GT 박스 리스트 (xyxy)
        conf_threshold: confidence 임계값
        iou_threshold: IoU 임계값
    
    Returns:
        success: 검출 성공 여부 (하나라도 매칭되면 True)
        max_conf: 최대 confidence
        num_matched: 매칭된 GT 수
    """
    if len(pred_boxes) == 0:
        return False, 0.0, 0
    
    # Confidence threshold 적용
    valid_mask = pred_confs >= conf_threshold
    pred_boxes = pred_boxes[valid_mask]
    pred_confs = pred_confs[valid_mask]
    
    if len(pred_boxes) == 0:
        return False, 0.0, 0
    
    max_conf = float(pred_confs.max())
    num_matched = 0
    
    # 각 GT에 대해 매칭 확인
    for gt_box in gt_boxes:
        best_iou = 0
        for pred_box in pred_boxes:
            iou = compute_iou(pred_box, gt_box)
            best_iou = max(best_iou, iou)
        
        if best_iou >= iou_threshold:
            num_matched += 1
    
    # 하나라도 매칭되면 성공
    success = num_matched > 0
    
    return success, max_conf, num_matched


def generate_labels_gt_based(
    yolo_model: YOLO,
    image_files: List[Path],
    label_root: Path,
    split: str,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    yolo_conf: float = 0.1,
    device: str = 'cuda'
) -> Tuple[Dict[str, int], List[Dict]]:
    """
    GT 기반 Gate 라벨 생성
    
    Args:
        yolo_model: Student YOLO 모델
        image_files: LR 이미지 파일 목록
        label_root: GT label 루트 경로 (HR 데이터셋)
        split: 'train' or 'val'
        conf_threshold: 검출 성공 판단 confidence 임계값
        iou_threshold: 검출 성공 판단 IoU 임계값
        yolo_conf: YOLO inference threshold (낮게 설정)
        device: 디바이스
    
    Returns:
        labels: {image_name: label}
        details: [{image_name, label, ...}, ...]
    """
    labels = {}
    details = []
    
    # 통계
    stats = {
        'total': 0,
        'has_ship': 0,
        'empty': 0,
        'sr_needed': 0,  # Has ship but detection failed
        'bypass_ok': 0,  # Has ship and detection success OR empty
        'detection_success': 0,
        'detection_fail': 0,
    }
    
    label_dir = label_root / 'labels' / split
    
    for img_path in tqdm(image_files, desc=f"Generating GT-based labels ({split})"):
        img_name = img_path.stem
        stats['total'] += 1
        
        # 1. GT 라벨 로드
        label_path = label_dir / f"{img_name}.txt"
        gt_labels = load_gt_labels(label_path)
        
        # 2. Empty 이미지 처리
        if len(gt_labels) == 0:
            stats['empty'] += 1
            stats['bypass_ok'] += 1
            labels[img_name] = 0  # Bypass (검출할 게 없음)
            details.append({
                'image_name': img_name,
                'label': 0,
                'reason': 'empty',
                'num_gt': 0,
                'num_pred': 0,
                'max_conf': 0.0,
                'num_matched': 0
            })
            continue
        
        # 3. 선박 있는 이미지
        stats['has_ship'] += 1
        
        # 이미지 로드
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[Warning] Failed to load: {img_path}")
            continue
        
        img_h, img_w = image.shape[:2]
        
        # GT boxes를 xyxy로 변환
        gt_boxes = [xywh_to_xyxy(gt, img_w, img_h) for gt in gt_labels]
        
        # 4. Student YOLO inference
        results = yolo_model(
            image,
            conf=yolo_conf,  # 낮은 threshold로 약한 검출도 포착
            verbose=False
        )
        
        result = results[0] if results else None
        
        # 예측 결과 추출
        if result is not None and hasattr(result, 'boxes') and len(result.boxes) > 0:
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_confs = result.boxes.conf.cpu().numpy()
        else:
            pred_boxes = np.zeros((0, 4))
            pred_confs = np.zeros(0)
        
        # 5. 검출 성공 여부 판단
        success, max_conf, num_matched = check_detection_success(
            pred_boxes, pred_confs, gt_boxes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # 6. 라벨 결정
        if success:
            # Student가 검출 성공 → Bypass OK
            label = 0
            stats['bypass_ok'] += 1
            stats['detection_success'] += 1
            reason = 'detection_success'
        else:
            # Student가 검출 실패 → SR needed
            label = 1
            stats['sr_needed'] += 1
            stats['detection_fail'] += 1
            reason = 'detection_fail'
        
        labels[img_name] = label
        details.append({
            'image_name': img_name,
            'label': label,
            'reason': reason,
            'num_gt': len(gt_labels),
            'num_pred': len(pred_boxes),
            'max_conf': round(max_conf, 4),
            'num_matched': num_matched
        })
    
    # 통계 출력
    print(f"\n{'='*60}")
    print(f"📊 GT-based Label Generation Statistics")
    print(f"{'='*60}")
    print(f"  Total images:        {stats['total']}")
    print(f"  ├─ Has ship:         {stats['has_ship']} ({stats['has_ship']/stats['total']*100:.1f}%)")
    print(f"  │   ├─ Det success:  {stats['detection_success']} → Bypass")
    print(f"  │   └─ Det fail:     {stats['detection_fail']} → SR needed")
    print(f"  └─ Empty:            {stats['empty']} ({stats['empty']/stats['total']*100:.1f}%) → Bypass")
    print(f"")
    print(f"  Final Labels:")
    print(f"  ├─ SR needed (1):    {stats['sr_needed']} ({stats['sr_needed']/stats['total']*100:.1f}%)")
    print(f"  └─ Bypass OK (0):    {stats['bypass_ok']} ({stats['bypass_ok']/stats['total']*100:.1f}%)")
    print(f"{'='*60}")
    
    return labels, details, stats


def save_labels(
    labels: Dict[str, int],
    details: List[Dict],
    stats: Dict,
    output_dir: Path,
    split: str
):
    """라벨 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON 저장 (간단한 형태)
    json_path = output_dir / f'gate_labels_gt_{split}.json'
    with open(json_path, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"[Saved] {json_path}")
    
    # CSV 저장 (상세 정보)
    csv_path = output_dir / f'gate_labels_gt_{split}.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['image_name', 'label', 'reason', 'num_gt', 'num_pred', 'max_conf', 'num_matched']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)
    print(f"[Saved] {csv_path}")
    
    # 통계 저장
    stats_path = output_dir / f'gate_stats_gt_{split}.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[Saved] {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate GT-based Gate Labels')
    
    # Paths
    parser.add_argument('--lr_root', type=str, required=True,
                        help='LR 데이터셋 루트 경로')
    parser.add_argument('--label_root', type=str, required=True,
                        help='GT Label 루트 경로 (HR 데이터셋)')
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.pt',
                        help='Student YOLO 가중치 경로')
    parser.add_argument('--output', type=str, default='./gate_labels_gt',
                        help='출력 디렉토리')
    
    # Thresholds
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='검출 성공 판단 confidence 임계값')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='검출 성공 판단 IoU 임계값')
    parser.add_argument('--yolo_conf', type=float, default=0.1,
                        help='YOLO inference threshold (낮게 설정)')
    
    # Split
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'both'],
                        help='처리할 split')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        help='디바이스')
    
    args = parser.parse_args()
    
    # 경로 확인
    lr_root = Path(args.lr_root)
    label_root = Path(args.label_root)
    
    if not lr_root.exists():
        print(f"[Error] LR root not found: {lr_root}")
        return
    
    if not label_root.exists():
        print(f"[Error] Label root not found: {label_root}")
        return
    
    output_dir = Path(args.output)
    
    # YOLO 로드
    print(f"\n[Loading] Student YOLO: {args.yolo_weights}")
    yolo_model = YOLO(args.yolo_weights)
    
    print(f"[Settings]")
    print(f"  Conf threshold (detection success): {args.conf_threshold}")
    print(f"  IoU threshold (detection success): {args.iou_threshold}")
    print(f"  YOLO inference conf: {args.yolo_conf}")
    print(f"  Device: {args.device}")
    
    # Split 처리
    splits = ['train', 'val'] if args.split == 'both' else [args.split]
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"📁 Processing: {split}")
        print(f"{'='*60}")
        
        try:
            image_files = get_image_files(lr_root, split)
            print(f"[Found] {len(image_files)} images in {split}")
        except ValueError as e:
            print(f"[Error] {e}")
            continue
        
        if len(image_files) == 0:
            print(f"[Warning] No images found in {split}")
            continue
        
        # 라벨 생성
        start_time = time.time()
        labels, details, stats = generate_labels_gt_based(
            yolo_model,
            image_files,
            label_root,
            split,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            yolo_conf=args.yolo_conf,
            device=args.device
        )
        elapsed = time.time() - start_time
        
        print(f"[Time] {elapsed:.1f}s ({len(image_files)/elapsed:.1f} img/s)")
        
        # 저장
        save_labels(labels, details, stats, output_dir, split)
    
    print(f"\n✓ GT-based label generation completed!")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()