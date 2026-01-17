#!/usr/bin/env python
"""
Tensor 입력 vs 파일 입력: mAP 차이 측정

목표: 두 방식의 실제 mAP 차이가 얼마인지 확인
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as T
from tqdm import tqdm


def load_image_as_tensor(img_path: str, device: str = 'cuda') -> torch.Tensor:
    """이미지를 Tensor로 로드"""
    img = Image.open(img_path).convert('RGB')
    tensor = T.ToTensor()(img).unsqueeze(0).to(device)
    return tensor


def load_labels(label_path: Path, img_w: int, img_h: int):
    """GT 라벨 로드 (YOLO format → xyxy)"""
    boxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, x_c, y_c, w, h = map(float, parts[:5])
                    # xywhn → xyxy
                    x1 = (x_c - w/2) * img_w
                    y1 = (y_c - h/2) * img_h
                    x2 = (x_c + w/2) * img_w
                    y2 = (y_c + h/2) * img_h
                    boxes.append([x1, y1, x2, y2])
    return np.array(boxes) if boxes else np.array([]).reshape(0, 4)


def box_iou(box1, box2):
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


def calculate_ap(predictions, gt_boxes, iou_threshold=0.5):
    """단일 이미지의 AP 계산에 필요한 TP/FP 반환"""
    if len(predictions) == 0:
        return [], [], len(gt_boxes)  # no predictions
    
    # confidence 기준 정렬
    sorted_indices = np.argsort(-predictions[:, 4])
    predictions = predictions[sorted_indices]
    
    matched_gt = set()
    tps = []
    fps = []
    
    for pred in predictions:
        pred_box = pred[:4]
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tps.append(1)
            fps.append(0)
            matched_gt.add(best_gt_idx)
        else:
            tps.append(0)
            fps.append(1)
    
    return tps, fps, len(gt_boxes)


def compute_map(all_tps, all_fps, total_gt):
    """전체 데이터셋의 mAP 계산"""
    if total_gt == 0:
        return 0.0
    
    tps = np.array(all_tps)
    fps = np.array(all_fps)
    
    # 누적
    tp_cumsum = np.cumsum(tps)
    fp_cumsum = np.cumsum(fps)
    
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    # AP 계산 (11-point interpolation)
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max()
    ap /= 11
    
    return ap


def compare_map(
    val_images_dir: str,
    val_labels_dir: str,
    weights_path: str,
    conf: float = 0.25,
    max_images: int = None
):
    """두 방식의 mAP 비교"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"📊 Tensor vs 파일 입력: mAP 비교")
    print(f"{'='*70}")
    
    # 이미지 목록
    images_dir = Path(val_images_dir)
    labels_dir = Path(val_labels_dir)
    
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"이미지 수: {len(image_files)}")
    print(f"모델: {weights_path}")
    print(f"Confidence: {conf}")
    
    # YOLO 로드
    yolo = YOLO(weights_path, verbose=False)
    
    # 결과 저장
    file_tps, file_fps, file_gt_count = [], [], 0
    tensor_tps, tensor_fps, tensor_gt_count = [], [], 0
    
    box_diffs = []
    score_diffs = []
    
    for img_path in tqdm(image_files, desc="평가 중"):
        # 이미지 정보
        img = Image.open(img_path)
        img_w, img_h = img.size
        
        # GT 로드
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = load_labels(label_path, img_w, img_h)
        
        # =====================================================================
        # 방법 1: 파일 입력
        # =====================================================================
        file_results = yolo(str(img_path), conf=conf, verbose=False)
        
        file_preds = []
        for r in file_results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                for box, score in zip(boxes, scores):
                    file_preds.append([*box, score])
        file_preds = np.array(file_preds) if file_preds else np.array([]).reshape(0, 5)
        
        tps, fps, n_gt = calculate_ap(file_preds, gt_boxes)
        file_tps.extend(tps)
        file_fps.extend(fps)
        file_gt_count += n_gt
        
        # =====================================================================
        # 방법 2: Tensor 입력
        # =====================================================================
        img_tensor = load_image_as_tensor(str(img_path), device)
        tensor_results = yolo.predict(source=img_tensor, conf=conf, verbose=False)
        
        tensor_preds = []
        for r in tensor_results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                for box, score in zip(boxes, scores):
                    tensor_preds.append([*box, score])
        tensor_preds = np.array(tensor_preds) if tensor_preds else np.array([]).reshape(0, 5)
        
        tps, fps, n_gt = calculate_ap(tensor_preds, gt_boxes)
        tensor_tps.extend(tps)
        tensor_fps.extend(fps)
        tensor_gt_count += n_gt
        
        # =====================================================================
        # 차이 기록
        # =====================================================================
        if len(file_preds) > 0 and len(tensor_preds) > 0:
            # 가장 높은 confidence 박스 비교
            file_best = file_preds[file_preds[:, 4].argmax()]
            tensor_best = tensor_preds[tensor_preds[:, 4].argmax()]
            
            box_diff = np.abs(file_best[:4] - tensor_best[:4]).max()
            score_diff = abs(file_best[4] - tensor_best[4])
            
            box_diffs.append(box_diff)
            score_diffs.append(score_diff)
    
    # =========================================================================
    # 결과 계산
    # =========================================================================
    file_map = compute_map(file_tps, file_fps, file_gt_count)
    tensor_map = compute_map(tensor_tps, tensor_fps, tensor_gt_count)
    
    print(f"\n{'='*70}")
    print(f"[결과]")
    print(f"{'='*70}")
    
    print(f"\n📈 mAP@0.5:")
    print(f"  파일 입력:   {file_map:.4f}")
    print(f"  Tensor 입력: {tensor_map:.4f}")
    print(f"  차이:        {abs(file_map - tensor_map):.4f} ({abs(file_map - tensor_map) / file_map * 100:.2f}%)")
    
    if box_diffs:
        print(f"\n📐 Box 차이 (픽셀):")
        print(f"  평균: {np.mean(box_diffs):.2f}")
        print(f"  최대: {np.max(box_diffs):.2f}")
        print(f"  최소: {np.min(box_diffs):.2f}")
        
        print(f"\n🎯 Score 차이:")
        print(f"  평균: {np.mean(score_diffs):.4f}")
        print(f"  최대: {np.max(score_diffs):.4f}")
    
    # 결론
    print(f"\n{'='*70}")
    print(f"[결론]")
    print(f"{'='*70}")
    
    map_diff_percent = abs(file_map - tensor_map) / file_map * 100 if file_map > 0 else 0
    
    if map_diff_percent < 0.5:
        print(f"  ✅ mAP 차이 {map_diff_percent:.2f}% - 무시해도 됨")
    elif map_diff_percent < 2:
        print(f"  ⚠️ mAP 차이 {map_diff_percent:.2f}% - 주의 필요, 같은 방식으로 비교 권장")
    else:
        print(f"  ❌ mAP 차이 {map_diff_percent:.2f}% - 반드시 같은 방식 사용해야 함")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_images', type=str, 
                        default='/home/changmin/smart_airbus_data/images/val',
                        help='Validation 이미지 폴더')
    parser.add_argument('--val_labels', type=str,
                        default='/home/changmin/smart_airbus_data/labels/val',
                        help='Validation 라벨 폴더')
    parser.add_argument('--weights', type=str, 
                        default='weights/yolohr/8s/best.pt',
                        help='YOLO 모델')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--max_images', type=int, default=None,
                        help='테스트할 최대 이미지 수 (빠른 테스트용)')
    args = parser.parse_args()
    
    compare_map(
        args.val_images,
        args.val_labels,
        args.weights,
        args.conf,
        args.max_images
    )