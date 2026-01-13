#!/usr/bin/env python
"""
test_wrapper_validation.py
위치: /home/changmin/dark_vessel_sr_yolo/inference/testing_code/
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

mamba_mock = MagicMock()
sys.modules["mamba_ssm"] = mamba_mock
sys.modules["einops"] = einops_mock = MagicMock()
# 2. 하위 모듈(ops) Mock 생성 및 연결
mamba_ops_mock = MagicMock()
sys.modules["mamba_ssm.ops"] = mamba_ops_mock
mamba_mock.ops = mamba_ops_mock

# 3. 더 깊은 하위 모듈(selective_scan_interface)도 있을 경우 대비
mamba_ssi_mock = MagicMock()
sys.modules["mamba_ssm.ops.selective_scan_interface"] = mamba_ssi_mock
mamba_ops_mock.selective_scan_interface = mamba_ssi_mock

# 4. (선택) modules 파일도 자주 호출되므로 추가
sys.modules["mamba_ssm.modules"] = MagicMock()
# =============================================================================
# [중요] 프로젝트 루트 경로 추가 (src 모듈을 찾기 위함)
# =============================================================================
# 현재 파일: .../dark_vessel_sr_yolo/inference/testing_code/test_wrapper_validation.py
# 목표 루트: .../dark_vessel_sr_yolo/
file_path = Path(__file__).resolve()
project_root = file_path.parents[2]  # 2단계 상위 폴더가 루트
sys.path.append(str(project_root))

print(f"[Info] Project Root added to path: {project_root}")

# =============================================================================
# Imports
# =============================================================================
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

# 이제 src 패키지를 정상적으로 불러올 수 있습니다.
try:
    from src.models.detectors.yolo_wrapper import YOLOWrapper
    print("[Info] Successfully imported YOLOWrapper from src.")
except ImportError as e:
    print(f"[Error] Failed to import src: {e}")
    sys.exit(1)


# =============================================================================
# Helper Functions
# =============================================================================

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
    """YOLO normalized xywh -> pixel xyxy"""
    x, y, w, h = box
    x1 = (x - w/2) * img_w
    y1 = (y - h/2) * img_h
    x2 = (x + w/2) * img_w
    y2 = (y + h/2) * img_h
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
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
    if len(precisions) == 0: return 0.0
    precisions = [0] + list(precisions) + [0]
    recalls = [0] + list(recalls) + [1]
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    recall_changes = []
    for i in range(1, len(recalls)):
        if recalls[i] != recalls[i - 1]:
            recall_changes.append(i)
    ap = 0
    for i in recall_changes:
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap

# =============================================================================
# Wrapper Evaluation Logic
# =============================================================================

def get_image_tensor(img_path, device):
    """이미지를 로드하여 YOLO 입력용 Tensor로 변환 (0~1 float)"""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    # ToTensor: 0~255 값을 0~1 float로 변환하고 (C, H, W)로 변경
    transform = T.Compose([
        T.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device) # [1, 3, H, W]
    return img_tensor, w, h

def evaluate_wrapper(wrapper, img_dir, label_dir, max_samples=None, desc="Evaluating"):
    """YOLOWrapper 평가 함수"""
    
    img_paths = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    if max_samples:
        img_paths = img_paths[:max_samples]
    
    all_detections = [] 
    total_gt = 0
    tp_count = 0
    fp_count = 0
    fn_count = 0
    
    device = wrapper.device

    for img_path in tqdm(img_paths, desc=desc):
        # 1. Load Image as Tensor
        img_tensor, img_w, img_h = get_image_tensor(img_path, device)
        
        # 2. Load GT
        label_path = label_dir / f"{img_path.stem}.txt"
        gt_boxes_norm = load_labels(label_path)
        gt_boxes = [xywh_to_xyxy(box[1:], img_w, img_h) for box in gt_boxes_norm]
        total_gt += len(gt_boxes)
        
        # 3. Wrapper Inference
        # YOLOWrapper.predict는 Tensor를 받아 List[Dict]를 반환
        # conf=0.001은 mAP 계산용
        results = wrapper.predict(img_tensor, conf=0.001, iou=0.6)
        
        # 결과 파싱
        pred_boxes = []
        pred_confs = []
        
        if len(results) > 0:
            result = results[0]
            if result['boxes'].numel() > 0:
                pred_boxes = result['boxes'].cpu().numpy() # xyxy
                pred_confs = result['scores'].cpu().numpy()
        
        # 4. Matching Logic
        gt_matched = [False] * len(gt_boxes)
        
        if len(pred_confs) > 0:
            sorted_indices = np.argsort(-pred_confs)
            
            for idx in sorted_indices:
                pred_box = pred_boxes[idx]
                conf = pred_confs[idx]
                
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_matched[gt_idx]: continue
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= 0.5 and best_gt_idx >= 0:
                    all_detections.append((conf, 1, 0))
                    gt_matched[best_gt_idx] = True
                    if conf >= 0.25: tp_count += 1
                else:
                    all_detections.append((conf, 0, 1))
                    if conf >= 0.25: fp_count += 1
        
        fn_count += sum(1 for m in gt_matched if not m)
    
    # Calculate Metrics
    if len(all_detections) == 0:
        return {'mAP@0.5': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': total_gt, 'total_gt': total_gt}
    
    all_detections.sort(key=lambda x: -x[0])
    
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
    
    ap = calculate_ap_coco(precisions, recalls)
    
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

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_root', type=str, required=True, help='HR dataset root')
    parser.add_argument('--yolo_weights', type=str, required=True, help='YOLO weights path')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Device] {device}")
    
    # 1. Load Custom Wrapper
    print(f"\n[Wrapper] Initializing YOLOWrapper from {args.yolo_weights}...")
    wrapper = YOLOWrapper(
        model_path=args.yolo_weights,
        device=device,
        verbose=False
    )
    wrapper.eval()
    
    hr_root = Path(args.hr_root)
    hr_img_dir = hr_root / 'images' / 'val'
    hr_label_dir = hr_root / 'labels' / 'val'
    
    print("\n" + "=" * 70)
    print("🛠️  Testing YOLOWrapper (Tensor Input -> Dict Output)")
    print("=" * 70)
    
    metrics = evaluate_wrapper(wrapper, hr_img_dir, hr_label_dir, args.max_samples, "Testing Wrapper")
    
    print(f"\n[Wrapper Results]")
    print(f"  mAP@0.5:    {metrics['mAP@0.5']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"  TP/FP/FN:   {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
    print(f"  Total GT:   {metrics['total_gt']}")
    
    print("\n" + "=" * 70)
    print("Note: 이 결과는 Official Ultralytics 결과와 유사해야 합니다.")
    print("=" * 70)

if __name__ == '__main__':
    main()