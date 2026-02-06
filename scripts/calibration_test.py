import torch
import os
import sys
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.ops import box_iou

# =========================================================
# [설정] 경로 확인
# =========================================================
VAL_IMG_DIR = "/home/changmin/smart_airbus_data_lr/images/val"
VAL_LABEL_DIR = "/home/changmin/smart_airbus_data_lr/labels/val"
MODEL_PATH = "/home/changmin/yolov8s+airbus_smartdata/weights/best.pt"
DATA_YAML = "/home/changmin/smart_airbus_data_lr/data.yaml"

# =========================================================
# 1. 1:1 Matching Logic (Greedy)
# =========================================================
def compute_tp_fp_fn(preds, targets, iou_thresh=0.5):
    """
    특정 Threshold에서 TP, FP, FN 계산
    preds: [N, 6] (x1, y1, x2, y2, conf, cls) - 이미 특정 conf 이상만 들어옴
    """
    if len(preds) == 0:
        return 0, 0, len(targets)
    if len(targets) == 0:
        return 0, len(preds), 0

    ious = box_iou(preds[:, :4], targets)
    
    # Conf 순 정렬 (이미 되어있을 수 있지만 안전하게)
    sorted_indices = torch.argsort(preds[:, 4], descending=True)
    preds_sorted = preds[sorted_indices]
    ious_sorted = ious[sorted_indices]
    
    tp = 0
    fp = 0
    target_matched = torch.zeros(len(targets), dtype=torch.bool)
    
    for i in range(len(preds_sorted)):
        iou_vals = ious_sorted[i]
        best_iou, best_gt_idx = iou_vals.max(dim=0)
        
        if best_iou > iou_thresh:
            if not target_matched[best_gt_idx]:
                tp += 1
                target_matched[best_gt_idx] = True
            else:
                fp += 1 # 중복
        else:
            fp += 1 # 매칭 실패
            
    fn = len(targets) - tp
    return tp, fp, fn

# =========================================================
# 2. Threshold Sweep Logic (핵심!)
# =========================================================
def find_best_threshold_and_score(all_preds_dict, all_targets_dict):
    """
    0.001 ~ 0.95 구간을 훑으며 F1이 가장 높은 지점을 찾음
    """
    print("\n🔄 Sweeping Thresholds to find Best F1...")
    
    thresholds = np.arange(0.0, 1.0, 0.05) # 0.05 단위로 검사
    thresholds[0] = 0.001 # 0.0 대신 0.001 시작
    
    best_f1 = 0.0
    best_conf = 0.0
    best_metrics = (0, 0, 0) # P, R, F1
    
    # 진행 상황 바
    for conf in tqdm(thresholds, desc="Searching Best Conf"):
        total_tp, total_fp, total_fn = 0, 0, 0
        
        # 전체 이미지에 대해 채점
        for img_id in all_preds_dict.keys():
            # 현재 Conf보다 높은 박스만 필터링
            preds = all_preds_dict[img_id]
            targets = all_targets_dict[img_id]
            
            if len(preds) > 0:
                preds_filtered = preds[preds[:, 4] >= conf]
            else:
                preds_filtered = preds
            
            tp, fp, fn = compute_tp_fp_fn(preds_filtered, targets)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
        p = total_tp / (total_tp + total_fp + 1e-6)
        r = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        
        if f1 > best_f1:
            best_f1 = f1
            best_conf = conf
            best_metrics = (p, r, f1)
            
    return best_conf, best_metrics

# =========================================================
# 메인 실행
# =========================================================
def run_calibration_v3():
    print("="*60)
    print("⚖️  FINAL LOGIC CALIBRATION (Best F1 Sweep)")
    print("="*60)
    
    # 1. Official Result
    print("[1] Official YOLO Val...")
    model = YOLO(MODEL_PATH)
    metrics = model.val(data=DATA_YAML, split='val', imgsz=192, batch=32, 
                        conf=0.001, iou=0.6, verbose=False)
    
    off_p = metrics.box.mp
    off_r = metrics.box.mr
    # Ultralytics는 내부적으로 P-R Curve에서 최고의 F1을 뽑아줍니다.
    # 정확한 F1 계산
    off_f1 = 2 * off_p * off_r / (off_p + off_r + 1e-6)
    
    print(f"  -> Official Best P: {off_p:.4f}")
    print(f"  -> Official Best R: {off_r:.4f}")
    print(f"  -> Official Best F1: {off_f1:.4f}")
    
    # 2. My Logic (Pre-calculate all boxes)
    print("\n[2] Running Custom Inference (Collecting Boxes)...")
    img_files = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.endswith(('.jpg', '.png'))])
    
    # 속도를 위해 결과를 메모리에 저장
    all_preds = {}
    all_targets = {}
    
    for img_file in tqdm(img_files):
        img_path = os.path.join(VAL_IMG_DIR, img_file)
        label_path = os.path.join(VAL_LABEL_DIR, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # Inference (아주 낮은 threshold로 모든 가능성 수집)
        results = model.predict(img_path, conf=0.001, iou=0.6, verbose=False)
        all_preds[img_file] = results[0].boxes.data.cpu()
        
        # Load GT
        if not os.path.exists(label_path):
            all_targets[img_file] = torch.empty((0, 4))
        else:
            h, w = results[0].orig_shape
            boxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    xc, yc, bw, bh = parts[1], parts[2], parts[3], parts[4]
                    x1 = (xc - bw/2) * w
                    y1 = (yc - bh/2) * h
                    x2 = (xc + bw/2) * w
                    y2 = (yc + bh/2) * h
                    boxes.append([x1, y1, x2, y2])
            all_targets[img_file] = torch.tensor(boxes)
            
    # 3. Find Best F1
    best_conf, (my_p, my_r, my_f1) = find_best_threshold_and_score(all_preds, all_targets)
    
    print(f"\n[Result] Custom Logic Best Score found at Conf={best_conf:.3f}")
    print(f"  -> My Best P:  {my_p:.4f}")
    print(f"  -> My Best R:  {my_r:.4f}")
    print(f"  -> My Best F1: {my_f1:.4f}")
    
    # 4. Final Comparison
    print("\n" + "-"*60)
    print(f"{'Metric':<10} | {'Official':<10} | {'My Logic':<10} | {'Diff':<10}")
    print("-" * 60)
    print(f"{'F1-Score':<10} | {off_f1:<10.4f} | {my_f1:<10.4f} | {abs(off_f1 - my_f1):.4f}")
    print("-" * 60)
    
    if abs(off_f1 - my_f1) < 0.03:
        print("✅ SUCCESS: 점수가 완벽하게 일치합니다!")
        print("   Arch4 평가 시에도 이 로직(find_best_threshold)을 사용하세요.")
    else:
        print("⚠️ Note: 약간의 차이는 IoU 계산 방식(C++ vs Python) 차이일 수 있습니다.")

if __name__ == "__main__":
    run_calibration_v3()