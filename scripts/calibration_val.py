"""
calibration_test.py
====================
grid_search_official.py의 평가 로직 신뢰도 검증

순수 YOLO 추론으로 비교:
  1. Official: model.val() → Ultralytics 공식 mAP50/P/R/F1
  2. Ours: model.predict() → match_predictions() → ap_per_class()
  
두 수치가 일치하면 grid_search_official.py의 Arch4 평가도 신뢰 가능.

Usage:
    python calibration_test.py
"""

import torch
import os
import sys
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class
from torchvision.ops import box_iou

# =============================================================================
# [설정] — 본인 환경에 맞게 확인
# =============================================================================
VAL_IMG_DIR = "/home/changmin/smart_airbus_data_lr/images/val"
VAL_LABEL_DIR = "/home/changmin/smart_airbus_data_lr/labels/val"
MODEL_PATH = "/home/changmin/yolov8s+airbus_smartdata/weights/best.pt"
DATA_YAML = "/home/changmin/smart_airbus_data_lr/data.yaml"

NMS_IOU = 0.5       # grid_search_official.py의 merge_iou와 동일
CONF_THRESH = 0.001  # 가능한 모든 detection 수집
IOUV = torch.linspace(0.5, 0.95, 10)


# =============================================================================
# match_predictions — grid_search_official.py와 100% 동일한 코드
# =============================================================================
def match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls, iouv):
    n_pred = len(pred_boxes)
    n_iou = len(iouv)
    correct = torch.zeros(n_pred, n_iou, dtype=torch.bool)

    if n_pred == 0 or len(gt_boxes) == 0:
        return correct

    iou = box_iou(gt_boxes, pred_boxes)
    correct_class = gt_cls[:, None] == pred_cls[None, :]

    for i, threshold in enumerate(iouv):
        valid = (iou >= threshold) & correct_class

        if not valid.any():
            continue

        gt_indices, pred_indices = torch.where(valid)
        iou_values = iou[gt_indices, pred_indices]

        sorted_idx = iou_values.argsort(descending=True)
        gt_indices = gt_indices[sorted_idx]
        pred_indices = pred_indices[sorted_idx]

        gt_used = set()
        pred_used = set()

        for gi, pi in zip(gt_indices.tolist(), pred_indices.tolist()):
            if gi not in gt_used and pi not in pred_used:
                correct[pi, i] = True
                gt_used.add(gi)
                pred_used.add(pi)

    return correct


# =============================================================================
# Main
# =============================================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("🔬 Calibration Test: Our Logic vs Ultralytics Official")
    print(f"   NMS IoU: {NMS_IOU} | Conf: {CONF_THRESH}")
    print(f"   Device: {device}")
    print("=" * 70)

    model = YOLO(MODEL_PATH)

    # ─── 1. Official val() ───
    print("\n[1] Ultralytics Official val()...")
    metrics = model.val(
        data=DATA_YAML, split='val', imgsz=192, batch=32,
        conf=CONF_THRESH, iou=NMS_IOU, verbose=False
    )

    off_map50 = metrics.box.map50
    off_map50_95 = metrics.box.map
    off_p = metrics.box.mp
    off_r = metrics.box.mr
    off_f1 = 2 * off_p * off_r / (off_p + off_r + 1e-16)

    print(f"  Official mAP50:    {off_map50:.4f}")
    print(f"  Official mAP50-95: {off_map50_95:.4f}")
    print(f"  Official P:        {off_p:.4f}")
    print(f"  Official R:        {off_r:.4f}")
    print(f"  Official F1:       {off_f1:.4f}")

    # ─── 2. Our Logic ───
    print(f"\n[2] Our Logic (predict → match_predictions → ap_per_class)...")
    img_files = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.endswith(('.jpg', '.png'))])
    print(f"  Images: {len(img_files)}")

    iouv = IOUV.to(device)

    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_target_cls = []

    for img_file in tqdm(img_files, desc="Inference"):
        img_path = os.path.join(VAL_IMG_DIR, img_file)
        label_path = os.path.join(
            VAL_LABEL_DIR,
            img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        )

        # ── Inference ──
        results = model.predict(img_path, conf=CONF_THRESH, iou=NMS_IOU, verbose=False)
        det = results[0].boxes.data.cpu()  # [N, 6] = x1,y1,x2,y2,conf,cls

        if len(det) > 0:
            pred_boxes = det[:, :4].to(device)
            pred_scores = det[:, 4].to(device)
            pred_cls = det[:, 5].long().to(device)
        else:
            pred_boxes = torch.empty((0, 4), device=device)
            pred_scores = torch.empty((0,), device=device)
            pred_cls = torch.empty((0,), dtype=torch.long, device=device)

        # ── GT ──
        h, w = results[0].orig_shape
        gt_boxes_list = []
        gt_cls_list = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    cls = int(parts[0])
                    xc, yc, bw, bh = parts[1], parts[2], parts[3], parts[4]
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h
                    gt_boxes_list.append([x1, y1, x2, y2])
                    gt_cls_list.append(cls)

        if gt_boxes_list:
            gt_boxes = torch.tensor(gt_boxes_list, device=device)
            gt_cls = torch.tensor(gt_cls_list, dtype=torch.long, device=device)
        else:
            gt_boxes = torch.empty((0, 4), device=device)
            gt_cls = torch.empty((0,), dtype=torch.long, device=device)

        all_target_cls.append(gt_cls.cpu())

        # ── TP Matching ──
        if len(pred_boxes) == 0:
            tp = torch.zeros((0, len(iouv)), dtype=torch.bool)
            conf_out = torch.empty((0,))
            pcls_out = torch.empty((0,), dtype=torch.long)
        else:
            tp = match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls, iouv).cpu()
            conf_out = pred_scores.cpu()
            pcls_out = pred_cls.cpu()

        all_tp.append(tp)
        all_conf.append(conf_out)
        all_pred_cls.append(pcls_out)

    # ── ap_per_class ──
    all_tp = torch.cat(all_tp, dim=0).numpy()
    all_conf = torch.cat(all_conf, dim=0).numpy()
    all_pred_cls = torch.cat(all_pred_cls, dim=0).numpy()
    all_target_cls = torch.cat(all_target_cls, dim=0).numpy()

    print(f"\n[3] Computing ap_per_class()...")
    print(f"  Total predictions: {len(all_conf)}")
    print(f"  Total GT: {len(all_target_cls)}")

    results = ap_per_class(
        all_tp, all_conf, all_pred_cls, all_target_cls,
        plot=False, names={0: 'ship'}
    )

    # Ultralytics 8.3.252 반환값 구조:
    # [0,1]: 내부 카운터, [2]: P, [3]: R, [4]: F1, [5]: AP
    p, r, f1, ap = results[2], results[3], results[4], results[5]

    if ap.ndim == 2:
        my_map50 = float(ap[:, 0].mean())
        my_map50_95 = float(ap.mean())
    else:
        my_map50 = float(ap.mean())
        my_map50_95 = float(ap.mean())

    my_p = float(p.mean()) if hasattr(p, 'mean') else float(p)
    my_r = float(r.mean()) if hasattr(r, 'mean') else float(r)
    my_f1 = float(f1.mean()) if hasattr(f1, 'mean') else float(f1)

    # ─── 3. Comparison ───
    print(f"\n{'='*70}")
    print(f"{'Metric':<18} | {'Official':<12} | {'Ours':<12} | {'Diff':<10}")
    print(f"{'-'*70}")
    print(f"{'mAP50':<18} | {off_map50:<12.4f} | {my_map50:<12.4f} | {abs(off_map50 - my_map50):.4f}")
    print(f"{'mAP50-95':<18} | {off_map50_95:<12.4f} | {my_map50_95:<12.4f} | {abs(off_map50_95 - my_map50_95):.4f}")
    print(f"{'F1':<18} | {off_f1:<12.4f} | {my_f1:<12.4f} | {abs(off_f1 - my_f1):.4f}")
    print(f"{'Precision':<18} | {off_p:<12.4f} | {my_p:<12.4f} | {abs(off_p - my_p):.4f}")
    print(f"{'Recall':<18} | {off_r:<12.4f} | {my_r:<12.4f} | {abs(off_r - my_r):.4f}")
    print(f"{'='*70}")

    # ─── 4. 판정 ───
    map50_diff = abs(off_map50 - my_map50)
    f1_diff = abs(off_f1 - my_f1)

    if map50_diff < 0.005 and f1_diff < 0.005:
        print("✅ PERFECT MATCH — grid_search_official.py 신뢰도 확인 완료!")
        print("   → 서버에서 바로 돌리면 됩니다.")
    elif map50_diff < 0.03 and f1_diff < 0.02:
        print("✅ ACCEPTABLE MATCH — 방향성 비교에 충분한 정밀도")
        print(f"   mAP50 차이: {map50_diff:.4f} (< 3%)")
        print(f"   F1 차이: {f1_diff:.4f} (< 2%)")
        print("   → Grid search 아키텍처 비교 OK")
        print("   → 논문 최종 수치는 Official val()로 재검증 권장")
    else:
        print("❌ MISMATCH — 로직 재점검 필요")
        print(f"   mAP50 차이: {map50_diff:.4f}")
        print(f"   F1 차이: {f1_diff:.4f}")
        print("   → match_predictions 구현 확인 필요")


if __name__ == "__main__":
    main()