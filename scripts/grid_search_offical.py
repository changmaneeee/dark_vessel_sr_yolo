"""
grid_search_official.py
========================
Arch4 Adaptive Grid Search — Ultralytics 공식 mAP50 기반

✅ 하나의 파일로 전부 해결:
  - 540개 파라미터 조합 × 2000장 샘플
  - Ultralytics ap_per_class()로 공식 mAP50/F1/P/R 계산
  - 중간 저장 (크래시 시 이어서 돌릴 수 있음)

환경: A5000 서버
예상 시간: ~1.5일 (540 × 2000장 × ~0.2s/img)

Usage:
    python grid_search_official.py
    
    # 이어서 돌리기 (중간 저장된 CSV에서 이미 완료된 조합 스킵)
    python grid_search_official.py --resume
"""

import torch
import torch.nn as nn
import cv2
import os
import sys
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from pathlib import Path
import time
import random
import argparse
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.pipelines.arch4_adaptive import Arch4Adaptive
from torchvision.ops import box_iou
from ultralytics.utils.metrics import ap_per_class

# =============================================================================
# [설정] 파라미터 그리드 (540개 조합)
# =============================================================================
PARAM_GRID = {
    'crop_size_lr':  [16, 32, 48],
    'pass1_conf':    [0.001, 0.005, 0.01],
    'pass2_conf':    [0.1, 0.3],
    'final_conf':    [0.2, 0.3, 0.4],
    'roi_expansion': [1.0, 1.5, 2.0]
}

NUM_SAMPLES = 2000
RANDOM_SEED = 42

# =============================================================================
# [고정 설정] A5000 서버 경로
# =============================================================================
CONFIG_BASE = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'yolo_weights_lr': "/home/changmin/yolov8s+airbus_smartdata/weights/best.pt",
    'yolo_weights_hr': "/home/changmin/yolov8s+HR_airbus_smartdata/weights/best.pt",
    'sr_weights': "/home/changmin/dark_vessel_sr_yolo/weights/rfdn/model_best.pt",
}

VAL_IMG_DIR = "/home/changmin/smart_airbus_data_lr/images/val/"
VAL_LABEL_DIR = "/home/changmin/smart_airbus_data_lr/labels/val/"
OUTPUT_CSV = "grid_search_official_results.csv"

# Ultralytics IoU thresholds
IOUV = torch.linspace(0.5, 0.95, 10)


# =============================================================================
# GT 캐싱 (매 조합마다 GT 다시 읽지 않도록)
# =============================================================================
class GTCache:
    """GT labels를 한 번만 읽고 캐싱"""
    def __init__(self, img_files, img_dir, label_dir):
        print("📦 Caching GT labels...")
        self.gt_data = {}
        for img_file in tqdm(img_files, desc="Loading GT"):
            label_path = os.path.join(
                label_dir,
                img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            )
            img_path = os.path.join(img_dir, img_file)
            
            # 이미지 크기 읽기 (GT 좌표 변환에 필요)
            img = cv2.imread(img_path)
            if img is None:
                continue
            H, W = img.shape[:2]
            
            boxes = []
            classes = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = list(map(float, line.strip().split()))
                        cls = int(parts[0])
                        xc, yc, w, h = parts[1], parts[2], parts[3], parts[4]
                        x1 = (xc - w / 2) * W
                        y1 = (yc - h / 2) * H
                        x2 = (xc + w / 2) * W
                        y2 = (yc + h / 2) * H
                        boxes.append([x1, y1, x2, y2])
                        classes.append(cls)
            
            self.gt_data[img_file] = {
                'boxes': torch.tensor(boxes) if boxes else torch.empty((0, 4)),
                'classes': torch.tensor(classes, dtype=torch.long) if classes else torch.empty((0,), dtype=torch.long),
                'img_hw': (H, W)
            }
        
        print(f"  ✓ Cached {len(self.gt_data)} images")
    
    def get(self, img_file, device='cpu'):
        data = self.gt_data[img_file]
        return {
            'boxes': data['boxes'].to(device),
            'classes': data['classes'].to(device),
            'img_hw': data['img_hw']
        }


# =============================================================================
# Ultralytics-compatible TP Matching
# =============================================================================
def match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls, iouv):
    """
    Greedy 1:1 matching (Ultralytics match_predictions 재현)
    
    Returns:
        correct: [N_pred, T] bool (T = len(iouv))
    """
    n_pred = len(pred_boxes)
    n_iou = len(iouv)
    correct = torch.zeros(n_pred, n_iou, dtype=torch.bool)
    
    if n_pred == 0 or len(gt_boxes) == 0:
        return correct
    
    iou = box_iou(gt_boxes, pred_boxes)  # [M_gt, N_pred]
    correct_class = gt_cls[:, None] == pred_cls[None, :]  # [M_gt, N_pred]
    
    for i, threshold in enumerate(iouv):
        valid = (iou >= threshold) & correct_class
        
        if not valid.any():
            continue
        
        gt_indices, pred_indices = torch.where(valid)
        iou_values = iou[gt_indices, pred_indices]
        
        # IoU 내림차순 정렬
        sorted_idx = iou_values.argsort(descending=True)
        gt_indices = gt_indices[sorted_idx]
        pred_indices = pred_indices[sorted_idx]
        
        # Greedy 1:1
        gt_used = set()
        pred_used = set()
        
        for gi, pi in zip(gt_indices.tolist(), pred_indices.tolist()):
            if gi not in gt_used and pi not in pred_used:
                correct[pi, i] = True
                gt_used.add(gi)
                pred_used.add(pi)
    
    return correct


# =============================================================================
# 단일 Config 평가 (공식 mAP50)
# =============================================================================
def evaluate_config(model, img_files, gt_cache, device):
    """
    Arch4 추론 → Ultralytics ap_per_class()로 공식 metric 계산
    """
    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_target_cls = []
    
    iouv = IOUV.to(device)
    
    for img_file in img_files:
        # GT (캐시에서 로드)
        gt = gt_cache.get(img_file, device)
        gt_boxes = gt['boxes']
        gt_cls = gt['classes']
        H, W = gt['img_hw']
        
        all_target_cls.append(gt_cls.cpu())
        
        # Arch4 추론
        img_path = os.path.join(VAL_IMG_DIR, img_file)
        img = cv2.imread(img_path)
        if img is None:
            all_tp.append(torch.zeros((0, len(iouv)), dtype=torch.bool))
            all_conf.append(torch.empty((0,)))
            all_pred_cls.append(torch.empty((0,), dtype=torch.long))
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = (torch.from_numpy(img_rgb)
                        .permute(2, 0, 1).float().div(255.0)
                        .unsqueeze(0).to(device))
        
        try:
            output = model(input_tensor, debug=False)
            det = output['detections'][0]
            pred_boxes = det['boxes'].to(device)
            pred_scores = det['scores'].to(device)
            pred_cls_t = det['classes'].long().to(device)
        except Exception:
            pred_boxes = torch.empty((0, 4), device=device)
            pred_scores = torch.empty((0,), device=device)
            pred_cls_t = torch.empty((0,), dtype=torch.long, device=device)
        
        # TP Matching
        if len(pred_boxes) == 0:
            tp = torch.zeros((0, len(iouv)), dtype=torch.bool)
            conf = torch.empty((0,))
            pcls = torch.empty((0,), dtype=torch.long)
        else:
            tp = match_predictions(pred_boxes, pred_cls_t, gt_boxes, gt_cls, iouv).cpu()
            conf = pred_scores.cpu()
            pcls = pred_cls_t.cpu()
        
        all_tp.append(tp)
        all_conf.append(conf)
        all_pred_cls.append(pcls)
    
    # 합치기
    all_tp = torch.cat(all_tp, dim=0).numpy()
    all_conf = torch.cat(all_conf, dim=0).numpy()
    all_pred_cls = torch.cat(all_pred_cls, dim=0).numpy()
    all_target_cls = torch.cat(all_target_cls, dim=0).numpy()
    
    # Ultralytics 공식 AP 계산
    if len(all_conf) == 0 or len(all_target_cls) == 0:
        return {
            'mAP50': 0.0, 'mAP50_95': 0.0,
            'P': 0.0, 'R': 0.0, 'F1': 0.0,
            'total_gt': len(all_target_cls), 'total_preds': 0
        }
    
    results = ap_per_class(
        all_tp, all_conf, all_pred_cls, all_target_cls,
        plot=False, names={0: 'ship'}
    )
    
    # Ultralytics 8.3.252 반환값 구조:
    # [0,1]: 내부 카운터, [2]: P, [3]: R, [4]: F1, [5]: AP
    p, r, f1, ap = results[2], results[3], results[4], results[5]
    
    if ap.ndim == 2:
        map50 = float(ap[:, 0].mean())
        map50_95 = float(ap.mean())
    else:
        map50 = float(ap.mean())
        map50_95 = float(ap.mean())
    
    return {
        'mAP50': map50,
        'mAP50_95': map50_95,
        'P': float(p.mean()) if hasattr(p, 'mean') else float(p),
        'R': float(r.mean()) if hasattr(r, 'mean') else float(r),
        'F1': float(f1.mean()) if hasattr(f1, 'mean') else float(f1),
        'total_gt': len(all_target_cls),
        'total_preds': len(all_conf)
    }


# =============================================================================
# Config → Arch4 모델 생성
# =============================================================================
def build_model(params, device):
    config_dict = CONFIG_BASE.copy()
    config_dict['model'] = {
        'yolo': {
            'weights_lr': CONFIG_BASE['yolo_weights_lr'],
            'weights_hr': CONFIG_BASE['yolo_weights_hr'],
            'num_classes': 1
        },
        'sr': {
            'type': 'rfdn',
            'weights': CONFIG_BASE['sr_weights'],
            'rfdn': {'nf': 50, 'num_modules': 4}
        },
        'arch4': {
            'pass1_conf': params['pass1_conf'],
            'pass2_conf': params['pass2_conf'],
            'final_conf': params['final_conf'],
            'roi_expansion': params['roi_expansion'],
            'crop_size_lr': params['crop_size_lr'],
            'merge_iou': 0.5,
            'batch_size_sr': 32
        }
    }
    config_dict['data'] = {'upscale_factor': 4}
    return Arch4Adaptive(config_dict).to(device)


# =============================================================================
# 조합 → 고유 키 (resume용)
# =============================================================================
def params_to_key(params):
    return (f"crop{params['crop_size_lr']}_"
            f"p1{params['pass1_conf']}_"
            f"p2{params['pass2_conf']}_"
            f"f{params['final_conf']}_"
            f"roi{params['roi_expansion']}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from saved CSV')
    args = parser.parse_args()
    
    device = CONFIG_BASE['device']
    print("=" * 70)
    print("🚀 Arch4 Grid Search — Ultralytics Official Metrics")
    print(f"   Device: {device}")
    print(f"   Samples: {NUM_SAMPLES}")
    print(f"   Output: {OUTPUT_CSV}")
    print("=" * 70)
    
    # 1. 데이터셋 샘플링
    all_files = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.endswith(('.jpg', '.png'))])
    print(f"\nTotal val images: {len(all_files)}")
    
    random.seed(RANDOM_SEED)
    if len(all_files) > NUM_SAMPLES:
        sampled_files = random.sample(all_files, NUM_SAMPLES)
    else:
        sampled_files = all_files
    print(f"Using {len(sampled_files)} sampled images (seed={RANDOM_SEED})")
    
    # 2. GT 캐싱 (한 번만!)
    gt_cache = GTCache(sampled_files, VAL_IMG_DIR, VAL_LABEL_DIR)
    
    # 3. 조합 생성
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    print(f"\nTotal combinations: {len(combinations)}")
    
    # 4. Resume 처리
    completed_keys = set()
    results = []
    
    if args.resume and os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        results = df_existing.to_dict('records')
        
        for row in results:
            key = params_to_key({
                'crop_size_lr': row['crop_size_lr'],
                'pass1_conf': row['pass1_conf'],
                'pass2_conf': row['pass2_conf'],
                'final_conf': row['final_conf'],
                'roi_expansion': row['roi_expansion']
            })
            completed_keys.add(key)
        
        print(f"📂 Resumed: {len(completed_keys)}/{len(combinations)} already done")
    
    remaining = [(i, p) for i, p in enumerate(combinations) 
                 if params_to_key(p) not in completed_keys]
    
    print(f"🔄 Remaining: {len(remaining)} combinations")
    
    est_hours = len(remaining) * len(sampled_files) * 0.2 / 3600
    print(f"⏱️  Estimated time: {est_hours:.1f} hours")
    
    # 5. Grid Search
    for combo_idx, (orig_idx, params) in enumerate(tqdm(remaining, desc="Grid Search")):
        
        # 모델 로드 (매 조합마다 - Arch4 threshold가 다르니까)
        model = build_model(params, device)
        
        # 평가
        st = time.time()
        metrics = evaluate_config(model, sampled_files, gt_cache, device)
        elapsed = time.time() - st
        
        # 결과 저장
        res = params.copy()
        res.update({
            'mAP50':     round(metrics['mAP50'], 4),
            'mAP50_95':  round(metrics['mAP50_95'], 4),
            'F1':        round(metrics['F1'], 4),
            'Precision':  round(metrics['P'], 4),
            'Recall':    round(metrics['R'], 4),
            'total_gt':  metrics['total_gt'],
            'total_preds': metrics['total_preds'],
            'time_sec':  round(elapsed, 1)
        })
        results.append(res)
        
        # 중간 저장 (매 조합마다)
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        # 20개마다 현재 best 출력
        if (combo_idx + 1) % 20 == 0:
            df_tmp = pd.DataFrame(results)
            best = df_tmp.loc[df_tmp['mAP50'].idxmax()]
            print(f"\n  📊 [{combo_idx+1}/{len(remaining)}] "
                  f"Current Best mAP50={best['mAP50']:.4f} | "
                  f"F1={best['F1']:.4f} | "
                  f"crop={int(best['crop_size_lr'])} "
                  f"p1={best['pass1_conf']} "
                  f"p2={best['pass2_conf']} "
                  f"final={best['final_conf']} "
                  f"roi={best['roi_expansion']}")
    
    # 6. 최종 결과
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 90)
    print("🏆 Top 10 by mAP50")
    print("=" * 90)
    df_map = df.sort_values('mAP50', ascending=False)
    cols = ['crop_size_lr', 'pass1_conf', 'pass2_conf', 'final_conf', 
            'roi_expansion', 'mAP50', 'F1', 'Precision', 'Recall']
    print(df_map.head(10)[cols].to_string(index=False))
    
    print("\n" + "=" * 90)
    print("🏆 Top 10 by F1")
    print("=" * 90)
    df_f1 = df.sort_values('F1', ascending=False)
    print(df_f1.head(10)[cols].to_string(index=False))
    
    print("\n" + "=" * 90)
    print("🏆 Top 10 by Recall")
    print("=" * 90)
    df_rec = df.sort_values('Recall', ascending=False)
    print(df_rec.head(10)[cols].to_string(index=False))
    
    # 최종 저장 (mAP50 순)
    df_map.to_csv(OUTPUT_CSV, index=False)
    
    # Best 출력
    best = df_map.iloc[0]
    print(f"\n{'='*70}")
    print(f"🥇 BEST CONFIG (by mAP50)")
    print(f"   crop_size_lr = {int(best['crop_size_lr'])}")
    print(f"   pass1_conf   = {best['pass1_conf']}")
    print(f"   pass2_conf   = {best['pass2_conf']}")
    print(f"   final_conf   = {best['final_conf']}")
    print(f"   roi_expansion = {best['roi_expansion']}")
    print(f"   ──────────────────────────")
    print(f"   mAP50     = {best['mAP50']:.4f}")
    print(f"   mAP50-95  = {best['mAP50_95']:.4f}")
    print(f"   F1        = {best['F1']:.4f}")
    print(f"   Precision = {best['Precision']:.4f}")
    print(f"   Recall    = {best['Recall']:.4f}")
    print(f"{'='*70}")
    
    print(f"\n✅ All {len(results)} results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()