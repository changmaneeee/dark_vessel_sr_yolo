# training/gate_arch2/generate_gate_labels_comparison.py

import argparse
import json
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# 프로젝트 루트 경로 추가 (모듈 import용)
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.sr_models.rfdn import RFDN

def setup_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_gt_labels(label_path: Path) -> List[List[float]]:
    """YOLO 포맷 GT 라벨 로드"""
    if not label_path.exists():
        return []
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # class, x, y, w, h
                labels.append([float(x) for x in parts[:5]])
    return labels

def xywh_to_xyxy(box, img_w, img_h):
    x, y, w, h = box
    x1 = (x - w/2) * img_w
    y1 = (y - h/2) * img_h
    x2 = (x + w/2) * img_w
    y2 = (y + h/2) * img_h
    return [x1, y1, x2, y2]

def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    unionArea = box1Area + box2Area - interArea
    if unionArea == 0: return 0
    return interArea / unionArea

def get_best_match_conf(pred_boxes, pred_confs, gt_boxes, iou_thresh=0.45):
    """GT와 매칭되는 예측 중 가장 높은 Confidence 반환"""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    
    max_conf = 0.0
    
    # 각 GT에 대해 가장 잘 맞는 예측을 찾음
    for gt_box in gt_boxes:
        gt_matched_conf = 0.0
        for pb, pc in zip(pred_boxes, pred_confs):
            if compute_iou(pb, gt_box) >= iou_thresh:
                gt_matched_conf = max(gt_matched_conf, float(pc))
        
        # 이미지 내에서 가장 확신있게 찾은 GT의 점수를 기록 (하나라도 잘 찾으면 됨)
        max_conf = max(max_conf, gt_matched_conf)
        
    return max_conf

def main():
    parser = argparse.ArgumentParser(description='Generate Comparison-based Gate Labels')
    parser.add_argument('--lr_root', type=str, required=True, help='LR Dataset Root')
    parser.add_argument('--label_root', type=str, required=True, help='HR Dataset Root (for labels)')
    parser.add_argument('--sr_weights', type=str, default='weights/rfdn/model_best.pt', help='RFDN Weights')
    parser.add_argument('--yolo_weights', type=str, default='weights/yolohr/8s/best.pt', help='YOLO Weights')
    parser.add_argument('--output', type=str, default='training/gate_arch2/labels_v2', help='Output Directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'both'])
    parser.add_argument('--device', type=str, default='cuda')
    
    # 전략 파라미터
    parser.add_argument('--margin', type=float, default=0.10, help='SR이 LR보다 이만큼은 더 좋아야 함 (0.1 = 10%)')
    
    args = parser.parse_args()
    setup_seed()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device

    # 1. 모델 로드
    print(f"[Init] Loading Models...")
    
    # SR Model (RFDN)
    rfdn = RFDN(in_channels=3, out_channels=3, nf=50, num_modules=4, upscale=4).to(device)
    if Path(args.sr_weights).exists():
        ckpt = torch.load(args.sr_weights, map_location=device)
        # state_dict 키 처리 (혹시 모를 prefix 제거)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        rfdn.load_state_dict(state_dict, strict=False)
        print(f"  ✓ RFDN Loaded: {args.sr_weights}")
    else:
        print(f"  ❌ RFDN weights not found at {args.sr_weights}")
        return
    rfdn.eval()
    
    # YOLO Model
    yolo = YOLO(args.yolo_weights)
    print(f"  ✓ YOLO Loaded: {args.yolo_weights}")
    
    splits = ['train', 'val'] if args.split == 'both' else [args.split]
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"🚀 Processing Split: {split}")
        print(f"{'='*60}")
        
        img_dir = Path(args.lr_root) / 'images' / split
        lbl_dir = Path(args.label_root) / 'labels' / split
        
        img_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        
        labels = {}
        details = []
        stats = {'total': 0, 'sr_better': 0, 'lr_good_enough': 0, 'no_gt': 0, 'both_fail': 0}
        
        for img_path in tqdm(img_files):
            stats['total'] += 1
            img_name = img_path.stem
            lbl_path = lbl_dir / f"{img_name}.txt"
            
            # GT 로드
            gt_labels = load_gt_labels(lbl_path)
            
            # 1. GT가 없는 경우 (배경) -> SR 불필요 (0)
            if not gt_labels:
                labels[img_name] = 0
                stats['no_gt'] += 1
                continue
                
            # 이미지 로드
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            h, w = img_bgr.shape[:2]
            
            # GT Box 좌표 변환 (Normalized xywh -> Pixel xyxy)
            gt_boxes = [xywh_to_xyxy(l[1:], w, h) for l in gt_labels] # LR 이미지 기준 좌표 (작음)
            # YOLO는 내부적으로 리사이징하므로 LR 이미지에 대한 box면 됨.
            # 하지만 비교를 위해 SR 이미지에서의 좌표는 4배 뻥튀기 필요할 수 있음.
            # *중요*: Ultralytics YOLO는 입력 이미지 크기에 맞춰 좌표를 뱉음.
            # 편의상 모든 좌표 비교를 "0.0~1.0 Normalized" 혹은 "GT 절대 좌표"로 통일해야 함.
            # 여기서는 YOLO가 반환하는 Box를 사용하여 IoU 계산 시 좌표계만 맞추면 됨.
            
            # 2. LR Inference
            res_lr = yolo(img_bgr, verbose=False)[0]
            boxes_lr = res_lr.boxes.xyxy.cpu().numpy()
            confs_lr = res_lr.boxes.conf.cpu().numpy()
            
            # LR 점수 계산 (GT와 매칭되는 것 중 최고점)
            score_lr = get_best_match_conf(boxes_lr, confs_lr, gt_boxes)
            
            # 3. SR Inference
            # Preprocess
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            t_img = torch.from_numpy(img_rgb).permute(2,0,1).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                t_sr = rfdn(t_img) # RFDN은 0~255 입력을 받아서 0~255 출력을 내뱉음
                
                # 2. 출력: 이미 0~255 범위이므로 * 255.0 하지 않음
                sr_np = t_sr.squeeze().permute(1,2,0).cpu().numpy()
                
                # 안전장치: 혹시 모를 범위 초과만 자름
                sr_np = np.clip(sr_np, 0, 255).astype(np.uint8)
                sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
            
            # 4. YOLO on SR
            res_sr = yolo(sr_bgr, verbose=False)[0]
            boxes_sr = res_sr.boxes.xyxy.cpu().numpy()
            confs_sr = res_sr.boxes.conf.cpu().numpy()
            
            # SR 이미지 좌표계는 LR의 4배임. GT 박스도 4배 해줘야 매칭 가능
            gt_boxes_sr = [[c*4 for c in b] for b in gt_boxes]
            
            # SR 점수 계산
            score_sr = get_best_match_conf(boxes_sr, confs_sr, gt_boxes_sr)
            
            # 5. 비교 및 라벨링 (Decision Logic)
            # SR이 LR보다 'margin' 만큼 더 확신할 때만 1
            if score_sr > score_lr + args.margin:
                label = 1
                stats['sr_better'] += 1
                reason = f"Improved: {score_lr:.2f} -> {score_sr:.2f}"
            else:
                label = 0
                if score_lr > 0.8:
                    stats['lr_good_enough'] += 1
                    reason = f"LR Sufficient: {score_lr:.2f}"
                else:
                    stats['both_fail'] += 1
                    reason = f"No Gain: {score_lr:.2f} -> {score_sr:.2f}"
            
            labels[img_name] = label
            details.append({
                'image': img_name,
                'label': label,
                'score_lr': round(score_lr, 3),
                'score_sr': round(score_sr, 3),
                'reason': reason
            })
            
        # 저장
        json_path = output_dir / f'gate_labels_v2_{split}.json'
        with open(json_path, 'w') as f:
            json.dump(labels, f, indent=2)
            
        csv_path = output_dir / f'gate_labels_v2_{split}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'label', 'score_lr', 'score_sr', 'reason'])
            writer.writeheader()
            writer.writerows(details)
            
        print(f"\n[Stats {split}]")
        print(f"  Total: {stats['total']}")
        print(f"  Label 1 (SR Needed): {stats['sr_better']} ({stats['sr_better']/stats['total']*100:.1f}%)")
        print(f"  Label 0 (Bypass):    {stats['total'] - stats['sr_better']}")
        print(f"    - LR Sufficient:   {stats['lr_good_enough']}")
        print(f"    - No Gain/Fail:    {stats['both_fail']}")
        print(f"    - No GT:           {stats['no_gt']}")
        print(f"Saved to: {json_path}")

if __name__ == '__main__':
    main()