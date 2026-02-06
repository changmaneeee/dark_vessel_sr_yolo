import torch
import cv2
import os
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from pathlib import Path
import time
import sys
import random

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.pipelines.arch4_adaptive import Arch4Adaptive
from torchvision.ops import box_iou

# =============================================================================
# [설정] 탐색할 파라미터 그리드 (Total: 108가지 조합)
# =============================================================================
PARAM_GRID = {
    # 1. Resizing 크기 (가장 중요!)
    # 16(SR후 64px), 32(SR후 128px), 48(SR후 192px)
    'crop_size_lr':  [16, 32, 48],   

    # 2. Scout 민감도 (LR 탐지)
    # 0.01은 필수, 0.05는 옵션
    'pass1_conf':    [0.001, 0.005, 0.01],         

    # 3. Filter 기준 (SR 보낼지 말지)
    # 너무 많으면 오래 걸리니 핵심 2개만
    'pass2_conf':    [0.1,0.3],      

    # 4. Final 기준 (최종 확정)
    # 정밀도 조절용
    'final_conf':    [0.2, 0.3, 0.4], 

    # 5. Crop 여유 공간
    # 1.5배가 국룰이지만, 좁게/넓게 테스트
    'roi_expansion': [1.0, 1.5, 2.0]       
}

NUM_SAMPLES = 2000  # 시간 절약을 위해 검증 이미지 2000장만 사용

# 총 조합 수 계산: 3 x 5 x 4 x 3 x 3 = 540개
# 장당 0.5초 * 100장 * 540개 = 약 7.5시간 소요 (상당히 오래 걸림)

# =============================================================================
# [고정 설정] 경로를 본인 환경에 맞게 확인하세요!
# =============================================================================
CONFIG_BASE = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 가중치 경로
    'yolo_weights_lr': "/home/changmin/yolov8s+airbus_smartdata/weights/best.pt",
    'yolo_weights_hr': "/home/changmin/yolov8s+HR_airbus_smartdata/weights/best.pt",
    'sr_weights': "/home/changmin/dark_vessel_sr_yolo/weights/rfdn/model_best.pt",
    
    'sr_type': 'rfdn',
    'upscale_factor': 4,
    'batch_size_sr': 32,
    'yolo_classes': 1,
    'merge_iou': 0.5,
    
    # RFDN 설정
    'rfdn_nf': 50,
    'rfdn_modules': 4
}

# 데이터 경로
VAL_IMG_DIR = "/home/changmin/smart_airbus_data_lr/images/val/"
VAL_LABEL_DIR = "/home/changmin/smart_airbus_data_lr/labels/val/"

# 결과 저장 파일
OUTPUT_CSV = "grid_search_results_v2.csv"

# =============================================================================
# Helper Functions
# =============================================================================
def load_yolo_labels(txt_path, img_w, img_h):
    if not os.path.exists(txt_path): return torch.empty((0, 4))
    boxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            xc, yc, w, h = parts[1], parts[2], parts[3], parts[4]
            x1 = (xc - w/2) * img_w
            y1 = (yc - h/2) * img_h
            x2 = (xc + w/2) * img_w
            y2 = (yc + h/2) * img_h
            boxes.append([x1, y1, x2, y2])
    return torch.tensor(boxes)

def evaluate_dataset(model, img_files, conf_threshold):
    """전체 데이터셋에 대해 Precision, Recall, F1 계산"""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for img_file in tqdm(img_files, desc = "Inferencing", leave=False): # Tqdm은 밖에서 찍음
        img_path = os.path.join(VAL_IMG_DIR, img_file)
        label_path = os.path.join(VAL_LABEL_DIR, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        img = cv2.imread(img_path)
        if img is None: continue
        H, W = img.shape[:2]
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(img_rgb).permute(2,0,1).float().div(255.0).unsqueeze(0).to(model.cfg.device)
        
        # Inference
        try:
            output = model(input_tensor, debug=False)
            preds = output['detections'][0]['boxes'].cpu()
        except Exception as e:
            # print(f"Error on {img_file}: {e}")
            preds = torch.empty((0,4))
            
        # Match with GT
        gt_boxes = load_yolo_labels(label_path, W, H)
        
        if len(gt_boxes) == 0:
            total_fp += len(preds)
            continue
            
        if len(preds) == 0:
            total_fn += len(gt_boxes)
            continue
            
        ious = box_iou(preds, gt_boxes)
        max_ious, _ = ious.max(dim=1)
        
        matches = (max_ious >= 0.5)
        tp = matches.sum().item()
        fp = len(preds) - tp
        fn = len(gt_boxes) - tp
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
    epsilon = 1e-6
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return precision, recall, f1

# =============================================================================
# Main Grid Search Loop
# =============================================================================
def main():
    # 1. 데이터셋 준비
    all_files = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.endswith(('.jpg', '.png'))])
    
    # ★시간 절약을 위해 100장만 샘플링 (전체 다 돌리려면 이 줄 주석 처리)★
    # all_files = all_files[:100] 
    
    print(f"Dataset Size: {len(all_files)} images")
    
    if len(all_files) > NUM_SAMPLES:
        random.seed(42)
        sampled_files = random.sample(all_files, NUM_SAMPLES)
        print(f"Sampling {NUM_SAMPLES} images for evaluation to save time.")
    else:
        sampled_files = all_files
        print(f"▶ Using all {len(all_files)} images (dataset is small).")

    # 2. 파라미터 조합 생성
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"Total Combinations: {len(combinations)}")
    print(f"Estimated Time: {len(combinations) * len(all_files) * 0.4 / 60:.1f} minutes") # 장당 0.4초 가정
    
    results = []

    # 3. 반복 수행 (tqdm으로 진행상황 표시)
    for idx, params in enumerate(tqdm(combinations, desc="Grid Search Progress")):
        
        # Config 업데이트
        current_config_dict = CONFIG_BASE.copy()
        current_config_dict['model'] = {
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
                'crop_size_lr': params['crop_size_lr'], # <--- 여기가 핵심 추가!
                'merge_iou': 0.5,
                'batch_size_sr': 32
            }
        }
        current_config_dict['data'] = {'upscale_factor': 4}
        
        # 모델 초기화
        # (logging을 끄기 위해 try-except나 내부 로직 수정이 필요할 수 있으나, 일단 진행)
        model = Arch4Adaptive(current_config_dict).to(CONFIG_BASE['device'])
        
        # 평가 수행
        st = time.time()
        prec, rec, f1 = evaluate_dataset(model, sampled_files, params['final_conf'])
        et = time.time()
        
        # 결과 저장
        res_entry = params.copy()
        res_entry.update({'Precision': prec, 'Recall': rec, 'F1': f1, 'Time': round(et-st, 2)})
        results.append(res_entry)
        
        # 중간 저장
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    # 4. 최종 결과 분석
    df = pd.DataFrame(results)
    df = df.sort_values(by='F1', ascending=False)
    
    print("\n" + "="*50)
    print("🏆 Best 5 Configurations (Sorted by F1)")
    print("="*50)
    print(df.head(5).to_string())
    
    # 파일 저장
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ All results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()