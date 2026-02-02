#!/usr/bin/env python
"""
=============================================================================
arch4_step1_extract_data.py - 시뮬레이션을 위한 Raw 데이터 추출
=============================================================================
- LR YOLO를 conf=0.001로 실행하여 모든 잠재적 박스 검출
- GT와 IoU 매칭을 수행하여 TP/FP 여부 미리 판별
- 결과를 JSON으로 저장 (이후 시뮬레이션에서 사용)
"""

import sys
from pathlib import Path
import argparse
import json
import torch
import yaml
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import torchvision.ops as ops

def load_gt_labels(label_path, img_width, img_height):
    """YOLO 라벨 파일 로드 및 절대 좌표(xyxy) 변환"""
    boxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) >= 5:
                    # class, x_center, y_center, w, h
                    cls, xc, yc, w, h = parts[0], parts[1], parts[2], parts[3], parts[4]
                    
                    # Normalized xywh -> Absolute xyxy
                    x1 = (xc - w / 2) * img_width
                    y1 = (yc - h / 2) * img_height
                    x2 = (xc + w / 2) * img_width
                    y2 = (yc + h / 2) * img_height
                    boxes.append([x1, y1, x2, y2])
    return torch.tensor(boxes, dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description='Arch4 데이터 추출기 (Step 1)')
    
    # ★ 여기가 입력하신 인자들을 받는 부분입니다 ★
    parser.add_argument('--weights', type=str, required=True, help='LR YOLO weights path')
    parser.add_argument('--data_yaml', type=str, required=True, help='Dataset YAML path')
    parser.add_argument('--output', type=str, default='result/arch4_step1_threshold_detections.json', help='Output JSON path')
    parser.add_argument('--conf_limit', type=float, default=0.001, help='최소 추출 conf (낮을수록 좋음)')
    
    args = parser.parse_args()

    # 1. 설정 로드
    print(f"📂 설정 로드 중: {args.data_yaml}")
    with open(args.data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # 경로 처리 (data.yaml 기준)
    base_path = Path(data_config.get('path', ''))
    
    # 절대 경로인지 상대 경로인지 확인하여 처리
    val_path_str = data_config.get('val', 'images/val')
    if Path(val_path_str).is_absolute():
        val_img_dir = Path(val_path_str)
    else:
        val_img_dir = base_path / val_path_str
        
    # 라벨 경로는 보통 images -> labels 로 치환
    val_label_dir = Path(str(val_img_dir).replace('images', 'labels'))
    
    # 이미지 리스트
    image_files = sorted(list(val_img_dir.glob('*.jpg')) + list(val_img_dir.glob('*.png')))
    print(f"📊 데이터셋 확인: {len(image_files)} images from {val_img_dir}")
    print(f"📂 라벨 경로 추정: {val_label_dir}")

    # 2. 모델 로드
    print(f"🤖 모델 로드 중: {args.weights}")
    model = YOLO(args.weights)

    extracted_data = []
    total_gt_count = 0

    print(f"\n🚀 데이터 추출 시작 (Conf >= {args.conf_limit})...")
    
    # 진행률 표시
    for img_path in tqdm(image_files):
        # GT 로드
        label_path = val_label_dir / f"{img_path.stem}.txt"
        
        # 추론 (이미지 크기 정보 필요하므로 로드 대신 predict 리턴값 활용)
        # iou=0.6: NMS를 위한 임계값 (중복 박스 제거용)
        results = model.predict(img_path, conf=args.conf_limit, verbose=False, iou=0.6)
        result = results[0]
        h, w = result.orig_shape
        
        # GT 박스 가져오기
        gt_boxes = load_gt_labels(label_path, w, h)
        gt_count = len(gt_boxes)
        total_gt_count += gt_count

        # 예측 박스 가져오기
        preds = result.boxes.data.cpu() # x1, y1, x2, y2, conf, cls
        
        img_result = {
            'image': img_path.name,
            'gt_count': gt_count,
            'detections': []
        }

        # 매칭 로직 (IoU > 0.5)
        if len(preds) > 0:
            det_boxes = preds[:, :4]
            scores = preds[:, 4]
            
            matched_gt = set()
            
            # GT가 있을 때만 매칭 수행
            if gt_count > 0:
                iou_matrix = ops.box_iou(det_boxes, gt_boxes) # (N_pred, M_gt)
                
                # Confidence 높은 순으로 정렬하여 Greedy Matching
                sorted_idx = torch.argsort(scores, descending=True)
                
                for idx in sorted_idx:
                    idx = idx.item()
                    box_ious = iou_matrix[idx]
                    
                    if box_ious.numel() == 0: break
                    
                    max_iou, max_gt_idx = torch.max(box_ious, dim=0)
                    max_gt_idx = max_gt_idx.item()
                    
                    det_type = 'FP'
                    if max_iou >= 0.5 and max_gt_idx not in matched_gt:
                        det_type = 'TP'
                        matched_gt.add(max_gt_idx)
                    
                    img_result['detections'].append({
                        'conf': float(scores[idx]),
                        'iou': float(max_iou),
                        'type': det_type
                    })
            else:
                # GT가 없으면 모든 예측은 FP
                for i in range(len(preds)):
                    img_result['detections'].append({
                        'conf': float(scores[i]),
                        'iou': 0.0,
                        'type': 'FP'
                    })

        extracted_data.append(img_result)

    # 3. 저장
    output_data = {
        'total_images': len(image_files),
        'total_gt_ships': total_gt_count,
        'data': extracted_data
    }
    
    # 출력 디렉토리 생성
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\n✅ 추출 완료! 저장됨: {args.output}")
    print(f"   총 GT 선박 수: {total_gt_count}")
    print(f"   총 이미지 수: {len(image_files)}")

if __name__ == '__main__':
    main()