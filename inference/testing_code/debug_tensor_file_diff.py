#!/usr/bin/env python
"""
mAP 계산 검증: Ultralytics 내장 val() 사용
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO


def main():
    weights_path = 'weights/yolohr/8s/best.pt'
    
    print(f"\n{'='*70}")
    print(f"📊 Ultralytics 공식 mAP 측정")
    print(f"{'='*70}")
    
    yolo = YOLO(weights_path)
    
    # =========================================================================
    # 방법 1: Ultralytics val() - 파일 입력 방식의 공식 mAP
    # =========================================================================
    print("\n[1] Ultralytics val() - 공식 mAP")
    
    # data.yaml 경로 (없으면 직접 만들어야 함)
    data_yaml = '/home/changmin/smart_airbus_data/data.yaml'
    
    if Path(data_yaml).exists():
        results = yolo.val(
            data=data_yaml,
            conf=0.001,  # val에서는 낮은 conf 사용
            iou=0.5,
            verbose=False
        )
        
        print(f"  mAP@0.5: {results.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"  Precision: {results.box.mp:.4f}")
        print(f"  Recall: {results.box.mr:.4f}")
    else:
        print(f"  ❌ data.yaml 없음: {data_yaml}")
        print(f"  → data.yaml 경로를 알려주세요!")
    
    # =========================================================================
    # 방법 2: 단일 이미지 비교 (좌표 차이 확인)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"[2] 단일 이미지 좌표 비교")
    print(f"{'='*70}")
    
    test_img = '/home/changmin/smart_airbus_data/images/val/0a7a0fa38.jpg'
    
    # 파일 입력
    file_results = yolo(test_img, conf=0.25, verbose=False)
    
    # Tensor 입력  
    img = Image.open(test_img).convert('RGB')
    tensor = T.ToTensor()(img).unsqueeze(0).cuda()
    tensor_results = yolo.predict(source=tensor, conf=0.25, verbose=False)
    
    print(f"\n[파일 입력]")
    for r in file_results:
        if r.boxes is not None and len(r.boxes) > 0:
            for i, (box, score) in enumerate(zip(r.boxes.xyxy, r.boxes.conf)):
                print(f"  [{i}] box={box.cpu().numpy()}, score={score.item():.4f}")
    
    print(f"\n[Tensor 입력]")
    for r in tensor_results:
        if r.boxes is not None and len(r.boxes) > 0:
            for i, (box, score) in enumerate(zip(r.boxes.xyxy, r.boxes.conf)):
                print(f"  [{i}] box={box.cpu().numpy()}, score={score.item():.4f}")


if __name__ == '__main__':
    main()