#!/usr/bin/env python
"""
Official YOLO vs YOLOWrapper 결과 비교 (v2 - 상세 분석)

목표: 동일한 이미지에 대해 두 방식의 결과가 같은지 확인
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as T

from src.models.detectors.yolo_wrapper import YOLOWrapper


def load_image_as_tensor(img_path: str, device: str = 'cuda') -> torch.Tensor:
    """이미지를 Tensor로 로드 (0~1, RGB, [1, 3, H, W])"""
    img = Image.open(img_path).convert('RGB')
    transform = T.ToTensor()  # 0~255 → 0~1, HWC → CHW
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor


def compare_methods(img_path: str, weights_path: str, conf: float = 0.25):
    """두 방식 비교"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"🔍 Official YOLO vs YOLOWrapper 비교 (상세 분석)")
    print(f"{'='*70}")
    print(f"이미지: {img_path}")
    print(f"모델: {weights_path}")
    print(f"Confidence: {conf}")
    
    # 이미지 정보
    img = Image.open(img_path)
    print(f"원본 이미지 크기: {img.size}")
    
    # =========================================================================
    # 방법 1: Official YOLO (파일 경로 입력)
    # =========================================================================
    print(f"\n[방법 1] Official YOLO (파일 경로 입력)")
    
    official_yolo = YOLO(weights_path, verbose=False)
    official_results = official_yolo(img_path, conf=conf, verbose=False)
    
    official_boxes = []
    official_scores = []
    for r in official_results:
        if r.boxes is not None and len(r.boxes) > 0:
            official_boxes = r.boxes.xyxy.cpu().numpy()
            official_scores = r.boxes.conf.cpu().numpy()
            # 추가 정보 출력
            print(f"  - 입력 이미지 shape: {r.orig_shape}")
    
    print(f"  - 탐지 수: {len(official_boxes)}")
    for i, (box, score) in enumerate(zip(official_boxes, official_scores)):
        print(f"    [{i}] box={box}, score={score:.4f}")
    
    # =========================================================================
    # 방법 2: YOLOWrapper (Tensor 입력)
    # =========================================================================
    print(f"\n[방법 2] YOLOWrapper (Tensor 입력)")
    
    wrapper = YOLOWrapper(weights_path, device=device, verbose=False)
    img_tensor = load_image_as_tensor(img_path, device)
    
    print(f"  - Tensor shape: {img_tensor.shape}")
    print(f"  - Tensor 값 범위: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    
    wrapper_results = wrapper.predict(img_tensor, conf=conf)
    
    wrapper_boxes = []
    wrapper_scores = []
    if wrapper_results and len(wrapper_results) > 0:
        wrapper_boxes = wrapper_results[0]['boxes'].cpu().numpy()
        wrapper_scores = wrapper_results[0]['scores'].cpu().numpy()
    
    print(f"  - 탐지 수: {len(wrapper_boxes)}")
    for i, (box, score) in enumerate(zip(wrapper_boxes, wrapper_scores)):
        print(f"    [{i}] box={box}, score={score:.4f}")
    
    # =========================================================================
    # 방법 3: Official YOLO + Tensor 입력 (직접 비교)
    # =========================================================================
    print(f"\n[방법 3] Official YOLO + Tensor 입력 (직접)")
    
    # 동일한 Tensor로 Official YOLO 호출
    tensor_results = official_yolo.predict(
        source=img_tensor,  # Tensor 입력
        conf=conf, 
        verbose=False
    )
    
    tensor_boxes = []
    tensor_scores = []
    for r in tensor_results:
        if r.boxes is not None and len(r.boxes) > 0:
            tensor_boxes = r.boxes.xyxy.cpu().numpy()
            tensor_scores = r.boxes.conf.cpu().numpy()
    
    print(f"  - 탐지 수: {len(tensor_boxes)}")
    for i, (box, score) in enumerate(zip(tensor_boxes, tensor_scores)):
        print(f"    [{i}] box={box}, score={score:.4f}")
    
    # =========================================================================
    # 방법 4: Official YOLO + numpy 입력
    # =========================================================================
    print(f"\n[방법 4] Official YOLO + numpy 입력")
    
    # PIL → numpy (RGB, 0-255)
    img_np = np.array(img)
    
    np_results = official_yolo.predict(
        source=img_np,
        conf=conf,
        verbose=False
    )
    
    np_boxes = []
    np_scores = []
    for r in np_results:
        if r.boxes is not None and len(r.boxes) > 0:
            np_boxes = r.boxes.xyxy.cpu().numpy()
            np_scores = r.boxes.conf.cpu().numpy()
    
    print(f"  - 탐지 수: {len(np_boxes)}")
    for i, (box, score) in enumerate(zip(np_boxes, np_scores)):
        print(f"    [{i}] box={box}, score={score:.4f}")
    
    # =========================================================================
    # 비교 분석
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"[비교 결과 요약]")
    print(f"{'='*70}")
    
    print(f"\n  방법 1 (파일 경로):  {len(official_boxes)} 탐지")
    print(f"  방법 2 (Wrapper):    {len(wrapper_boxes)} 탐지")
    print(f"  방법 3 (Tensor):     {len(tensor_boxes)} 탐지")
    print(f"  방법 4 (numpy):      {len(np_boxes)} 탐지")
    
    # 방법 2 vs 방법 3 비교 (둘 다 Tensor)
    if len(wrapper_boxes) > 0 and len(tensor_boxes) > 0:
        print(f"\n[Wrapper vs Official+Tensor 비교]")
        box_diff = np.abs(wrapper_boxes - tensor_boxes).max()
        score_diff = np.abs(wrapper_scores - tensor_scores).max()
        print(f"  Box 최대 차이: {box_diff:.4f} pixels")
        print(f"  Score 최대 차이: {score_diff:.6f}")
        
        if box_diff < 0.1 and score_diff < 0.001:
            print(f"  ✅ Wrapper와 Official+Tensor 결과 동일!")
        else:
            print(f"  ⚠️ Wrapper 내부에서 무언가 다름")
    
    # 방법 1 vs 방법 3 비교 (파일 vs Tensor)
    if len(official_boxes) > 0 and len(tensor_boxes) > 0:
        print(f"\n[파일 입력 vs Tensor 입력 비교]")
        box_diff = np.abs(official_boxes - tensor_boxes).max()
        score_diff = np.abs(official_scores - tensor_scores).max()
        print(f"  Box 최대 차이: {box_diff:.4f} pixels")
        print(f"  Score 최대 차이: {score_diff:.6f}")
        
        if box_diff < 0.1 and score_diff < 0.001:
            print(f"  ✅ 파일/Tensor 결과 동일 → 전처리 동일")
        else:
            print(f"  ⚠️ Ultralytics 내부에서 파일/Tensor 전처리가 다름!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='테스트 이미지 경로')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='YOLO 모델')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()
    
    compare_methods(args.image, args.weights, args.conf)