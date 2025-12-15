"""
inference_arch0.py - Arch0Sequential 클래스 활용
"""

import torch
import cv2
from pathlib import Path
from types import SimpleNamespace

from src.models.pipelines.arch0_sequential import Arch0Sequential


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # =========================================================================
    # 1. Config 설정
    # =========================================================================
    config = SimpleNamespace(
        model=SimpleNamespace(
            rfdn=SimpleNamespace(nf=50, num_modules=4),
            yolo=SimpleNamespace(
                weights_path="checkpoints/yolo_ship.pt",  # YOLO 가중치
                num_classes=1
            )
        ),
        data=SimpleNamespace(upscale_factor=4),
        training=SimpleNamespace(
            sr_weight=1.0,
            det_weight=0.0,
            freeze_detector=True
        ),
        device=device
    )
    
    # =========================================================================
    # 2. 모델 생성 (Arch0Sequential이 알아서 RFDN + YOLO 생성)
    # =========================================================================
    print("[1/3] Creating Arch0 Pipeline...")
    model = Arch0Sequential(config)
    model = model.to(device)
    
    # =========================================================================
    # 3. 사전학습 가중치 로드
    # =========================================================================
    print("[2/3] Loading pretrained weights...")
    
    # SR 가중치 로드 (우리가 학습시킨 RFDN)
    rfdn_ckpt = torch.load("checkpoints/rfdn_best.pth", map_location=device)
    model.sr_model.load_state_dict(rfdn_ckpt['model_state_dict'])
    print("  ✓ RFDN weights loaded")
    
    # YOLO는 config에서 이미 로드됨
    print("  ✓ YOLO weights loaded (from config)")
    
    # =========================================================================
    # 4. 추론
    # =========================================================================
    print("[3/3] Running inference...")
    model.eval()
    
    # 이미지 로드
    lr_image = cv2.imread("test_image.jpg")
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
    
    # numpy → tensor
    lr_tensor = torch.from_numpy(lr_image).permute(2, 0, 1).float() / 255.0
    lr_tensor = lr_tensor.unsqueeze(0).to(device)
    
    # Forward (Arch0Sequential.forward 호출!)
    with torch.no_grad():
        sr_image, detections = model(lr_tensor)
    
    print(f"  SR shape: {sr_image.shape}")
    print(f"  Detections: {len(detections)} ships found")
    
    print("\n✓ Inference completed!")


if __name__ == "__main__":
    main()
