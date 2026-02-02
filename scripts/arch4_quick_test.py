"""
=============================================================================
Arch4 Quick Test - 동작 확인용
=============================================================================

[실행 방법]
python arch4_quick_test.py

먼저 이 스크립트로 모든 경로와 모델이 정상 동작하는지 확인하세요.
"""

import os
import sys
import torch
import torch.nn.functional as F

# ============================================================================
# 경로 설정 (여기만 수정하세요!)
# ============================================================================

PATHS = {
    # YOLO 가중치
    "yolo_lr": "/home/jovyan/changmin/yolov8s+airbus_smartdata/weights/best.pt",
    "yolo_hr": "/home/jovyan/changmin/yolov8s+HR_airbus_smartdata/weights/best.pt",
    
    # RFDN 가중치 (경로 확인 필요)
    "rfdn": "/home/jovyan/changmin/rfdn_model/experiment/rfdn_smart_airbus_final_fix/model/model_best.pt",  # ← 실제 경로로 수정
    
    # 데이터
    "lr_images": "/home/jovyan/changmin/cv_ship_detact/datas/smart_airbus_dataset_lr/images/val",
    "hr_images": "/home/jovyan/changmin/cv_ship_detact/datas/smart_airbus_dataset/images/val",
    "labels": "/home/jovyan/changmin/cv_ship_detact/datas/smart_airbus_dataset/labels/val",
}

UPSCALE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def check_paths():
    """경로 존재 확인"""
    print("\n[1] 경로 확인")
    print("-" * 50)
    
    all_ok = True
    for name, path in PATHS.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗ NOT FOUND"
        print(f"  {name}: {status}")
        print(f"    → {path}")
        if not exists:
            all_ok = False
    
    return all_ok


def check_yolo():
    """YOLO 로드 테스트"""
    print("\n[2] YOLO 로드 테스트")
    print("-" * 50)
    
    try:
        from ultralytics import YOLO
        print("  ultralytics import: ✓")
    except ImportError:
        print("  ultralytics import: ✗")
        print("    → pip install ultralytics")
        return False, None, None
    
    yolo_lr, yolo_hr = None, None
    
    # YOLO_LR
    if os.path.exists(PATHS['yolo_lr']):
        try:
            yolo_lr = YOLO(PATHS['yolo_lr'])
            yolo_lr.to(DEVICE)
            print(f"  YOLO_LR 로드: ✓")
        except Exception as e:
            print(f"  YOLO_LR 로드: ✗ ({e})")
    
    # YOLO_HR
    if os.path.exists(PATHS['yolo_hr']):
        try:
            yolo_hr = YOLO(PATHS['yolo_hr'])
            yolo_hr.to(DEVICE)
            print(f"  YOLO_HR 로드: ✓")
        except Exception as e:
            print(f"  YOLO_HR 로드: ✗ ({e})")
    
    return True, yolo_lr, yolo_hr


def check_rfdn():
    """RFDN 로드 테스트"""
    print("\n[3] RFDN 로드 테스트")
    print("-" * 50)
    
    # RFDN 클래스 import 시도
    rfdn = None
    
    try:
        from src.models.sr_models.rfdn import RFDN
        print("  RFDN import: ✓")
        
        rfdn = RFDN(
            in_channels=3,
            out_channels=3,
            nf=50,
            num_modules=4,
            upscale=UPSCALE,
            input_range='0-255'
        )
        print("  RFDN 모델 생성: ✓")
        
        # 가중치 로드
        if os.path.exists(PATHS['rfdn']):
            checkpoint = torch.load(PATHS['rfdn'], map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'params_ema' in checkpoint:
                    state_dict = checkpoint['params_ema']
                elif 'params' in checkpoint:
                    state_dict = checkpoint['params']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            rfdn.load_state_dict(state_dict, strict=False)
            print(f"  RFDN 가중치 로드: ✓")
        else:
            print(f"  RFDN 가중치: ✗ (파일 없음 - 랜덤 초기화)")
        
        rfdn.to(DEVICE)
        rfdn.eval()
        
    except ImportError as e:
        print(f"  RFDN import: ✗ ({e})")
        print("    → src/models/sr_models/rfdn.py 경로를 확인하세요")
        return None
    except Exception as e:
        print(f"  RFDN 오류: {e}")
        return None
    
    return rfdn


def quick_inference_test(yolo_lr, yolo_hr, rfdn):
    """간단한 추론 테스트"""
    print("\n[4] 추론 테스트")
    print("-" * 50)
    
    from PIL import Image
    import torchvision.transforms as T
    
    # 테스트 이미지 찾기
    lr_dir = PATHS['lr_images']
    if not os.path.exists(lr_dir):
        print(f"  LR 이미지 디렉토리 없음: {lr_dir}")
        return
    
    img_files = [f for f in os.listdir(lr_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not img_files:
        print("  LR 이미지 없음")
        return
    
    test_img_path = os.path.join(lr_dir, img_files[0])
    print(f"  테스트 이미지: {test_img_path}")
    
    # 이미지 로드
    img = Image.open(test_img_path).convert('RGB')
    transform = T.ToTensor()
    lr_tensor = transform(img).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]
    
    print(f"  LR 크기: {lr_tensor.shape}")
    
    # =========================================================================
    # Test 1: YOLO_LR on bilinear upscaled
    # =========================================================================
    print("\n  [Test 1] YOLO_LR on Bilinear Upscaled")
    
    lr_up = F.interpolate(lr_tensor, scale_factor=UPSCALE, mode='bilinear', align_corners=False)
    print(f"    Upscaled 크기: {lr_up.shape}")
    
    if yolo_lr:
        results = yolo_lr.predict(lr_up, conf=0.1, verbose=False)
        num_det = len(results[0].boxes) if results else 0
        print(f"    탐지 수: {num_det}")
        if num_det > 0:
            print(f"    Confidence 범위: {results[0].boxes.conf.min():.3f} ~ {results[0].boxes.conf.max():.3f}")
    
    # =========================================================================
    # Test 2: SR → YOLO_HR
    # =========================================================================
    print("\n  [Test 2] SR (RFDN) → YOLO_HR")
    
    if rfdn:
        with torch.no_grad():
            lr_255 = lr_tensor * 255.0
            hr_255 = rfdn(lr_255)
            hr_tensor = torch.clamp(hr_255 / 255.0, 0, 1)
        print(f"    SR 출력 크기: {hr_tensor.shape}")
        
        if yolo_hr:
            results = yolo_hr.predict(hr_tensor, conf=0.1, verbose=False)
            num_det = len(results[0].boxes) if results else 0
            print(f"    탐지 수: {num_det}")
            if num_det > 0:
                print(f"    Confidence 범위: {results[0].boxes.conf.min():.3f} ~ {results[0].boxes.conf.max():.3f}")
    
    print("\n  ✓ 추론 테스트 완료!")


def main():
    print("=" * 60)
    print("Arch4 Quick Test")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # 1. 경로 확인
    paths_ok = check_paths()
    
    # 2. YOLO 로드
    yolo_ok, yolo_lr, yolo_hr = check_yolo()
    
    # 3. RFDN 로드
    rfdn = check_rfdn()
    
    # 4. 추론 테스트
    if yolo_lr or yolo_hr:
        quick_inference_test(yolo_lr, yolo_hr, rfdn)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    print(f"  경로: {'OK' if paths_ok else 'FAIL - 경로 수정 필요'}")
    print(f"  YOLO_LR: {'OK' if yolo_lr else 'FAIL'}")
    print(f"  YOLO_HR: {'OK' if yolo_hr else 'FAIL'}")
    print(f"  RFDN: {'OK' if rfdn else 'FAIL'}")
    
    if paths_ok and yolo_lr and yolo_hr and rfdn:
        print("\n✓ 모든 준비 완료! arch4_inference.py 실행 가능")
        print("\n실행 예시:")
        print("  python arch4_inference.py --mode compare --num_samples 100")
    else:
        print("\n✗ 위의 FAIL 항목을 먼저 해결하세요")


if __name__ == "__main__":
    main()