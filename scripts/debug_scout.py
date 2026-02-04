import torch
import cv2
import numpy as np
import os
from src.models.pipelines.arch4_adaptive import Arch4Adaptive

# =============================================================================
# [설정] 경로 확인 필수!
# =============================================================================
IMG_PATH = "/home/changmin/smart_airbus_data_lr/images/val/0a24a4100.jpg"          # 테스트할 이미지
WEIGHTS_LR = "/home/changmin/yolov8s+airbus_smartdata/weights/best.pt"   # LR 가중치
WEIGHTS_HR = "/home/changmin/yolov8s+HR_airbus_smartdata/weights/best.pt"   # (형식상 필요)
WEIGHTS_SR = "/home/changmin/dark_vessel_sr_yolo/weights/rfdn/model_best.pt"      # (형식상 필요)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 임시 Config (Scout 문턱을 바닥까지 낮춤)
    config = {
        'data': {'upscale_factor': 4},
        'model': {
            'yolo': {'weights_hr': WEIGHTS_HR, 'weights_lr': WEIGHTS_LR, 'classes': 1},
            'sr': {'type': 'rfdn', 'weights': WEIGHTS_SR, 'rfdn': {'nf': 50, 'num_modules': 4}},
            'arch4': {
                'pass1_conf': 0.01,   # ★★★ 0.01로 극단적으로 낮춤 ★★★
                'high_conf': 0.45,
                'final_conf': 0.25,
                'merge_iou': 0.5, 'roi_expansion': 1.5, 'crop_size_lr': 16, 'batch_size_sr': 32
            }
        }
    }
    
    # 2. 모델 로드
    print(f"🚀 Scout 정밀 진단 시작 (Threshold: 0.01)...")
    model = Arch4Adaptive(config).to(device)
    
    # 3. 이미지 로드
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        print(f"❌ 이미지를 못 찾았습니다: {IMG_PATH}")
        return

    # 전처리
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    # 4. 추론 (Debug Mode)
    output = model(input_tensor, debug=True)
    debug_info = output['debug_info']
    
    # 5. Pass 1 (Scout) 결과 강제 시각화
    pass1_res = debug_info['pass1_raw'][0]
    boxes = pass1_res['boxes'].cpu().numpy()
    scores = pass1_res['scores'].cpu().numpy()
    
    print(f"\n📊 [진단 결과]")
    print(f"  > 탐지된 후보 박스 개수: {len(boxes)}")
    
    if len(boxes) > 0:
        print(f"  > 최고 점수: {scores.max():.4f}")
        print(f"  > 평균 점수: {scores.mean():.4f}")
        
        # 그림 그리기
        vis_img = img_bgr.copy()
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            
            # 점수에 따라 색상 다르게 (낮으면 빨강, 높으면 초록)
            if score < 0.1:
                color = (0, 0, 255) # 빨강 (원래라면 버려질 놈)
            else:
                color = (0, 255, 0) # 초록 (살아남은 놈)
                
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(vis_img, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        save_path = "debug_scout_result.jpg"
        cv2.imwrite(save_path, vis_img)
        print(f"✅ 시각화 저장 완료: {save_path}")
        print("👉 빨간 박스: 점수 0.1 미만 (원래 무시됨)")
        print("👉 초록 박스: 점수 0.1 이상 (SR 후보)")
    else:
        print("❌ 0.01로 낮췄는데도 아무것도 못 찾았습니다. 가중치(weights)나 이미지를 확인하세요.")

if __name__ == "__main__":
    main()