import torch
import cv2
import numpy as np
import os
from torchvision.ops import box_iou
from src.models.pipelines.arch4_adaptive import Arch4Adaptive
# 큰 선박: 0a9bc3e3a.jpg, 작은 선박: 0a24a4100.jpg, 0aab77784.jpg
# =============================================================================
# [설정] 데이터 경로 (3종 세트가 필요합니다!)
# =============================================================================
IMG_LR_PATH = "/home/changmin/smart_airbus_data_lr/images/val/0a24a4100.jpg"       # 입력 (문제)
IMG_HR_PATH = "/home/changmin/smart_airbus_data/images/val/0a24a4100.jpg"       # 정답 이미지 (SR 평가용)
LABEL_PATH  = "/home/changmin/smart_airbus_data_lr/labels/val/0a24a4100.txt"       # 정답 라벨 (탐지 평가용, YOLO 포맷)

# 가중치 경로
WEIGHTS_LR = "/home/changmin/yolov8s+airbus_smartdata/weights/best.pt"
WEIGHTS_HR = "/home/changmin/yolov8s+HR_airbus_smartdata/weights/best.pt"
WEIGHTS_SR = "/home/changmin/dark_vessel_sr_yolo/weights/rfdn/model_best.pt"

# 평가 기준
IOU_THRESHOLD = 0.5  # 정답으로 인정할 겹침 정도
CONF_THRESHOLD = 0.1

# =============================================================================
# 1. 헬퍼 함수: 라벨 읽기 & PSNR 계산
# =============================================================================
def load_yolo_labels(txt_path, img_w, img_h):
    """YOLO 포맷(.txt) 라벨을 읽어서 [x1, y1, x2, y2] 픽셀 좌표로 변환"""
    if not os.path.exists(txt_path):
        print("⚠️ 라벨 파일이 없습니다. 탐지 정확도를 계산할 수 없습니다.")
        return torch.empty((0, 4))

    boxes = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # format: class x_center y_center w h
            parts = list(map(float, line.strip().split()))
            cls, xc, yc, w, h = parts[0], parts[1], parts[2], parts[3], parts[4]
            
            # 정규화 좌표 -> 픽셀 좌표 변환
            x1 = (xc - w/2) * img_w
            y1 = (yc - h/2) * img_h
            x2 = (xc + w/2) * img_w
            y2 = (yc + h/2) * img_h
            boxes.append([x1, y1, x2, y2])
            
    return torch.tensor(boxes, dtype=torch.float32)

def calculate_psnr(img1, img2):
    """PSNR 계산 (img1: SR, img2: HR)"""
    # 크기가 다르면 HR을 SR 크기로 맞춰서 비교 (혹은 반대)
    if img1.shape != img2.shape:
        h, w = img1.shape[:2]
        img2 = cv2.resize(img2, (w, h))
        
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# =============================================================================
# 2. 메인 검증 로직
# =============================================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 모델 초기화 (설정 생략 - 위와 동일)
    config = {
        'data': {'upscale_factor': 4},
        'model': {
            'yolo': {'weights_hr': WEIGHTS_HR, 'weights_lr': WEIGHTS_LR, 'classes': 1},
            'sr': {'type': 'rfdn', 'weights': WEIGHTS_SR, 'rfdn': {'nf': 50, 'num_modules': 4}},
            'arch4': {
                'pass1_conf': 0.01, 'pass2_conf': 0.45, 'final_conf': CONF_THRESHOLD,
                'merge_iou': 0.5, 'roi_expansion': 1.5, 'crop_size_lr': 16, 'batch_size_sr': 32
            }
        }
    }
    model = Arch4Adaptive(config).to(device)
    
    # 2. 이미지 로드
    img_bgr_lr = cv2.imread(IMG_LR_PATH)
    img_bgr_hr = cv2.imread(IMG_HR_PATH) # PSNR 비교용
    
    if img_bgr_lr is None:
        print("❌ LR 이미지가 없습니다.")
        return

    H, W = img_bgr_lr.shape[:2]
    
    # 전처리
    img_rgb = cv2.cvtColor(img_bgr_lr, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    # 3. Arch4 실행 (Debug Mode)
    print("🚀 Arch4 추론 시작...")
    output = model(input_tensor, debug=True)
    
    final_dets = output['detections'][0]
    debug_info = output['debug_info']
    
    # =========================================================================
    # [검증 1] SR 품질 평가 (PSNR)
    # =========================================================================
    print("\n🔍 [Metric 1] SR Performance (PSNR)")
    
    crops_sr = debug_info['crops_sr'] # 64x64 SR 이미지들 (Tensor)
    crop_meta = debug_info['crop_meta'] # 좌표 정보
    
    psnr_values = []
    
    if img_bgr_hr is not None:
        for i, sr_tensor in enumerate(crops_sr):
            # SR 텐서 -> Numpy 이미지
            sr_numpy = sr_tensor.permute(1, 2, 0).cpu().numpy()  # HWC

            sr_img_float = sr_numpy.astype(np.float32)

            if sr_img_float.max() <= 1.5:
                sr_img_float = sr_img_float * 255.0
            
            sr_img = np.clip(sr_img_float, 0.0, 255.0).astype(np.uint8)
            sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)

            # 해당 좌표의 실제 HR 이미지(GT) 잘라오기
            # meta: (img_idx, (x1, y1, x2, y2)) <- 이건 LR 좌표임!
            _, (lx1, ly1, lx2, ly2) = crop_meta[i]
            print(f"\n[Crop {i}]")
            print(f"  LR 좌표: ({lx1:.1f}, {ly1:.1f}) ~ ({lx2:.1f}, {ly2:.1f})")
            print(f"  HR 좌표: ({lx1*4:.1f}, {ly1*4:.1f}) ~ ({lx2*4:.1f}, {ly2*4:.1f})")
            print(f"  SR tensor shape: {sr_tensor.shape}")           
            # HR 좌표로 변환 (x4)
            scale = 4
            hx1, hy1 = int(lx1 * scale), int(ly1 * scale)
            hx2, hy2 = int(lx2 * scale), int(ly2 * scale)
            
            # HR 이미지 경계 체크
            h_hr, w_hr = img_bgr_hr.shape[:2]
            hx2, hy2 = min(hx2, w_hr), min(hy2, h_hr)
            
            hr_crop_gt = img_bgr_hr[hy1:hy2, hx1:hx2]
            
            if hr_crop_gt.size == 0: continue

            # PSNR 계산
            val = calculate_psnr(sr_img, hr_crop_gt)
            psnr_values.append(val)

            #hr_crop_gt_rgb = cv2.cvtColor(hr_crop_gt, cv2.COLOR_BGR2RGB)
            #val = calculate_psnr(sr_img, hr_crop_gt_rgb)  # 둘 다 RGB
            
            # 샘플 저장 (눈으로 확인)
            #if i < 5:
            #        vis_sr = cv2.resize(sr_img, (128, 128), interpolation=cv2.INTER_NEAREST)
            #        vis_gt = cv2.resize(hr_crop_gt, (128, 128), interpolation=cv2.INTER_NEAREST)
            #        compare = np.hstack([vis_sr, vis_gt])
            #        cv2.imwrite(f"sr_check_{i}.jpg", compare)
            if i < 5:
                sr_for_save = sr_img[:, :, ::-1]  # RGB → BGR (numpy slicing)
                vis_sr = cv2.resize(sr_for_save, (128, 128), interpolation=cv2.INTER_NEAREST)
                vis_gt = cv2.resize(hr_crop_gt, (128, 128), interpolation=cv2.INTER_NEAREST)
                compare = np.hstack([vis_sr, vis_gt])
                cv2.imwrite(f"sr_check_{i}.jpg", compare)

        
        if psnr_values:
            print(f"  > Average PSNR (on {len(psnr_values)} crops): {np.mean(psnr_values):.2f} dB")
        else:
            print("  > SR이 수행된 조각이 없습니다.")
    else:
        print("⚠️ HR 이미지가 없어서 PSNR을 계산할 수 없습니다.")

    # =========================================================================
    # [검증 2] 탐지 정확도 (Accuracy / Recall)
    # =========================================================================
    print("\n🔍 [Metric 2] Detection Accuracy (vs Ground Truth)")
    
    pred_boxes = final_dets['boxes'].cpu()
    gt_boxes = load_yolo_labels(LABEL_PATH, W, H)
    print(f"  > GT Objects: {len(gt_boxes)}")
    print(f"  > Predicted:  {len(pred_boxes)}")
    
    if len(gt_boxes) > 0 and len(pred_boxes) > 0:
        # IoU 계산 (모든 예측박스 vs 모든 정답박스)
        ious = box_iou(pred_boxes, gt_boxes)
        
        # 각 예측 박스가 정답과 0.5 이상 겹쳤는지 확인
        # max(dim=1): 각 예측 박스별로 가장 높은 IoU 값
        max_ious, _ = ious.max(dim=1)
        
        # 맞춘 개수 (True Positive)
        correct_preds = (max_ious >= IOU_THRESHOLD).sum().item()
        
        precision = correct_preds / len(pred_boxes) # 정밀도
        recall = correct_preds / len(gt_boxes)      # 재현율
        
        print(f"  > GT Objects: {len(gt_boxes)}")
        print(f"  > Predicted:  {len(pred_boxes)}")
        print(f"  > Correct (IoU>0.5): {correct_preds}")
        print(f"  > Precision: {precision*100:.2f}% (예측한 것 중 정답 비율)")
        print(f"  > Recall:    {recall*100:.2f}% (실제 정답 중 찾은 비율)")
    else:
        print("  > ⚠️ 예측이 없거나 정답이 없어서 정밀도/재현율 계산 불가")
        
        # 시각화 (정답은 초록, 예측은 빨강)
    vis = img_bgr_lr.copy()

    for box in gt_boxes: # GT
        cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    for box in pred_boxes: # Pred
        cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    cv2.imwrite("accuracy_check.jpg", vis)
    print("  > 시각화 저장됨: accuracy_check.jpg (초록:정답, 빨강:예측)")
        

if __name__ == "__main__":
    main()