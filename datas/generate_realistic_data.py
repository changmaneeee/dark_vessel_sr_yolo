import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import random

# ==========================================
# ⚙️ Configuration (실험 설정)
# ==========================================
INPUT_HR_DIR = '/home/octolab-rtx4090/Desktop/changmin/smart_airbus_data/hr/images/val'  # 원본 HR 이미지 경로 (사용자 환경에 맞게 수정)
OUTPUT_LR_DIR = '/home/octolab-rtx4090/Desktop/changmin/smart_airbus_data/lr/images/val' # 생성될 LR 이미지 경로
SCALE_FACTOR = 4  # 1.5m -> 6m (4배 축소)

# Degradation Hyperparameters (논문에 명시할 값들)
BLUR_SIGMA = 1.0   # 블러 강도 (대기 불안정성 모사, 보통 1.0~1.5 사용)
NOISE_LEVEL = 0.02 # 노이즈 레벨 (픽셀 값의 변동 폭, 0.0~1.0)
SEED = 42          # 재현성을 위한 시드 고정

# ==========================================
# 🛠️ Functions
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def add_gaussian_noise(img, mean=0, sigma=0.05):
    """
    이미지에 Gaussian Noise를 추가합니다.
    이미지는 0~1 사이로 정규화된 상태여야 합니다.
    """
    noise = np.random.normal(mean, sigma, img.shape).astype('float32')
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 1) # 0~1 사이로 다시 클리핑

def degradation_pipeline(img_hr, scale):
    """
    Process: HR -> Blur -> Downsample -> Noise -> LR
    """
    # 1. Gaussian Blur (Simulating Atmospheric Turbulence)
    # 커널 사이즈는 보통 sigma의 3배~6배 사이의 홀수로 설정
    k_size = int(np.ceil(BLUR_SIGMA * 3) * 2 + 1)
    img_blur = cv2.GaussianBlur(img_hr, (k_size, k_size), BLUR_SIGMA)
    
    # 2. Downsampling (Bicubic)
    h, w, _ = img_blur.shape
    new_h, new_w = h // scale, w // scale
    img_lr = cv2.resize(img_blur, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 3. Add Noise (Simulating Sensor Thermal Noise)
    # 노이즈 추가를 위해 float32로 변환 (0~255 -> 0.0~1.0)
    img_lr_float = img_lr.astype(np.float32) / 255.0
    img_lr_noisy = add_gaussian_noise(img_lr_float, sigma=NOISE_LEVEL)
    
    # 다시 8bit 이미지로 변환
    img_lr_final = (img_lr_noisy * 255.0).round().astype(np.uint8)
    
    return img_lr_final

# ==========================================
# 🚀 Main Execution
# ==========================================
def main():
    set_seed(SEED)
    
    if not os.path.exists(OUTPUT_LR_DIR):
        os.makedirs(OUTPUT_LR_DIR)
        print(f"📁 Created directory: {OUTPUT_LR_DIR}")
    
    # 이미지 파일 리스트 로드 (.jpg, .png 등)
    extensions = ['*.jpg', '*.png', '*.jpeg']
    img_list = []
    for ext in extensions:
        img_list.extend(glob.glob(os.path.join(INPUT_HR_DIR, ext)))
    
    print(f"🔍 Found {len(img_list)} images. Starting degradation process...")
    print(f"   - Blur Sigma: {BLUR_SIGMA}")
    print(f"   - Noise Level: {NOISE_LEVEL}")
    print(f"   - Scale Factor: x{SCALE_FACTOR}")

    for img_path in tqdm(img_list):
        # 이미지 로드
        img_name = os.path.basename(img_path)
        img_hr = cv2.imread(img_path)
        
        if img_hr is None:
            print(f"⚠️ Error reading {img_path}, skipping...")
            continue
            
        # 크기가 scale로 나누어떨어지지 않는 경우 처리 (옵션)
        h, w, _ = img_hr.shape
        h = h - (h % SCALE_FACTOR)
        w = w - (w % SCALE_FACTOR)
        img_hr = img_hr[:h, :w, :]

        # 파이프라인 적용
        img_lr = degradation_pipeline(img_hr, SCALE_FACTOR)
        
        # 저장
        save_path = os.path.join(OUTPUT_LR_DIR, img_name)
        cv2.imwrite(save_path, img_lr)

    print("✅ Done! Realistic LR dataset generated.")

if __name__ == '__main__':
    main()