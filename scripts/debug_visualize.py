import sys
from pathlib import Path
import torch
import yaml
import cv2
import numpy as np
from PIL import Image
from types import SimpleNamespace
import torchvision.transforms as T
from src.models.pipelines.arch4_adaptive import Arch4Adaptive

# ==========================================
# 사용자 설정 (경로를 본인 환경에 맞게 수정하세요)
# ==========================================
CONFIG_PATH = 'configs/experiment/arch4_adaptive.yaml'
LR_DATA_YAML = '/home/jovyan/changmin/cv_ship_detact/datas/smart_airbus_dataset_lr/data_smart.yaml'
HR_DATA_YAML = '/home/jovyan/changmin/cv_ship_detact/datas/smart_airbus_dataset/data_smart.yaml'
OUTPUT_DIR = 'debug_images'
DEVICE = 'cuda'

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def get_image_paths(yaml_path):
    cfg = load_config(yaml_path)
    base_path = Path(cfg.get('path', ''))
    # val 경로가 절대 경로인지 상대 경로인지 확인 필요 (보통 yaml 내부는 상대경로)
    img_dir = base_path / 'images' / 'val'
    return sorted(list(img_dir.glob('*.jpg')))[:10] # 10장만 확인

def draw_box(img, box, color, label, thickness=2):
    # box: [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # 1. Config 로드
    with open(CONFIG_PATH) as f:
        config_dict = yaml.safe_load(f)
    
    # Namespace 변환 (Arch4 호환용)
    def dict_to_namespace(d):
        if isinstance(d, dict):
            for k, v in d.items(): d[k] = dict_to_namespace(v)
            return SimpleNamespace(**d)
        return d
    config = dict_to_namespace(config_dict)
    
    # 2. 모델 로드
    print("Loading Model...")
    model = Arch4Adaptive(config).to(DEVICE)
    model.eval()
    
    # 3. 이미지 목록 로드
    lr_paths = get_image_paths(LR_DATA_YAML)
    
    # GT 라벨 경로 계산용
    hr_cfg = load_config(HR_DATA_YAML)
    hr_base = Path(hr_cfg.get('path', ''))
    label_dir = hr_base / 'labels' / 'val'
    
    print(f"Checking {len(lr_paths)} images...")
    
    for i, img_path in enumerate(lr_paths):
        # 이미지 로드
        pil_img = Image.open(img_path).convert('RGB')
        w_lr, h_lr = pil_img.size
        
        # -------------------------------------------------------------
        # [핵심 의심 구간] 스케일링 팩터 확인
        # -------------------------------------------------------------
        # Arch4는 내부적으로 4배 업스케일링을 합니다.
        # 따라서 Canvas 크기는 LR의 4배입니다.
        w_canvas, h_canvas = w_lr * 4, h_lr * 4
        
        # 시각화용 캔버스 (Arch4가 보는 세상)
        # LR 이미지를 4배 키워서 그 위에 박스를 그려봅니다.
        canvas = np.array(pil_img.resize((w_canvas, h_canvas), Image.BILINEAR))
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        
        # 1. 모델 예측 (파란색)
        img_tensor = T.ToTensor()(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            result = model(img_tensor)
        
        det = result['detections'][0]
        action = result['actions'][0]
        
        for box in det['boxes']:
            draw_box(canvas, box.cpu().tolist(), (255, 0, 0), "Pred", 2) # Blue
            
        # 2. 정답(GT) 로드 (빨간색)
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    # YOLO Format: cls cx cy w h (normalized)
                    cls, cx, cy, w, h = parts
                    
                    # ★★★ 여기가 문제일 수 있음 ★★★
                    # GT를 w_canvas(4배)에 맞추는 게 맞는지 확인해야 함
                    x1 = (cx - w/2) * w_canvas
                    y1 = (cy - h/2) * h_canvas
                    x2 = (cx + w/2) * w_canvas
                    y2 = (cy + h/2) * h_canvas
                    
                    draw_box(canvas, [x1, y1, x2, y2], (0, 0, 255), "GT", 2) # Red
        
        # 저장
        cv2.putText(canvas, f"Action: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out_path = f"{OUTPUT_DIR}/debug_{img_path.name}"
        cv2.imwrite(out_path, canvas)
        print(f"Saved {out_path} (LR Size: {w_lr}x{h_lr} -> Canvas: {w_canvas}x{h_canvas})")

if __name__ == '__main__':
    main()