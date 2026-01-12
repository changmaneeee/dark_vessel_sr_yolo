import os
import shutil
import random
from tqdm import tqdm
from pathlib import Path

# =========================================================
# [설정] PC 환경에 맞게 경로 수정
# =========================================================
SOURCE_ROOT = "/home/changmin/cv_ship_detact/datas/airbus_dataset"
DEST_ROOT = "/home/changmin/smart_airbus_data"
# =========================================================
def create_smart_subset_v2():
    print("🚀 스크립트가 시작되었습니다! (Logic: Scan Images first)", flush=True)
    
    src_path = Path(SOURCE_ROOT)
    dest_path = Path(DEST_ROOT)
    
    # 소스 경로
    src_train_img = src_path / "images" / "train"
    src_train_lbl = src_path / "labels" / "train"
    
    # 목적지 경로
    dst_train_img = dest_path / "images" / "train"
    dst_train_lbl = dest_path / "labels" / "train"

    # 1. 이미지 폴더 스캔 (여기가 핵심 변경점!)
    if not src_train_img.exists():
        print(f"❌ [Error] 이미지 폴더를 찾을 수 없습니다: {src_train_img}")
        return

    # 폴더 생성
    for p in [dst_train_img, dst_train_lbl]:
        p.mkdir(parents=True, exist_ok=True)

    print(f"🔍 [Phase 1] 이미지 파일 스캔 중... (Images 기준 분류)", flush=True)
    
    ship_images = []   # (img_path, lbl_path) 튜플 저장
    empty_images = []  # (img_path, lbl_path_expected) 튜플 저장

    # 이미지 확장자 (보통 jpg)
    valid_exts = {'.jpg', '.jpeg', '.png'}
    
    # scandir로 고속 스캔
    img_files = [f for f in os.scandir(src_train_img) if f.is_file() and Path(f.name).suffix in valid_exts]
    print(f"📂 총 {len(img_files)}개의 이미지를 발견했습니다. 분류 시작...", flush=True)

    for entry in tqdm(img_files, desc="Classifying"):
        img_path = Path(entry.path)
        lbl_name = img_path.stem + ".txt"
        lbl_path = src_train_lbl / lbl_name
        
        # 라벨 파일이 존재하고, 내용이 있으면 배가 있는 것
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            ship_images.append((img_path, lbl_path))
        else:
            # 라벨이 없거나 비어있으면 빈 바다
            empty_images.append((img_path, lbl_path))

    num_ships = len(ship_images)
    num_empty = len(empty_images)
    
    print(f"   🚢 Ships (Positive): {num_ships} 장")
    print(f"   🌊 Empty (Negative): {num_empty} 장")

    if num_ships == 0:
        print("⚠️ 경고: 배가 있는 이미지가 0장입니다. 경로를 확인하세요!")
        return

    # 2. 샘플링 전략 (1:2 비율)
    target_empty_count = num_ships * 2
    if target_empty_count > num_empty:
        target_empty_count = num_empty

    selected_empty = random.sample(empty_images, target_empty_count)
    
    # 최종 리스트 합치기
    final_set = ship_images + selected_empty
    random.shuffle(final_set)

    print(f"🎯 [Plan] 최종 데이터: {len(final_set)} 장 (Ship {num_ships} + Empty {len(selected_empty)})")
    print(f"🚀 [Action] 복사 시작...", flush=True)

    # 3. 복사 실행
    for img_src, lbl_src in tqdm(final_set, desc="Copying"):
        # 이미지 복사
        shutil.copy2(img_src, dst_train_img / img_src.name)
        
        # 라벨 처리
        if lbl_src.exists():
            # 원본 라벨이 있으면 복사
            shutil.copy2(lbl_src, dst_train_lbl / lbl_src.name)
        else:
            # [중요] 원본 라벨이 없으면(빈 바다), 빈 텍스트 파일을 생성해줘야 함!
            # YOLO는 이미지만 있고 라벨 파일이 없으면 에러가 날 수 있음.
            with open(dst_train_lbl / lbl_src.name, 'w') as f:
                pass # 빈 파일 생성

    # 4. Validation 셋 복사
    print("🔍 [Phase 2] Validation Set 복사 (Full)...")
    src_val_img = src_path / "images" / "val"
    src_val_lbl = src_path / "labels" / "val"
    
    dst_val_img = dest_path / "images" / "val"
    dst_val_lbl = dest_path / "labels" / "val"
    
    if src_val_img.exists() and not dst_val_img.exists():
        shutil.copytree(src_val_img, dst_val_img)
    if src_val_lbl.exists() and not dst_val_lbl.exists():
        shutil.copytree(src_val_lbl, dst_val_lbl)

    print("✨ 모든 작업 완료!")

if __name__ == "__main__":
    create_smart_subset_v2()