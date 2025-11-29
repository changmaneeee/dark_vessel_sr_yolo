import os
import random
import shutil

# ================= 설정 =================
# 원본 검증 데이터 경로 (심볼릭 링크 된 경로)
SRC_HR = "/home/octolab-rtx4090/Desktop/changmin/cv_detact_ship/datas/airbus_dataset/images/val"
SRC_LR = "/home/octolab-rtx4090/Desktop/changmin/airbus_data/images/airbus_ships_lr_realistic_val"

# 새로 만들 미니 데이터 경로
DST_ROOT = "/home/octolab-rtx4090/Desktop/changmin/airbus_data/images/Airbus_Val_Mini"
DST_HR = os.path.join(DST_ROOT, "HR")
DST_LR = os.path.join(DST_ROOT, "LR_bicubic/X4")

SAMPLE_COUNT = 100  # 100장만 추출
# ========================================

def create_mini_dataset():
    if not os.path.exists(SRC_HR):
        print(f"❌ 원본 경로를 찾을 수 없습니다: {SRC_HR}")
        return

    # 폴더 생성
    os.makedirs(DST_HR, exist_ok=True)
    os.makedirs(DST_LR, exist_ok=True)

    # 이미지 리스트 로드
    all_files = [f for f in os.listdir(SRC_HR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"🔍 전체 검증 데이터: {len(all_files)}장")

    if len(all_files) < SAMPLE_COUNT:
        print("⚠️ 파일 수가 샘플 수보다 적습니다. 전체를 복사합니다.")
        selected_files = all_files
    else:
        selected_files = random.sample(all_files, SAMPLE_COUNT)
    
    print(f"🚀 {len(selected_files)}장 샘플링 및 복사 시작...")

    for f in selected_files:
        # HR 복사
        shutil.copy(os.path.join(SRC_HR, f), os.path.join(DST_HR, f))
        
        # LR 복사 (파일명 규칙 체크 필요, 여기선 이름이 같다고 가정)
        # 만약 LR 파일명에 x4가 붙어있다면 아래 줄 수정 필요: f"{os.path.splitext(f)[0]}x4{os.path.splitext(f)[1]}"
        lr_name = f 
        # 만약 LR 폴더에 파일이 없다면 x4를 붙여서 시도
        if not os.path.exists(os.path.join(SRC_LR, lr_name)):
             name, ext = os.path.splitext(f)
             lr_name = f"{name}x4{ext}"
        
        if os.path.exists(os.path.join(SRC_LR, lr_name)):
            shutil.copy(os.path.join(SRC_LR, lr_name), os.path.join(DST_LR, lr_name))
        else:
            print(f"⚠️ LR 짝을 못 찾음: {lr_name}")

    print("✅ Mini Validation Set 생성 완료!")
    print(f"   HR: {DST_HR}")
    print(f"   LR: {DST_LR}")

if __name__ == "__main__":
    create_mini_dataset()