# 4090 PC에서 Arch4 번들 직접 실행하는 방법

이 문서는 **다른 4090 Linux PC** 에서, 현재 준비된 Arch4 번들과 데이터셋을 직접 복사해서 실행하는 절차를 처음부터 끝까지 설명한다.

---

## 0. 무엇을 옮겨야 하나

### 반드시 복사할 것 3개
1. 번들 폴더
   - `20260319_arch4_4090_bundle`
2. LR 데이터
   - `smart_airbus_data_lr`
3. HR 데이터
   - `smart_airbus_data`

권장:
- `arch4_sniper_crops`는 가능하면 복사하지 말고 **4090에서 다시 생성**한다.
- 외장하드 대용량 파일 복사 중 에러가 많다면 이 방법이 더 안전하다.

즉, 다른 PC에는 최소 아래 3개가 있으면 된다.

```text
<어딘가>/20260319_arch4_4090_bundle
<어딘가>/smart_airbus_data_lr
<어딘가>/smart_airbus_data
```

---

## 1. 가장 쉬운 방법: 경로를 최대한 원본과 비슷하게 맞추기

가능하면 다른 PC에서도 아래처럼 두는 게 가장 편하다.

```text
/home/<USER>/dark_vessel_sr_yolo
/home/<USER>/smart_airbus_data_lr
/home/<USER>/smart_airbus_data
/home/<USER>/dark_vessel_sr_yolo/data/arch4_sniper_crops
```

추천 배치 예시:

```bash
mkdir -p /home/$USER/dark_vessel_sr_yolo
cp -r smart_airbus_data_lr /home/$USER/
cp -r smart_airbus_data /home/$USER/
mkdir -p /home/$USER/dark_vessel_sr_yolo/data
```

그 다음 **번들 복원 스크립트**를 실행한다.

```bash
bash /path/to/20260319_arch4_4090_bundle/RESTORE_BUNDLE_LAYOUT.sh \
  /path/to/20260319_arch4_4090_bundle \
  /home/$USER/dark_vessel_sr_yolo
```

이 스크립트가 아래를 자동으로 맞춰준다.
- `code/arch4_roi_awareNMS_ablation.py` -> `src/models/pipelines/arch4_roi_awareNMS_ablation.py`
- `code/arch4_wiring_check.py` -> `iac_jetson/arch4_wiring_check.py`
- `configs/*.yaml` -> `configs/experiment/*.yaml`
- `weights/*.pt` -> `weights/*.pt`

즉 **번들을 그냥 repo 루트에 풀기만 하면 경로가 꼬인다. 반드시 restore 스크립트로 repo 구조를 복원한 뒤 실행해야 한다.**

---

## 1-1. 권장 경로: crop dataset은 4090에서 다시 생성

`arch4_sniper_crops` 복사 중 에러가 많다면, 아래 스크립트로 4090에서 직접 다시 만드는 게 맞다.

```bash
bash /path/to/20260319_arch4_4090_bundle/REBUILD_SNIPER_CROPS_ON_4090.sh \
  /home/$USER/dark_vessel_sr_yolo \
  /home/$USER/smart_airbus_data_lr \
  /home/$USER/smart_airbus_data
```

이 스크립트가 하는 일:
- val raw pair 무결성 검사
- Arch4 실제 ROI-SR path로 train crop dump
- Arch4 실제 ROI-SR path로 val crop dump

즉 **기존 crop dataset을 복사해서 쓰는 대신, 4090에서 새로 `data/arch4_sniper_crops`를 만든다.**

이게 지금 상황에서는 가장 안전하다.

---

## 2. 환경 만들기

### 2-1. conda 환경 생성

```bash
conda create -n dark_vessel python=3.12 -y
conda activate dark_vessel
```

### 2-2. 패키지 설치

번들 안에 환경 참고 파일이 있다.

```text
env/pip_freeze.txt
env/dark_vessel_from_history.yml
```

가장 현실적인 방법:

```bash
cd /home/$USER/dark_vessel_sr_yolo
pip install -r env/pip_freeze.txt
```

만약 일부 패키지 충돌이 나면 최소한 아래가 필요하다.
- `torch`
- `torchvision`
- `ultralytics`
- `numpy`
- `Pillow`
- `PyYAML`
- `opencv-python`
- `matplotlib`
- `pandas`

추가로 Arch4 코드가 `mamba_ssm`를 요구하므로 이 패키지도 필요하다.

```bash
pip install mamba-ssm
```

주의:
- 4090 PC의 CUDA / torch 조합에 맞춰 torch를 먼저 설치한 뒤 나머지를 설치하는 편이 안전하다.

---

## 3. 경로가 바뀌는 경우 원칙

다른 PC에서는 주소가 달라질 수 있다.  
이때 제일 중요한 원칙은 하나다.

**쉘 스크립트보다 Python 스크립트를 직접 실행하면서 경로를 인자로 넘기는 방식이 가장 안전하다.**

왜냐하면:
- 번들은 전달용이라 `code/`, `configs/`, `weights/`로 납작하게 묶여 있다.
- 실제 코드는 `src/models/pipelines`, `iac_jetson`, `configs/experiment` 구조를 기대한다.
- 따라서 먼저 restore로 repo 구조를 맞춘 뒤, Python 스크립트에 경로를 직접 넘기는 게 가장 안전하다.

즉 4090 PC에서는:
- `.sh`를 바로 실행하는 것보다
- **Python 스크립트를 직접 실행**하는 걸 기본으로 추천한다.

---

## 4. 실행 전에 먼저 할 것: 이미지/라벨 검증 + valid pair 생성

가장 안전한 순서는 **유효한 pair를 먼저 만든 뒤** 그 목록을 기준으로 판단하는 것이다.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dark_vessel

PROJECT_ROOT=/home/$USER/dark_vessel_sr_yolo
LR_DIR=/home/$USER/smart_airbus_data_lr/images/val
HR_LABELS=/home/$USER/smart_airbus_data/labels/val
OUT_DIR=$PROJECT_ROOT/iac_runs/$(date +%Y%m%d)_validate_pairs_val
mkdir -p $OUT_DIR

python $PROJECT_ROOT/iac_jetson/validate_paired_dataset.py \
  --images_dir $LR_DIR \
  --labels_dir $HR_LABELS \
  --out_dir $OUT_DIR \
  --allow_empty_labels
```

이 스크립트가 만들어주는 핵심 파일:
- `valid_pairs.json`
- `valid_images.txt`
- `valid_labels.txt`
- `invalid_images.txt`
- `invalid_labels.txt`
- `missing_label_for_image.txt`

현재 원본 PC의 val 검증 결과:
- paired candidates: `6418`
- valid_pairs: `6418`
- invalid_pairs: `0`
- missing_label_for_image: `22466`  
  이건 label이 없는 background 이미지라 정상이다.

추가로, crop dataset까지 새로 만들 계획이면 이 검증은 `REBUILD_SNIPER_CROPS_ON_4090.sh` 안에서 val 기준으로 한 번 더 자동 수행된다.

---

## 4. 먼저 해야 할 것: 현재 best 재현 확인

가장 먼저 할 일은 **현재 best Arch4를 재현**하는 것이다.

현재 best:
- Scout: `weights/scout_yolo_lr_best.pt`
- Sniper: `weights/interp_a03.pt`
- SR: `weights/rfdn_arch4_model_best.pt`
- config: `configs/arch4_sizecond_interp_a03.yaml`

### 실행 명령

아래에서 경로만 자기 PC에 맞게 바꿔서 실행하면 된다.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dark_vessel

PROJECT_ROOT=/home/$USER/dark_vessel_sr_yolo
LR_DIR=/home/$USER/smart_airbus_data_lr/images/val
HR_DIR=/home/$USER/smart_airbus_data/images/val
HR_LABELS=/home/$USER/smart_airbus_data/labels/val
OUT_DIR=$PROJECT_ROOT/iac_runs/$(date +%Y%m%d)_arch4_interp_recheck
mkdir -p $OUT_DIR

python $PROJECT_ROOT/iac_jetson/arch4_wiring_check.py \
  --project_root $PROJECT_ROOT \
  --arch4_config $PROJECT_ROOT/configs/experiment/arch4_sizecond_interp_a03.yaml \
  --arch4_py $PROJECT_ROOT/src/models/pipelines/arch4_roi_awareNMS_ablation.py \
  --lr_images_dir $LR_DIR \
  --hr_images_dir $HR_DIR \
  --hr_labels_dir $HR_LABELS \
  --max_images 0 \
  --device cuda \
  --half \
  --modes sr \
  --sniper_imgsz_mode fixed \
  --sniper_imgsz_fixed 256 \
  --sr_weights $PROJECT_ROOT/weights/rfdn_arch4_model_best.pt \
  --yolo_weights_lr $PROJECT_ROOT/weights/scout_yolo_lr_best.pt \
  --yolo_weights_hr $PROJECT_ROOT/weights/interp_a03.pt \
  --out_json $OUT_DIR/arch4_interp_a03_direct_full6418.json
```

### 기대 결과

대략 아래와 비슷해야 한다.

- direct F1: `0.7238`
- mAP50: `0.7561`

결과가 크게 다르면:
- 환경 문제
- 경로 문제
- 패키지 버전 문제
중 하나다.

---

## 5. 두 번째: Scout 진단 재실행

Scout가 병목인지 다시 확인하려면:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dark_vessel

PROJECT_ROOT=/home/$USER/dark_vessel_sr_yolo
LR_DIR=/home/$USER/smart_airbus_data_lr/images/val
HR_LABELS=/home/$USER/smart_airbus_data/labels/val
OUT_DIR=$PROJECT_ROOT/iac_runs/$(date +%Y%m%d)_scout_diagnostic
mkdir -p $OUT_DIR

python $PROJECT_ROOT/iac_jetson/scout_recall_diagnostic.py \
  --project_root $PROJECT_ROOT \
  --scout_weights $PROJECT_ROOT/weights/scout_yolo_lr_best.pt \
  --lr_images_dir $LR_DIR \
  --hr_labels_dir $HR_LABELS \
  --upscale_factor 4.0 \
  --scout_conf 0.0075 \
  --match_iou 0.5 \
  --device cuda \
  --out_json $OUT_DIR/scout_recall_conf00075.json
```

현재 기준선:
- recall@IoU0.5 = `0.7280`

---

## 6. 세 번째: Scout 재학습 실행

4090 PC로 가져가는 가장 큰 이유가 이 단계다.

### 실행 명령

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dark_vessel

PROJECT_ROOT=/home/$USER/dark_vessel_sr_yolo
LR_DATA_YAML=/home/$USER/smart_airbus_data_lr/data.yaml

python $PROJECT_ROOT/iac_jetson/train_scout_yolo.py \
  --data $LR_DATA_YAML \
  --base_weights $PROJECT_ROOT/weights/scout_yolo_lr_best.pt \
  --imgsz 640 \
  --epochs 100 \
  --batch 16 \
  --patience 20 \
  --optimizer AdamW \
  --lr0 0.0005 \
  --lrf 0.01 \
  --warmup_epochs 5 \
  --mosaic 1.0 \
  --mixup 0.15 \
  --copy_paste 0.10 \
  --project $PROJECT_ROOT/weights/yolo_lr_improved \
  --name 8s_aug_v2_4090 \
  --device 0 \
  --workers 0 \
  --save_period 10 \
  --amp false
```

### 왜 이렇게 실행하나
- `workers=0`, `amp=false`는 4060 PC에서 안정성을 위해 쓴 값이다.
- 4090 PC에서도 처음 한 번은 **안정성 우선**으로 그대로 가는 게 낫다.
- 잘 돌면 그 다음에
  - `workers`
  - `batch`
  - `amp`
를 키워서 더 빠르게 돌리면 된다.

---

## 7. Scout 재학습 후 Arch4 재평가

Scout가 학습 끝나면, 새 Scout를 current best Sniper와 결합해 본다.

가정:
- 새 Scout best weights:
  - `/home/$USER/dark_vessel_sr_yolo/weights/yolo_lr_improved/8s_aug_v2_4090/weights/best.pt`

실행:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dark_vessel

PROJECT_ROOT=/home/$USER/dark_vessel_sr_yolo
LR_DIR=/home/$USER/smart_airbus_data_lr/images/val
HR_DIR=/home/$USER/smart_airbus_data/images/val
HR_LABELS=/home/$USER/smart_airbus_data/labels/val
OUT_DIR=$PROJECT_ROOT/iac_runs/$(date +%Y%m%d)_arch4_with_new_scout
mkdir -p $OUT_DIR

python $PROJECT_ROOT/code/arch4_wiring_check.py \
  --project_root $PROJECT_ROOT \
  --arch4_config $PROJECT_ROOT/configs/arch4_sizecond_interp_a03.yaml \
  --arch4_py $PROJECT_ROOT/code/arch4_roi_awareNMS_ablation.py \
  --lr_images_dir $LR_DIR \
  --hr_images_dir $HR_DIR \
  --hr_labels_dir $HR_LABELS \
  --max_images 0 \
  --device cuda \
  --half \
  --modes sr \
  --sniper_imgsz_mode fixed \
  --sniper_imgsz_fixed 256 \
  --sr_weights $PROJECT_ROOT/weights/rfdn_arch4_model_best.pt \
  --yolo_weights_lr /home/$USER/dark_vessel_sr_yolo/weights/yolo_lr_improved/8s_aug_v2_4090/weights/best.pt \
  --yolo_weights_hr $PROJECT_ROOT/weights/interp_a03.pt \
  --out_json $OUT_DIR/arch4_newscout_interp_a03_direct_full6418.json
```

이 결과가 현재 기준 `F1 0.7238`을 넘는지가 핵심이다.

---

## 8. 경로가 완전히 달라질 때 수정해야 할 것

직접 실행 방식이면 사실 크게 두 가지만 맞추면 된다.

1. `PROJECT_ROOT`
2. 데이터 경로
   - `smart_airbus_data_lr`
   - `smart_airbus_data`

config 파일 안에 weight 경로가 있어도, 위 명령처럼
- `--sr_weights`
- `--yolo_weights_lr`
- `--yolo_weights_hr`

로 덮어쓰면 대부분 해결된다.

즉 4090 PC에서는 **Python 스크립트 + 직접 인자 전달 방식**으로 가면 주소가 달라도 충분히 대응 가능하다.

---

## 9. 실행 순서 추천

처음부터 한 번에 정리하면:

1. 번들 + 데이터 3개 복사
2. conda env 생성
3. 패키지 설치
4. `interp_a03` 재현 확인
5. Scout diagnostic 재확인
6. Scout 재학습 실행
7. 새 Scout + `interp_a03`로 Arch4 full-val 평가

---

## 10. 가장 중요한 한 줄

**4090 PC에서는 먼저 current best를 재현한 뒤, Scout만 교체해서 Arch4가 실제로 얼마나 오르는지를 보는 것이 가장 빠르고 안전한 다음 단계다.**
