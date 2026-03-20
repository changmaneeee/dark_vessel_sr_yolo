# Arch4 4090 PC 전달 번들

## 1. 목적
- 이 번들은 Arch4 개인연구를 **다른 4090 PC에서 바로 이어서** 진행할 수 있도록 만든 전달 패키지다.
- 현재 기준 핵심 목표는 두 가지다.
  1. **현재 best Arch4 (`interp_a03`) 재현**
  2. **Scout 개선 실험 가속**

## 2. 현재 기준선

### 현재 best Arch4
- Sniper weights: `weights/interp_a03.pt`
- direct 결과:
  - F1 `0.7238`
  - 파일: `results/arch4_interp_a03_direct_full6418.json`
- mAP 결과:
  - mAP50 `0.7561`
  - 파일: `results/arch4_interp_a03_eval_full6418.json`

### hard-negative 결과
- Sniper weights: `weights/sniper_hardneg_best.pt`
- direct 결과:
  - F1 `0.7174`
  - 파일: `results/arch4_hardneg_direct_full6418.json`

### Scout 진단 결과
- current config 기준 `pass1_conf=0.0075`
- Scout recall@IoU0.5 = `0.7280`
- Scout recall@IoU0.3 = `0.8051`
- 파일:
  - `results/scout_recall_conf00075.json`
  - `results/scout_recall_conf00075_miou030.json`

## 3. 이 번들에 포함된 것

### code/
- `arch4_roi_awareNMS_ablation.py`
- `arch4_wiring_check.py`
- `arch4_dump_sniper_crops.py`
- `scout_recall_diagnostic.py`
- `validate_paired_dataset.py`
- `train_scout_yolo.py`
- `train_sniper_crop_yolo.py`
- `build_sniper_hardneg_dataset.py`
- `mine_sniper_hard_negatives.py`
- `interpolate_sniper_checkpoints.py`
- `arch4_overnight_helper.py`
- `run_scout_retrain_v2.sh`
- `run_scout_diagnostic_day1.sh`
- `run_overnight_optimization.sh`

### configs/
- `arch4_sizecond_interp_a03.yaml`
- `arch4_sizecond_hardneg.yaml`
- `arch4_roi_awareNMS_deploy.yaml`

### weights/
- `interp_a03.pt`
- `scout_yolo_lr_best.pt`
- `rfdn_arch4_model_best.pt`
- `sniper_cropft_best.pt`
- `sniper_hardneg_best.pt`

### results/
- current best direct/mAP json
- hard-negative direct/mAP json
- Scout diagnostic json
- 실험 요약 markdown

### env/
- `dark_vessel_from_history.yml`
- `pip_freeze.txt`

### docs/
- Notion에 바로 붙여넣을 수 있는 한국어 요약 문서

## 4. 이 번들에 포함되지 않은 것
- 원본 데이터셋 전체
- 기존 crop dataset 전체(권장: 복사하지 말고 4090에서 재생성)
- 현재 4060 PC에서 진행 중인 Scout 재학습 산출물

즉, **아래 데이터는 별도로 복사해야 한다.**

## 5. 4090 PC에 추가로 반드시 복사할 데이터

### 필수 1: 원본 LR/HR 데이터
- `/home/changmin/smart_airbus_data_lr`
- `/home/changmin/smart_airbus_data`

이 두 폴더가 있어야:
- Scout 재학습
- Arch4 full-val 평가
- Scout recall 진단
을 그대로 재현할 수 있다.

### 선택 2: ROI crop dataset
- `/home/changmin/dark_vessel_sr_yolo/data/arch4_sniper_crops`

이 폴더는 있으면 편하지만, **복사 에러가 많으면 굳이 옮기지 말고 4090에서 새로 만들면 된다.**

권장:
- LR/HR raw dataset만 복사
- 번들 안의 `REBUILD_SNIPER_CROPS_ON_4090.sh`로 crop dataset 재생성

## 6. 4090 PC에서 권장 디렉토리 구조
- project root:
  - `/home/changmin/dark_vessel_sr_yolo`
- data:
  - `/home/changmin/smart_airbus_data_lr`
  - `/home/changmin/smart_airbus_data`

가능하면 **동일 경로로 맞추는 것**이 가장 안전하다.
경로를 바꾸면 yaml/script 안의 절대경로를 다시 패치해야 한다.

## 7. 권장 초기 세팅

### conda env
1. Python 3.12 계열 환경 생성
2. 이 번들의 `env/pip_freeze.txt`를 참고해 패키지 설치
3. 환경 이름은 기존과 맞춰 `dark_vessel` 권장

### 가장 먼저 확인할 것
1. `python code/scout_recall_diagnostic.py --help`
2. `python code/arch4_wiring_check.py --help`
3. `python code/train_scout_yolo.py --help`
4. 필요하면 `bash REBUILD_SNIPER_CROPS_ON_4090.sh ...`

## 8. 4090 PC에서 가장 먼저 할 일

### 옵션 A: 현재 best 재현 확인
1. `weights/interp_a03.pt`를 Sniper로 사용
2. `weights/scout_yolo_lr_best.pt`를 Scout로 사용
3. `configs/arch4_sizecond_interp_a03.yaml` 기준으로
4. `code/arch4_wiring_check.py`로 full-val direct 재평가

### 옵션 B: Scout 재학습 바로 이어가기
1. `code/train_scout_yolo.py`
2. `code/run_scout_retrain_v2.sh`
3. 현재 4060보다 훨씬 빠른 속도로 학습

## 9. 판단 포인트
- Scout를 개선했을 때:
  - Scout recall@0.5가 얼마나 올라가는지
  - Arch4 final F1이 `0.7238`을 넘는지
- 목표는 Scout 개선만으로 Arch4가 실제로 상승하는지 확인하는 것이다.

## 10. 참고
- 현재 기준으로는 Scout가 병목이지만, downstream도 함께 병목이다.
- 따라서 4090 PC에서는 **Scout 재학습 + current best Sniper 유지**가 가장 우선순위가 높다.
