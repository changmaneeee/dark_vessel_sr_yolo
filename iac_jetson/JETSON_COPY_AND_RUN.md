# Jetson quick start (iac_jetson)

이 문서는 **개발 PC에서 수정한 최소 파일만 Jetson으로 옮겨서**, Jetson에서 `iac_jetson/` 폴더 안의 스크립트로 Arch0/2/4 논문용 지표를 한 번에 뽑는 방법을 정리한 것입니다.

## A. Jetson으로 꼭 옮겨야 하는 최소 파일
Jetson에 repo가 이미 clone되어 있다고 가정하면, 아래만 덮어쓰면 됩니다.

### 1) 코드 파일
- `src/models/pipelines/arch4_roi_awareNMS.py`
- `src/models/detectors/yolo_wrapper.py`
- `iac_scripts/arch0_bench_jetson.py`
- `iac_scripts/arch2_bench_jetson.py`
- `iac_scripts/arch4_eval_ultralytics.py`

### 2) config 파일
- `configs/experiment/arch4_roi_awareNMS_deploy.yaml`
- `configs/experiment/arch0_sequential.yaml`
- `configs/experiment/arch2_softgate.yaml`

### 3) Jetson helper 폴더(`iac_jetson/`)
- `measure_jetson_job.sh`
- `jetson_job_summary.py`
- `run_jetson_arch024_suite_iac.sh`

## B. Jetson에서 경로가 바뀔 가능성이 큰 항목
아래는 **반드시 Jetson 실제 경로로 바꿔야 할 가능성이 큰 것들**입니다.

- `PROJECT_ROOT`
- `HR_DATA_YAML`
- `LR_DATA_YAML`
- `LR_IMAGES_DIR`
- `SR_WEIGHTS`
- `YOLO_SR_WEIGHTS`
- `YOLO_LR_WEIGHTS`
- `GATE_WEIGHTS`
- `ARCH4_BASE_CONFIG`

## C. 어떤 weight를 어디에 쓰나
### Arch0
- SR: `SR_WEIGHTS` (RFDN)
- Detector: `YOLO_SR_WEIGHTS`

### Arch2
- SR: `SR_WEIGHTS` (RFDN)
- Gate: `GATE_WEIGHTS`
- Detector: `YOLO_SR_WEIGHTS`

### Arch4
- Scout: `YOLO_LR_WEIGHTS`
- Sniper: `YOLO_SR_WEIGHTS`
- SR: `SR_WEIGHTS`
- Config base: `ARCH4_BASE_CONFIG` (deploy YAML 추천)

## D. 권장 레이아웃
예시:
- repo root: `/home/octolab/dark_vessel_sr_yolo`
- helper dir: `/home/octolab/dark_vessel_sr_yolo/iac_jetson`

## E. 실행 예시
```bash
RUN_TAG=jetson_suite_run1 \
PROJECT_ROOT=/home/octolab/dark_vessel_sr_yolo \
SR_WEIGHTS=/home/octolab/dark_vessel_sr_yolo/weights/rfdn/model_best.pt \
YOLO_SR_WEIGHTS=/home/octolab/dark_vessel_sr_yolo/weights/yolo_8s_rfdn/best.pt \
YOLO_LR_WEIGHTS=/home/octolab/dark_vessel_sr_yolo/models/yolo8s_lr/best.pt \
GATE_WEIGHTS=/home/octolab/dark_vessel_sr_yolo/training/gate_arch2/checkpoints/gate_gt/gate_best.pt \
ARCH4_BASE_CONFIG=/home/octolab/dark_vessel_sr_yolo/configs/experiment/arch4_roi_awareNMS_deploy.yaml \
HR_DATA_YAML=/home/octolab/dark_vessel_sr_yolo/dataset/smart_airbus_data/data.yaml \
LR_DATA_YAML=/home/octolab/dark_vessel_sr_yolo/dataset/smart_airbus_data_lr/data.yaml \
LR_IMAGES_DIR=/home/octolab/dark_vessel_sr_yolo/dataset/smart_airbus_data_lr/images/val \
USE_JETSON_CLOCKS=1 \
NVP_MODE_ID=0 \
MAX_IMAGES=200 \
bash /home/octolab/dark_vessel_sr_yolo/iac_jetson/run_jetson_arch024_suite_iac.sh
```

## F. 결과를 다시 공유할 때 꼭 필요한 파일
- `jetson_runs/<RUN_TAG>/suite_summary.tsv`
- `jetson_runs/<RUN_TAG>/00_system_snapshot.txt`
- `jetson_runs/<RUN_TAG>/logs/arch0.summary.json`
- `jetson_runs/<RUN_TAG>/logs/arch2.summary.json`
- `jetson_runs/<RUN_TAG>/logs/arch4.summary.json`
