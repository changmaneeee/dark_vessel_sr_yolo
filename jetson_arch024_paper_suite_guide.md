# Jetson Arch0/Arch2/Arch4 Paper Suite Guide

이 가이드는 Jetson에서 **Arch0 / Arch2 / Arch4** 를 한 번에 돌리면서,
- latency / fps
- tegrastats 기반 평균 전력 / 최대 전력
- Joule per image
- (가능한 경우) precision / recall / mAP / direct recall
를 함께 남기기 위한 실행 가이드입니다.

## 제공 파일
- `measure_jetson_job.sh`: 임의의 명령을 tegrastats로 감싸서 전력 로그와 summary 생성
- `jetson_job_summary.py`: 결과 JSON + tegrastats 로그를 합쳐 summary JSON 생성
- `run_jetson_arch024_paper_suite.sh`: Arch0/2/4를 한 번에 실행하는 논문용 suite

## 권장 사용법

```bash
cp /mnt/data/measure_jetson_job.sh .
cp /mnt/data/jetson_job_summary.py .
cp /mnt/data/run_jetson_arch024_paper_suite.sh .
chmod +x measure_jetson_job.sh run_jetson_arch024_paper_suite.sh
```

### 기본 실행 예시
```bash
RUN_TAG=jetson_paper_suite_a11 \
PROJECT_ROOT=$PWD \
SCOUT_WEIGHTS=/home/octolab/dark_vessel_sr_yolo/models/yolo8s_lr/best.pt \
SNIPER_WEIGHTS=/home/octolab/dark_vessel_sr_yolo/models/yolo8s_hr/best.pt \
SR_SNIPER_WEIGHTS=/home/octolab/dark_vessel_sr_yolo/models/yolo8s_sr_domain/best.pt \
SR_WEIGHTS=/home/octolab/dark_vessel_sr_yolo/models/rfdn/model_best.pt \
GATE_WEIGHTS=/home/octolab/dark_vessel_sr_yolo/training/gate_arch2/checkpoints/gate_gt/gate_best.pt \
ARCH4_CONFIG=$PWD/configs/experiment/arch4_run037_like_deploy.yaml \
MAX_IMAGES=200 \
DEVICE=cuda \
USE_JETSON_CLOCKS=1 \
NVP_MODE_ID=0 \
bash run_jetson_arch024_paper_suite.sh
```

## 핵심 출력
모든 결과는 아래에 저장됩니다.

```text
jetson_runs/<RUN_TAG>/
```

중요 파일:
- `00_system_snapshot.txt`: Jetson 환경 / nvpmodel / jetson_clocks 상태
- `suite_summary.tsv`: 논문 표에 바로 옮기기 쉬운 요약 표
- `logs/*.summary.json`: 각 arch별 상세 summary
- `logs/*.tegrastats.log`: 원본 전력 로그
- `results/*.json`: 각 arch의 원본 결과 JSON

## Joule/image 해석
스크립트는 tegrastats에서 `VDD_IN` 또는 `POM_5V_IN` 같은 총 입력 전력 값을 우선적으로 사용해

```text
energy_per_image_j = avg_power_watt * avg_ms_per_image / 1000
```

로 계산합니다.

## 주의
1. `arch0_bench_jetson.py`, `arch2_bench_jetson.py` 의 인자명이 repo와 다르면, `run_jetson_arch024_paper_suite.sh` 내부 `run_arch0()`, `run_arch2()` 명령 블록만 수정하세요.
2. Arch4는 현재 best deploy config를 base로 삼고, 실행 전에 `weights_lr`, `weights_hr`만 patched config로 덮어씁니다.
3. 논문용 비교에서는 **Jetson 성능/전성비 표**와 **PC full-val 정확도 표**를 분리해서 쓰는 것이 안전합니다.
