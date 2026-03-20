# [2026-03-19-03] Scout 재학습 v2 시작

## 무엇을 시작했는가
- current Arch4 진단 결과를 바탕으로 LR Scout 재학습을 시작했다.
- 목표는 current Scout의 recall 병목을 개선하는 것이다.

## 왜 시작했는가
- current canonical 기준 Scout recall 결과:
  - `scout_conf=0.0075`, `IoU>=0.5`에서 recall `0.7280`
  - `IoU>=0.3`에서 recall `0.8051`
- 즉 Scout는 완전히 붕괴된 수준은 아니지만, GT의 약 27%를 IoU 0.5 기준으로 놓치고 있다.
- 따라서 결과 분류상 **결과 B (Scout recall 0.70~0.85)** 에 해당하며, Scout 개선 실험 가치가 충분하다.

## 이번 재학습 설정
- base weights:
  - `/home/changmin/dark_vessel_sr_yolo/weights/yolo_lr/8s/best.pt`
- data:
  - `/home/changmin/smart_airbus_data_lr/data.yaml`
- train script:
  - `/home/changmin/dark_vessel_sr_yolo/iac_jetson/train_scout_yolo.py`
- launch script:
  - `/home/changmin/dark_vessel_sr_yolo/iac_runs/run_scout_retrain_v2.sh`

## 핵심 하이퍼파라미터
- `epochs=100`
- `imgsz=640`
- `batch=16`
- `optimizer=AdamW`
- `lr0=0.0005`
- `warmup_epochs=5`
- augmentation:
  - `mosaic=1.0`
  - `mixup=0.15`
  - `copy_paste=0.10`

## 안정성 설정
- `workers=0`
- `amp=false`

이유:
- 이전 WSL/CUDA 환경에서 dataloader/AMP 관련 불안정 이력이 있어, 이번 Scout 재학습은 속도보다 안정성을 우선했다.

## 다음 단계
1. 학습 완료 후 best.pt 확보
2. 재학습 Scout + current best Sniper(`interp_a03`)로 Arch4 full-val direct 재평가
3. Scout 개선이 실제 Arch4 F1을 얼마나 올리는지 확인
