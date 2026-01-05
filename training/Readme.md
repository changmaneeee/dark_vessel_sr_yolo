# 🚀 SR-YOLO Training Pipeline

## 📋 개요

**Dark Vessel SR-YOLO** 프로젝트의 학습 파이프라인

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        학습 전략 요약                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  SR (MambaSR/RFDN)  │  개별 학습 (git clone)      │  ✅ 완료 가정          │
│  YOLO               │  개별 학습 (ultralytics)    │  ✅ 완료 가정          │
│  Arch 0, 2, 4       │  학습 불필요               │  개별 가중치 조합       │
│  Arch 5B            │  Fusion 모듈 학습 필요      │  ⭐ 이 스크립트 사용   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 아키텍처별 학습 필요성

| Architecture | 구조 | End-to-End 학습 필요? | 이유 |
|--------------|------|----------------------|------|
| **Arch0** | LR → SR → YOLO | ❌ NO | SR, YOLO 완전 분리 |
| **Arch2** | Gate → SR/Bypass → YOLO | ⚠️ Gate만 | Gate만 feedback 필요 |
| **Arch4** | 2-pass (저해상도 먼저) | ❌ NO | Inference 로직일 뿐 |
| **Arch5B** | SR + YOLO → **Fusion** | ✅ **YES** | Fusion이 두 feature 연결 |

**결론**: **Arch5B만 별도 학습 필요**, 나머지는 개별 학습된 가중치 조합으로 Inference

---

## 📁 파일 구조

```
training/
├── train_arch5b.py    # 메인 학습 스크립트 (Arch5B 전용)
├── dataset.py         # DataLoader (HR/LR + YOLO label)
├── __init__.py        # 모듈 초기화
└── README.md          # 이 문서
```

---

## 📂 데이터셋 구조

```
smart_airbus_data/
├── hr/                          # HR 데이터셋 루트
│   ├── data.yaml               # YOLO용 (우리 코드에서 불필요)
│   ├── images/
│   │   ├── train/              # 학습 이미지
│   │   │   ├── img_001.jpg
│   │   │   └── ...
│   │   └── val/                # 검증 이미지
│   └── labels/
│       ├── train/              # YOLO format labels
│       │   ├── img_001.txt     # class x_center y_center w h
│       │   └── ...
│       └── val/
└── lr/                          # LR 데이터셋 루트
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

### YOLO Label 형식 (`.txt`)
```
# class x_center y_center width height (normalized 0-1)
0 0.5 0.5 0.1 0.2
0 0.3 0.7 0.05 0.08
```

---

## 🎯 Arch5B 학습 모드

### Mode 1: Scratch (처음부터 학습)

| 항목 | 설정 |
|------|------|
| **SR 가중치** | 기본 (ImageNet pretrained 또는 없음) |
| **YOLO 가중치** | COCO pretrained (`yolov8n.pt`) |
| **학습 대상** | SR 모델 + Fusion 모듈 |

```bash
python training/train_arch5b.py \
    --mode scratch \
    --sr_type mamba \
    --hr_root /path/to/smart_airbus_data/hr \
    --lr_root /path/to/smart_airbus_data/lr \
    --epochs 100 \
    --batch_size 2
```

### Mode 2: Finetune (선박 특화 가중치)

| 항목 | 설정 |
|------|------|
| **SR 가중치** | 선박 데이터로 학습된 가중치 |
| **YOLO 가중치** | 선박 데이터로 학습된 가중치 |
| **학습 대상** | Fusion 모듈만 |

```bash
python training/train_arch5b.py \
    --mode finetune \
    --sr_type mamba \
    --sr_weights /path/to/mamba_ship.pth \
    --yolo_weights /path/to/yolo_ship.pt \
    --hr_root /path/to/smart_airbus_data/hr \
    --lr_root /path/to/smart_airbus_data/lr \
    --epochs 50 \
    --batch_size 2
```

---

## 📊 CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--mode` | 학습 모드: `scratch` / `finetune` | `scratch` |
| `--sr_type` | SR 모델: `rfdn` / `mamba` | `mamba` |
| `--sr_weights` | SR 가중치 경로 (finetune용) | `None` |
| `--yolo_weights` | YOLO 가중치 경로 | `yolov8n.pt` |
| `--num_classes` | 클래스 수 | `1` |
| `--hr_root` | HR 데이터셋 루트 경로 | **필수** |
| `--lr_root` | LR 데이터셋 루트 경로 | **필수** |
| `--hr_size` | HR 이미지 크기 | `640` |
| `--lr_size` | LR 이미지 크기 | `160` |
| `--batch_size` | 배치 크기 | `8` |
| `--epochs` | 에폭 수 | `100` |
| `--lr` | Learning rate | `1e-4` |
| `--workers` | DataLoader workers | `4` |
| `--exp_name` | 실험 이름 | `arch5b_{sr_type}_{mode}` |
| `--log_dir` | 로그 디렉토리 | `./logs` |
| `--ckpt_dir` | 체크포인트 디렉토리 | `./checkpoints` |
| `--device` | 디바이스 | `cuda` |
| `--no_amp` | Mixed Precision 비활성화 | `False` |

---

## 💾 가중치 저장 구조

### 저장 경로

```
checkpoints/
└── {exp_name}/                    # 예: arch5b_mamba_scratch
    ├── best.pt                   # ⭐ 최고 성능 모델
    ├── last.pt                   # 마지막 에폭
    ├── epoch_010.pt              # 10 에폭마다 저장
    ├── epoch_020.pt
    └── ...
```

### 체크포인트 내용 (`.pt` 파일)

```python
checkpoint = {
    'epoch': 50,                           # 현재 에폭
    'model_state_dict': {...},             # ⭐ 전체 모델 가중치
    'optimizer_state_dict': {...},         # Optimizer 상태
    'scheduler_state_dict': {...},         # Scheduler 상태
    'metrics': {                           # 메트릭
        'loss': 0.1234,
        'psnr': 28.5
    },
    'args': {...}                          # 학습 설정
}
```

### ⭐ Arch5B model_state_dict 구조

```python
model_state_dict = {
    # 1. SR 모델 (MambaSR 또는 RFDN)
    'sr_model.model.conv_first.weight': tensor(...),
    'sr_model.model.conv_first.bias': tensor(...),
    'sr_model.model.layers.0.xxx': tensor(...),
    ...
    
    # 2. Fusion 모듈 (MultiScaleAttentionFusion)
    'fusion.p3_sr_proj.weight': tensor(...),
    'fusion.p3_sr_proj.bias': tensor(...),
    'fusion.p3_yolo_proj.weight': tensor(...),
    'fusion.p3_attention.xxx': tensor(...),
    'fusion.p4_xxx': tensor(...),
    'fusion.p5_xxx': tensor(...),
    ...
    
    # 3. YOLO Detector (frozen이지만 포함됨)
    'detector.detection_model.model.0.conv.weight': tensor(...),
    'detector.detection_model.model.0.bn.weight': tensor(...),
    ...
}
```

### 가중치 분리 로드 예시

```python
import torch

# 전체 체크포인트 로드
ckpt = torch.load('checkpoints/arch5b_mamba_scratch/best.pt')
state_dict = ckpt['model_state_dict']

# SR 모델만 추출
sr_weights = {k.replace('sr_model.', ''): v 
              for k, v in state_dict.items() if k.startswith('sr_model.')}

# Fusion 모듈만 추출
fusion_weights = {k.replace('fusion.', ''): v 
                  for k, v in state_dict.items() if k.startswith('fusion.')}

# YOLO만 추출
yolo_weights = {k.replace('detector.', ''): v 
                for k, v in state_dict.items() if k.startswith('detector.')}
```

---

## 📈 로그 및 모니터링

### Tensorboard

```bash
tensorboard --logdir logs/
```

### 로그 구조

```
logs/
└── {exp_name}/
    └── events.out.tfevents.xxx    # Tensorboard 로그
```

### 기록되는 메트릭

| 메트릭 | 설명 |
|--------|------|
| `train/loss` | 학습 Total Loss |
| `val/loss` | 검증 Loss |
| `val/psnr` | SR 이미지 PSNR (dB) |

---

## 💡 권장 학습 순서

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: SR 개별 학습                                                       │
│  ───────────────────                                                        │
│  git clone MambaIR → train → mamba_ship.pth                                │
│  또는 RFDN train → rfdn_ship.pth                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Step 2: YOLO 개별 학습                                                     │
│  ────────────────────                                                       │
│  yolo train data=hr/data.yaml → yolo_ship.pt                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Step 3: Arch5B Fusion 학습                                                 │
│  ──────────────────────────                                                 │
│  Option A: Mode scratch (기본 가중치로 처음부터)                             │
│  Option B: Mode finetune (선박 특화 가중치로) ← 권장                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚠️ 문제 해결

### CUDA Out of Memory

```bash
# 배치 크기 줄이기
python train_arch5b.py ... --batch_size 1

# 이미지 크기 줄이기
python train_arch5b.py ... --hr_size 512 --lr_size 128
```

### GPU 메모리 권장 설정

| GPU | 권장 Batch Size | 권장 HR Size |
|-----|----------------|--------------|
| RTX 4090 (24GB) | 2 | 640 |
| RTX 3090 (24GB) | 2 | 640 |
| RTX 3080 (10GB) | 1 | 512 |

### Dataset 0 images 에러

```bash
# 경로 확인 - 루트만 지정!
--hr_root /path/to/hr    # ✅
--hr_root /path/to/hr/images/train  # ❌
```

### data.yaml 필요?

| 상황 | 필요 여부 |
|------|----------|
| YOLO 개별 학습 (`yolo train`) | ✅ 필요 |
| Arch5B Fusion 학습 (이 스크립트) | ❌ **불필요** |

---

## 📦 설치 위치

```
dark_vessel_sr_yolo/
├── src/
│   ├── models/
│   │   └── pipelines/
│   │       └── arch5b_fusion.py
│   └── losses/
│       └── detection_loss.py   # hyp 수정된 버전 필요!
├── training/                   # ← 여기에 복사!
│   ├── train_arch5b.py
│   ├── dataset.py
│   └── __init__.py
├── checkpoints/                # 가중치 저장됨
└── logs/                       # Tensorboard 로그
```

---

## 🔄 학습 재개 (Resume)

현재 스크립트에는 resume 기능이 없음. 필요시 추가 가능:

```python
# 체크포인트에서 로드
ckpt = torch.load('checkpoints/exp/last.pt')
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
start_epoch = ckpt['epoch'] + 1
```

---

## ✅ 체크리스트

- [ ] `detection_loss.py` 수정 버전인지 확인 (hyp에 box/cls/dfl 추가)
- [ ] 데이터셋 경로 확인 (`images/`, `labels/` 폴더)
- [ ] GPU 메모리 확인 (batch_size 조정)
- [ ] SR 개별 학습 완료 (finetune 모드용)
- [ ] YOLO 개별 학습 완료 (finetune 모드용)
