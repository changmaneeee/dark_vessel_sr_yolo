# 🔍 Arch 0, 2, 4, 5B Inference Pipeline

## 📋 개요

개별 학습된 SR + YOLO 가중치를 조합하여 inference 수행

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Arch0:  LR → SR → YOLO                    (순차, 항상 SR)                  │
│  Arch2:  LR → Gate → SR/Bypass → YOLO      (조건부 SR)                      │
│  Arch4:  LR → YOLO → [조건부 SR] → YOLO    (2-pass 적응형)                  │
│  Arch5B: LR → SR+YOLO Features → Fusion    (Feature 융합) ⭐               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 파일 구조

```
inference/
├── inference.py              # 메인 inference 스크립트 (4개 아키텍처)
├── compare_architectures.py  # 아키텍처 비교 실험
├── soft_gate.py              # Arch2용 Gate 모듈
├── __init__.py
└── README.md
```

---

## 🚀 사용법

### 1. 단일 아키텍처 Inference

#### Arch0 (Sequential)

```bash
python inference/inference.py \
    --arch arch0 \
    --sr_type mamba \
    --sr_weights /path/to/mamba_ship.pth \
    --yolo_weights /path/to/yolo_ship.pt \
    --input /path/to/images \
    --output ./results/arch0
```

#### Arch2 (Soft Gate)

```bash
python inference/inference.py \
    --arch arch2 \
    --sr_type mamba \
    --sr_weights /path/to/mamba_ship.pth \
    --yolo_weights /path/to/yolo_ship.pt \
    --gate_weights /path/to/gate.pth \
    --gate_threshold 0.5 \
    --input /path/to/images \
    --output ./results/arch2
```

#### Arch4 (Adaptive 2-Pass)

```bash
python inference/inference.py \
    --arch arch4 \
    --sr_type rfdn \
    --sr_weights /path/to/rfdn_ship.pth \
    --yolo_weights /path/to/yolo_ship.pt \
    --adaptive_threshold 0.5 \
    --input /path/to/images \
    --output ./results/arch4
```

#### ⭐ Arch5B (Feature Fusion)

```bash
python inference/inference.py \
    --arch arch5b \
    --arch5b_checkpoint /path/to/checkpoints/arch5b_mamba_scratch/best.pt \
    --input /path/to/images \
    --output ./results/arch5b
```

**Note**: Arch5B는 학습된 체크포인트(best.pt) 필요!

---

### 2. 아키텍처 비교 실험

```bash
python inference/compare_architectures.py \
    --sr_type mamba \
    --sr_weights /path/to/mamba_ship.pth \
    --yolo_weights /path/to/yolo_ship.pt \
    --hr_root /path/to/hr_dataset \
    --lr_root /path/to/lr_dataset \
    --split val \
    --output ./comparison_results
```

---

## 📊 CLI 옵션

### inference.py

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--arch` | 아키텍처 (arch0/arch2/arch4/arch5b) | **필수** |
| `--sr_type` | SR 모델 (rfdn/mamba) | mamba |
| `--sr_weights` | SR 가중치 경로 | None |
| `--yolo_weights` | YOLO 가중치 경로 | yolov8n.pt |
| `--gate_weights` | Gate 가중치 (Arch2) | None |
| `--arch5b_checkpoint` | Arch5B 체크포인트 (best.pt) | None |
| `--input` | 입력 이미지/폴더 | **필수** |
| `--output` | 출력 폴더 | ./inference_results |
| `--conf_threshold` | 검출 신뢰도 임계값 | 0.25 |
| `--gate_threshold` | Gate 임계값 (Arch2) | 0.5 |
| `--adaptive_threshold` | 적응형 임계값 (Arch4) | 0.5 |

### compare_architectures.py

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--hr_root` | HR 데이터셋 경로 | **필수** |
| `--lr_root` | LR 데이터셋 경로 | **필수** |
| `--split` | 데이터셋 분할 (train/val) | val |
| `--max_images` | 최대 이미지 수 | None (전체) |
| `--output` | 결과 저장 경로 | ./comparison_results |

---

## 📈 출력 결과

### inference.py 출력

```
inference_results/
├── image1_sr.jpg      # SR 처리된 이미지
├── image1_det.jpg     # 검출 결과 시각화
├── image2_sr.jpg
├── image2_det.jpg
└── results.json       # 전체 결과 요약
```

### compare_architectures.py 출력

```
comparison_results/
└── comparison_results.json

# JSON 내용:
{
  "arch0": {
    "precision": 0.85,
    "recall": 0.78,
    "f1": 0.81,
    "sr_ratio": 1.0,
    "avg_time_ms": 45.2
  },
  "arch2": {...},
  "arch4": {...}
}
```

### 비교 결과 예시

```
================================================================================
📊 Architecture Comparison Results
================================================================================

Metric               Arch0           Arch2           Arch4          
-----------------------------------------------------------------
Precision            0.8521          0.8234          0.8156         
Recall               0.7812          0.7956          0.8023         
F1 Score             0.8151          0.8093          0.8089         
SR Usage             100.0%          45.2%           32.1%          
Avg Time (ms)        45.2            28.7            35.4           
================================================================================

🏆 Best Performance:
  - Best F1: ARCH0 (0.8151)
  - Fastest: ARCH2 (28.7 ms)
  - Best Efficiency (F1/time): ARCH2
```

---

## 🎯 아키텍처별 특성

| 아키텍처 | SR 사용률 | 속도 | 정확도 | 특징 |
|----------|----------|------|--------|------|
| **Arch0** | 100% | 느림 | 높음 | 항상 최고 품질 |
| **Arch2** | ~50% | 빠름 | 중간 | Gate가 필요시에만 SR |
| **Arch4** | ~30% | 중간 | 중상 | 1차 검출 후 필요시 SR |
| **Arch5B** | 100% | 중간 | **최고** | Feature 융합 (학습 필요) |

---

## ⚠️ 필요 가중치

| 가중치 | 용도 | 학습 방법 |
|--------|------|----------|
| `mamba_ship.pth` | MambaSR | git clone MambaIR → train |
| `rfdn_ship.pth` | RFDN | 개별 학습 |
| `yolo_ship.pt` | YOLO 검출 | `yolo train data=data.yaml` |
| `gate.pth` | Arch2 Gate | 별도 학습 또는 기본값 |
| `best.pt` | **Arch5B 전용** | `train_arch5b.py`로 학습 |

---

## 💡 Gate 학습 (Arch2)

Gate는 별도로 학습하거나 기본값(EdgeBasedGate)을 사용할 수 있어:

### Option 1: 기본 Gate 사용 (학습 불필요)

```bash
# --gate_weights 없이 실행
python inference.py --arch arch2 ...
# EdgeBasedGate가 자동으로 사용됨
```

### Option 2: Gate 학습 (권장)

```python
# Gate 학습은 SR 성능을 기반으로 수행
# - SR 적용 시 mAP 향상 → label 1
# - SR 적용 시 mAP 동일/하락 → label 0
# 추후 train_gate.py 제공 예정
```

---

## 📦 설치 위치

```
dark_vessel_sr_yolo/
├── src/
│   ├── models/
│   │   ├── sr_models/
│   │   │   ├── mamba_sr.py
│   │   │   └── rfdn.py
│   │   └── gates/
│   │       └── soft_gate.py  ← inference/soft_gate.py 복사
│   └── ...
├── inference/                 ← 여기에 복사!
│   ├── inference.py
│   ├── compare_architectures.py
│   ├── soft_gate.py
│   └── README.md
└── ...
```

---

## 🔧 문제 해결

### Gate 모듈 import 에러

```bash
# soft_gate.py를 src/models/gates/에 복사
cp inference/soft_gate.py src/models/gates/soft_gate.py
```

### SR 가중치 로드 실패

```python
# 가중치 파일 구조 확인
import torch
ckpt = torch.load('mamba_ship.pth')
print(ckpt.keys())
# 'model_state_dict' 또는 'net_g' 키가 있어야 함
```