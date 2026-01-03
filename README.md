# AIS-SAT-PIPELINE

**VLEO 위성 온보드 AI를 위한 SR-Detection Feature Fusion 기반 소형 선박 탐지 파이프라인**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)

## 연구 목적

본 프로젝트는 초저궤도(VLEO) 위성의 저해상도 이미지에서 소형 선박(15~30m)을 탐지하는 문제를 해결하기 위해 개발되었습니다.

### 핵심 과제
- **저해상도 문제**: 6m GSD(Ground Sample Distance) 위성 영상에서 소형 선박 탐지
- **온보드 처리**: Edge Device (Jetson Xavier NX) 실시간 처리 요구
- **성능 향상**: SR(Super-Resolution)과 Detection의 효과적인 결합

### 기술적 접근
- Feature-level fusion을 통한 SR과 Detection 통합
- Multi-scale feature 활용
- End-to-end 학습 최적화

## 아키텍처

### 4가지 실험 아키텍처

| Architecture | Description | Target | Status |
|--------------|-------------|--------|--------|
| **Arch 0** | Sequential (LR→SR→HR→YOLO) | Baseline | 🟡 구현 예정 |
| **Arch 2** | Soft Gate Fusion | 연산 효율성 | 🟡 구현 예정 |
| **Arch 4** | Confidence-Adaptive | FN 최소화 | 🟡 구현 예정 |
| **Arch 5-B** | Feature Fusion ⭐ | 최고 성능 | 🟡 구현 예정 |

### Arch 5-B: Feature Fusion (주력 아키텍처)

```
LR Image (192x192)
    │
    ├─────────────────┐
    ▼                 ▼
SR Encoder      Detection Backbone
    │                 │
    │    ┌────────────┤
    │    │  Multi-scale
    └────┤  Feature Fusion
         │  (Attention)
         └────────────┐
                      ▼
                 YOLO Head
                      │
                      ▼
              Detection Results
```

## 설치

### Requirements
- Python 3.10+
- CUDA 11.8+ (GPU 학습용)
- PyTorch 2.0+

### 환경 설정

```bash
# Conda 환경 생성
conda create -n ais-sat python=3.10
conda activate ais-sat

# 의존성 설치
pip install -r requirements.txt

# 패키지 설치 (개발 모드)
pip install -e .
```

## 프로젝트 구조

```
AIS-SAT-PIPELINE/
├── configs/              # 설정 파일
│   ├── default.yaml
│   ├── paths.yaml
│   └── experiment/       # 아키텍처별 설정
├── src/
│   ├── models/          # 모델 정의
│   │   ├── sr_models/   # SR 모델 (RFDN, Mamba-SR, TTST)
│   │   ├── detectors/   # Detection 모델 (YOLO)
│   │   ├── fusion/      # Fusion 모듈
│   │   └── pipelines/   # 전체 파이프라인
│   ├── data/            # 데이터 로딩 & 전처리
│   ├── losses/          # Loss 함수
│   └── utils/           # 유틸리티
├── scripts/             # 학습/평가 스크립트
├── tests/               # 유닛 테스트
└── docs/                # 문서
```

## 사용법

### 1. 데이터 준비

```bash
# RLE 마스크 → YOLO format 변환
python scripts/data_preparation/convert_rle_to_yolo.py \
    --csv data/raw/airbus/train_ship_segmentations_v2.csv \
    --images data/raw/airbus/train_v2 \
    --output data/processed/hr

# LR 데이터셋 생성 (degradation)
python scripts/data_preparation/create_lr_dataset.py \
    --hr_dir data/processed/hr \
    --lr_dir data/processed/lr \
    --scale 4 \
    --degradation bicubic

# Train/Val/Test split
python scripts/data_preparation/split_dataset.py \
    --data_dir data/processed \
    --split 0.7 0.15 0.15
```

### 2. 학습

```bash
# Arch 0: Sequential Baseline
python scripts/train.py --config configs/experiment/arch0_sequential.yaml

# Arch 2: Soft Gate
python scripts/train.py --config configs/experiment/arch2_softgate.yaml

# Arch 4: Confidence-Adaptive
python scripts/train.py --config configs/experiment/arch4_adaptive.yaml

# Arch 5-B: Feature Fusion (주력)
python scripts/train.py --config configs/experiment/arch5b_fusion.yaml
```

### 3. 평가

```bash
# 성능 평가
python scripts/evaluate.py \
    --config configs/experiment/arch5b_fusion.yaml \
    --checkpoint checkpoints/arch5b_best.pth \
    --data_dir data/processed/test

# 추론
python scripts/inference.py \
    --config configs/experiment/arch5b_fusion.yaml \
    --checkpoint checkpoints/arch5b_best.pth \
    --image path/to/test/image.png \
    --output results/inference
```

### 4. 선박 크기 분석

```bash
# 데이터셋 내 선박 크기 분포 분석
python scripts/analyze_ship_sizes.py \
    --labels_dir data/processed/hr/train/labels \
    --output analysis/ship_sizes.json
```

## 데이터셋

### Airbus Ship Detection Dataset
- **출처**: [Kaggle Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection)
- **해상도**: HR 기준 (768x768)
- **선박 크기**: 15m ~ 300m (소형~대형)
- **사용 방식**: Proxy dataset (VLEO 시뮬레이션용)

### LR Degradation
- **GSD**: 6m (VLEO 위성 시뮬레이션)
- **입력 크기**: 192x192 (scale=4)
- **Degradation**: Bicubic downsampling + noise (optional)

## 성능 목표

| Metric | Target | Hardware |
|--------|--------|----------|
| mAP@0.5 | 0.75+ | - |
| Recall | 0.80+ | - |
| PSNR | 28.0+ | - |
| Latency | <100ms | Jetson Xavier NX |
| Memory | <3GB | Jetson Xavier NX |

## 개발 로드맵

- [x] 프로젝트 구조 초기화
- [ ] 데이터 전처리 파이프라인
- [ ] SR 모델 구현 (RFDN, Mamba-SR, TTST)
- [ ] YOLO Wrapper 구현
- [ ] Arch 0: Sequential Pipeline
- [ ] Arch 2: Soft Gate
- [ ] Arch 4: Confidence-Adaptive
- [ ] Arch 5-B: Feature Fusion
- [ ] Loss 함수 구현
- [ ] 학습 스크립트
- [ ] 평가 메트릭
- [ ] Jetson Xavier NX 최적화
- [ ] 문서화 완성

## 문서

- [마스터 문서](docs/MASTER_DOCUMENT.md) - 전체 연구 계획 및 아키텍처 설계
- [작업 분배 가이드](docs/WORK_DISTRIBUTION_GUIDE.md) - 팀 협업 가이드

## 기여

프로젝트 기여는 환영합니다!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 라이센스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

## 인용

이 코드를 연구에 사용하는 경우 다음과 같이 인용해주세요:

```bibtex
@misc{ais-sat-pipeline,
  title={AIS-SAT-PIPELINE: SR-Detection Feature Fusion for VLEO Satellite Ship Detection},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/ais-sat-pipeline}
}
```

## 연락처

- 이메일: changmin3702@kau.kr
- 이슈: [GitHub Issues](https://github.com/yourusername/ais-sat-pipeline/issues)

---

**Built with ❤️ for VLEO Satellite AI Research**
