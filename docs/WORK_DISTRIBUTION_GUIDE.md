# Work Distribution Guide

## 작업 분배 가이드

이 문서는 팀원 간 작업을 효율적으로 분배하기 위한 가이드입니다.

## 모듈별 작업 분배

### 1. 데이터 파이프라인 (Data Pipeline)
**담당**: Data Engineer

**작업 항목**:
- [ ] RLE to YOLO 변환 (`scripts/data_preparation/convert_rle_to_yolo.py`)
- [ ] LR dataset 생성 (`scripts/data_preparation/create_lr_dataset.py`)
- [ ] Train/Val/Test split (`scripts/data_preparation/split_dataset.py`)
- [ ] Dataset 클래스 구현 (`src/data/dataset.py`)
- [ ] Transforms 구현 (`src/data/transforms.py`)
- [ ] Degradation pipeline (`src/data/degradation.py`)

**예상 소요 시간**: 2-3주

### 2. SR 모델 (Super-Resolution Models)
**담당**: SR Researcher

**작업 항목**:
- [ ] RFDN 구현 (`src/models/sr_models/rfdn.py`)
- [ ] Mamba-SR 구현 (`src/models/sr_models/mamba_sr.py`)
- [ ] TTST 구현 (`src/models/sr_models/ttst.py`)
- [ ] SR loss 함수 (`src/losses/sr_loss.py`)
- [ ] SR 평가 메트릭 (`src/utils/metrics.py` - PSNR/SSIM)

**예상 소요 시간**: 3-4주

### 3. Detection 모델 (Object Detection)
**담당**: Detection Engineer

**작업 항목**:
- [ ] YOLO wrapper 구현 (`src/models/detectors/yolo_wrapper.py`)
- [ ] Detection loss (`src/losses/detection_loss.py`)
- [ ] Detection 메트릭 (`src/utils/metrics.py` - mAP)
- [ ] Visualization (`src/utils/visualization.py`)

**예상 소요 시간**: 2-3주

### 4. Fusion & Pipelines
**담당**: Architecture Researcher

**작업 항목**:
- [ ] Attention fusion (`src/models/fusion/attention_fusion.py`)
- [ ] Gate network (`src/models/fusion/gate_network.py`)
- [ ] Arch0: Sequential (`src/models/pipelines/arch0_sequential.py`)
- [ ] Arch2: Soft Gate (`src/models/pipelines/arch2_softgate.py`)
- [ ] Arch4: Adaptive (`src/models/pipelines/arch4_adaptive.py`)
- [ ] Arch5B: Feature Fusion (`src/models/pipelines/arch5b_fusion.py`)

**예상 소요 시간**: 4-5주

### 5. Training & Evaluation
**담당**: ML Engineer

**작업 항목**:
- [ ] Training loop (`scripts/train.py`)
- [ ] Evaluation script (`scripts/evaluate.py`)
- [ ] Inference script (`scripts/inference.py`)
- [ ] Checkpoint manager (`src/utils/checkpoint.py`)
- [ ] Logger (`src/utils/logger.py`)
- [ ] Combined loss (`src/losses/combined_loss.py`)

**예상 소요 시간**: 3-4주

### 6. Testing & Documentation
**담당**: All team members

**작업 항목**:
- [ ] Unit tests (`tests/`)
- [ ] Integration tests
- [ ] Documentation
- [ ] README updates
- [ ] Code review

**예상 소요 시간**: Ongoing

## 우선순위

### High Priority (Sprint 1)
1. 데이터 파이프라인 완성
2. Arch0 (Sequential) 구현 - Baseline
3. 기본 학습 루프

### Medium Priority (Sprint 2)
4. SR 모델 구현
5. Arch5B (Feature Fusion) 구현
6. 평가 시스템

### Low Priority (Sprint 3)
7. Arch2, Arch4 구현
8. 최적화 및 profiling
9. TensorRT conversion

## 협업 규칙

### Git Workflow
- `main` 브랜치: 안정 버전
- `dev` 브랜치: 개발 버전
- Feature branches: `feature/모듈명`

### Code Review
- 모든 PR은 최소 1명의 리뷰 필요
- TODO 코멘트는 Issue로 전환

### Communication
- Daily standup (선택적)
- Weekly progress meeting
- Slack/Discord for quick questions

## 테스트 전략

### Unit Tests
- 각 모듈별 테스트 작성
- Coverage 목표: 80%+

### Integration Tests
- End-to-end pipeline 테스트
- 실제 데이터로 검증

## 문서화

### Docstrings
- Google style docstrings 사용
- 모든 public 함수/클래스 문서화

### README
- 각 모듈별 README
- Usage examples

## 리소스

### Computing
- GPU: NVIDIA RTX 3090 / A100
- Jetson Xavier NX (profiling용)

### Storage
- Dataset: ~100GB
- Checkpoints: ~50GB

### References
- Internal wiki
- Paper references
- Code examples
