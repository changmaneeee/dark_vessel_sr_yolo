# AIS-SAT-PIPELINE Master Document

## 프로젝트 개요

### 연구 목표
VLEO(Very Low Earth Orbit) 위성의 저해상도 이미지에서 소형 선박(15~30m) 탐지를 위한 SR-Detection Feature Fusion 파이프라인 개발

### 핵심 과제
1. **저해상도 문제**: 6m GSD 위성 영상에서 소형 선박 탐지
2. **온보드 제약**: Jetson Xavier NX에서 실시간 처리
3. **성능 향상**: SR과 Detection의 효과적인 통합

## 아키텍처 설계

### Architecture 0: Sequential (Baseline)
- **구조**: LR → SR → HR → YOLO
- **장점**: 구현 단순, SR 품질 높음
- **단점**: Latency 높음, SR 오류 전파

### Architecture 2: Soft Gate
- **구조**: LR/SR feature를 soft gate로 혼합
- **장점**: 연산 효율성, adaptive feature selection
- **단점**: Gate 학습 필요

### Architecture 4: Confidence-Adaptive
- **구조**: 2-pass detection with adaptive SR
- **장점**: FN 최소화, adaptive latency
- **단점**: 2-pass 오버헤드

### Architecture 5-B: Feature Fusion ⭐ (주력)
- **구조**: Multi-scale feature fusion (SR encoder ↔ YOLO backbone)
- **장점**: 최고 성능, end-to-end 최적화
- **단점**: 복잡도 높음, 메모리 사용량 큼

## 데이터셋

### Airbus Ship Detection Dataset
- **출처**: Kaggle
- **용도**: Proxy dataset (VLEO 시뮬레이션)
- **처리**: RLE → YOLO format 변환

### LR Degradation
- **GSD**: 6m (scale=4 from 1.5m HR)
- **방법**: Bicubic / Realistic / Sensor simulation

## 성능 목표

| Metric | Target |
|--------|--------|
| mAP@0.5 | 0.75+ |
| Recall | 0.80+ |
| PSNR | 28.0+ |
| Latency | <100ms (Jetson Xavier NX) |
| Memory | <3GB |

## 구현 계획

### Phase 1: 기반 구축 ✅
- [x] 프로젝트 구조
- [x] Base classes
- [x] Config 시스템

### Phase 2: 데이터 파이프라인
- [ ] RLE to YOLO 변환
- [ ] LR degradation
- [ ] Dataset & DataLoader

### Phase 3: 모델 구현
- [ ] SR 모델 (RFDN, Mamba-SR, TTST)
- [ ] YOLO wrapper
- [ ] Fusion modules

### Phase 4: 파이프라인
- [ ] Arch 0: Sequential
- [ ] Arch 2: Soft Gate
- [ ] Arch 4: Adaptive
- [ ] Arch 5-B: Feature Fusion

### Phase 5: 학습 & 평가
- [ ] Training loop
- [ ] Metrics & Evaluation
- [ ] Visualization

### Phase 6: 최적화
- [ ] Jetson Xavier NX profiling
- [ ] TensorRT conversion
- [ ] Inference optimization

## 참고 자료

### 관련 논문
- RFDN: "Residual Feature Distillation Network for Lightweight Image Super-Resolution"
- YOLOv8: Ultralytics documentation
- Feature Fusion: 관련 논문 추가 예정

### 데이터셋
- [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection)
