# Arch4 전문가 전달 문서

## 목적
이 문서는 Arch4를 처음 보는 외부 전문가가 현재 구조, 이미 확인된 사실, 실패 패턴, 최근 개선 사항, 그리고 다음에 무엇을 토론해야 하는지를 빠르게 이해하도록 만들었다.

이 문서를 읽고 아래 번들 폴더의 코드와 결과 파일을 함께 보면, 바로 현 상황을 토의하고 다음 실험 방향을 제안할 수 있다.

---

## 1. 프로젝트 맥락

프로젝트 목표는 LR(low-resolution) 항공/위성 영상에서 ship detection을 수행할 때:

- 정확도를 유지하거나 높이고
- 온보드/Jetson 환경에서 latency, FPS, 전력 효율을 개선하는 것이다.

현재 실질적인 구조는 두 축이다.

1. `Arch2`
- gate가 높은 이미지에만 SR을 적용하는 selective-skip 구조
- 현재 가장 현실적인 실전/배포 후보

2. `Arch4`
- `Scout -> uncertain ROI crop -> refine -> Sniper -> merge`
- 전체 이미지를 SR하지 않고, 불확실한 ROI만 정밀하게 다루는 구조
- 지금은 성능 튜닝보다 구조 병목과 후속 개선 가능성 판단이 핵심

---

## 2. Arch4 구조 요약

Arch4의 현재 분석 대상 runtime은 `arch4_roi_awareNMS_ablation.py`이다.

큰 흐름은 아래와 같다.

1. LR 전체 이미지에 Scout detector 실행
2. Scout 출력 중 `score >= pass2_conf`는 confident로 유지
3. 나머지 uncertain box들을 ROI group으로 묶음
4. 각 ROI group에 대해 crop 생성
5. crop refinement mode를 적용
6. refined crop을 Sniper detector에 입력
7. Sniper 출력을 원본 LR 좌표계로 되돌림
8. 최종 global NMS 후 detection 산출

여기서 refinement mode는 다음 3개가 핵심이다.

- `sr`
  - LR ROI crop을 SR 모델로 복원해서 Sniper에 입력
- `bilinear`
  - LR ROI crop을 bilinear interpolation만 해서 Sniper에 입력
- `hr_ref`
  - paired HR 이미지에서 같은 ROI 위치를 잘라 Sniper에 입력
  - 배포용이 아니라 oracle reference 상한선

중요한 점:
- `sr`, `bilinear`, `hr_ref`는 **같은 Scout 결과, 같은 ROI 좌표, 같은 Sniper/merge 구조**에서
- **Sniper 앞단의 ROI 표현만 바꿔서 비교**하는 실험이다.

---

## 3. 이전에 확인한 핵심 사실

### 3-1. Arch4는 wiring이 죽은 구조가 아니다

이전 wiring check에서 다음이 확인되었다.

- `refined_crop_hash_same_ratio`가 낮음
- `pass2_hash_same_ratio`도 낮음

즉:
- crop mode switch는 실제로 refined crop tensor를 바꾸고 있음
- Sniper raw output도 실제로 달라짐

따라서 현재 병목은
- mode switch 미작동
- hr_ref 미전달
- Sniper 미실행

같은 wiring failure가 아니다.

### 3-2. 초기 Arch4의 문제는 `sr`가 `bilinear`보다도 약할 수 있다는 점이었다

초기 100장 local wiring check에서:

- `sr`: F1 `0.5872`
- `bilinear`: F1 `0.6230`
- `hr_ref`: F1 `0.6549`

즉 당시 구조는 `hr_ref > bilinear > sr` 였다.

이건 SR이 ROI crop 환경에서 충분히 도움이 되지 못하거나, 오히려 localization drift / fragment FP를 만든다는 신호였다.

---

## 4. 대표 실패 패턴

대표 케이스 3개를 이전에 뽑아 상세 디버깅했다.

### 패턴 A. 작은 선박에서 localization drift

예:
- `000d42241`
- `025c5fdca`

초기 상태:
- `sr`: TP 0, FP 1, FN 1
- `hr_ref`: TP 1, FP 0, FN 0

해석:
- Sniper가 아예 안 도는 것이 아니라
- `sr` refined crop에서 잘못된 위치/크기의 박스를 내서 IoU 0.5를 넘지 못하는 문제

### 패턴 B. 큰 선박에서 fragment FP

예:
- `016ec07ac`

초기 상태:
- `sr`: TP 1, FP 2
- `bilinear`: TP 1, FP 0

해석:
- `sr`가 선박 본체는 맞추지만, 추가 조각 박스를 만들어 FP를 유발

---

## 5. ROI 기반 RFDN 재학습을 왜 했는가

핵심 가설은 다음이었다.

- 기존 범용 RFDN은 전체 이미지 분포에서 학습됨
- 하지만 Arch4는 실제 추론 시 Scout가 잘라낸 ROI crop만 SR에 입력함
- 따라서 train domain과 inference domain이 어긋나 있음

그래서:
- Scout가 실제로 만드는 uncertain ROI crop 분포로 `(LR ROI, HR ROI)` pair dataset을 구축
- 그 dataset으로 RFDN을 재학습

이 dataset은 단순히 ship만 자른 것이 아니라,
- uncertain ROI group 기준으로 잘린 crop
- positive / negative ROI를 모두 포함

하는 Arch4 task-driven SR dataset이다.

---

## 6. ROI 기반 RFDN 재학습 결과

재학습 후 best checkpoint:

- `Best PSNR = 33.143 @ epoch 100`

이를 Arch4의 `sr` branch에 연결해 기존과 동일한 100장 wiring check를 다시 수행했다.

### post-train HR-domain Sniper 기준

- `sr`: F1 `0.5872 -> 0.6291`
- `bilinear`: `0.6230` 그대로
- `hr_ref`: `0.6549` 그대로

의미:

1. `sr` branch는 실제로 개선되었다.
2. 이제 `sr > bilinear`가 되었다.
3. 그러나 아직 `hr_ref`와 gap이 남아 있다.

즉:
- ROI 기반 SR 재학습은 효과가 있었다.
- Arch4는 더 이상 “SR이 bilinear보다 못한 구조”라고 단정할 수 없다.
- 하지만 아직 oracle HR 수준까진 못 갔다.

---

## 7. post-train 대표 케이스 재검증 결과

재학습 후 동일 3개 케이스를 다시 확인했다.

### `000d42241`, `025c5fdca`

변화:
- `sr`: `TP=0, FP=1, FN=1` -> `TP=0, FP=0, FN=1`

해석:
- 잘못된 위치의 오검출은 사라졌다.
- 하지만 여전히 실제 TP 회복까지는 못 갔다.
- 즉, `오검출 -> 보수적 무검출`로 바뀌었다.

### `016ec07ac`

변화:
- `sr`: `TP=1, FP=2, FN=0` 유지
- 다만 best IoU는 `0.8524 -> 0.8856`로 상승

해석:
- localization은 개선되었으나
- fragment FP는 아직 남아 있다.

종합:
- ROI-RFDN 재학습은 small-object drift를 완화하는 방향으로 작동
- 그러나 대표 실패 유형을 완전히 해결하지는 못함

---

## 8. Sniper detector domain mismatch 가설 검증

추가 가설:
- 문제가 SR 자체보다 Sniper detector domain mismatch일 수 있는가?

검증 방법:
- 같은 post-train SR 가중치 고정
- Sniper만 HR-domain detector vs SR-domain detector로 교체
- 동일한 100장 wiring check 수행

결과:

### HR-domain Sniper + post-train SR
- `sr`: F1 `0.6291`

### SR-domain Sniper + post-train SR
- `sr`: F1 `0.6218`

즉:
- SR-domain Sniper로 바꿔도 좋아지지 않았고 오히려 약간 나빠졌다.

더 중요한 점:
- `hr_ref`도 SR-domain Sniper에서 같이 나빠졌다.

해석:
- 단순한 detector domain mismatch 하나가 핵심 병목은 아니다.
- “HR detector를 SR detector로 바꾸면 해결된다”는 방향은 현재 근거가 약하다.

---

## 9. 현재 가장 강한 가설

지금까지의 결과를 합치면, 가장 강한 가설은 아래다.

### 가설: 병목은 `final decision / merge policy` 쪽이다

특히 의심되는 지점:
- `drop_uncertain_if_sniper_hits = True`

현재 구조는:
- Sniper가 조금이라도 유효 박스를 내면
- 해당 ROI의 uncertain Scout fallback을 버린다.

이게 작은 선박에서는:
- 약간 흔들린 Sniper 박스가 fallback보다 더 나쁜데도
- fallback을 버리면서 miss를 키울 수 있다.

큰 선박에서는:
- fragment FP가 추가되면 그대로 final 결과를 오염시킬 수 있다.

즉 현재 문제는 더 이상
- wiring failure
- SR-domain detector 부재

보다
- `Sniper 결과를 어떤 조건에서 최종 결과로 신뢰할 것인가`
의 문제일 가능성이 높다.

---

## 10. Arch2 / Arch0와의 위치 비교

### Arch2

현재 실전 기준선은 Arch2 `thr=0.5`다.

- F1 `0.7538`
- `17.0 ms/img`

Arch4 post-train `sr`는
- F1 `0.6291`
- `91.62 ms/img`

따라서 현재 시점에서:
- 배포/실전 후보는 여전히 Arch2가 우세하다.

### Arch0

Arch0 full-image SR baseline은 여전히 구조적으로 단순하고 안정적이다.

현재 Arch4는
- Arch0/Arch2를 이겼다고 보기 어렵고
- 다만 “ROI 기반 SR이 실제로 통한다”는 첫 증거를 확보한 단계

정도로 보는 것이 정확하다.

---

## 11. 지금까지 배제된 가설

아래 가설들은 현재 근거가 약하다.

1. Arch4는 wiring이 죽어 있다
2. `crop_refine_mode`가 실제로 반영되지 않는다
3. Sniper가 refine mode 차이를 못 본다
4. 단순히 Sniper detector를 SR-domain detector로 바꾸면 해결된다

---

## 12. 현재 판단

### 확실한 것
- Arch4는 포기할 구조는 아니다.
- ROI 기반 RFDN 재학습은 실제로 효과가 있었다.
- `sr > bilinear`를 만들어냈다.

### 아직 부족한 것
- Arch2와 경쟁할 수준은 아니다.
- 대표 실패 케이스가 완전히 해결되지 않았다.
- detector 교체만으로는 병목이 해결되지 않았다.

### 현재 가장 합리적인 다음 단계
- Arch4는 계속 볼 가치가 있다.
- 하지만 다음 실험은 **SR 재학습 확대**보다
- **merge policy / final decision rule**을 검증하는 방향이 더 타당하다.

예:
- `drop_uncertain_if_sniper_hits=False` 비교
- fallback과 Sniper를 score/IoU 조건부로 병합
- 작은 ROI와 큰 ROI에서 다른 replacement policy 사용

---

## 13. 외부 전문가에게 토론하고 싶은 핵심 질문

1. 현재 Arch4의 가장 유력한 병목이 `merge policy`라는 해석에 동의하는가?
2. 작은 선박 miss와 큰 선박 fragment FP를 동시에 줄이기 위한 가장 타당한 final decision rule은 무엇인가?
3. `drop_uncertain_if_sniper_hits=True`를 유지하는 것이 적절한가?
4. Sniper output을 fallback Scout보다 우선할 조건을 어떤 식으로 설계하는 것이 좋은가?
5. ROI 기반 SR 자체는 어느 정도 개선됐다고 볼 때, 다음 실험은 detector 학습보다 merge rule 쪽이 맞는가?

---

## 14. 번들 구성

이 폴더에는 아래 자료가 포함되어 있다.

- `code/`
  - Arch4 runtime, wiring check, case debug, Arch2 selective 코드
- `results/`
  - Arch4 baseline/post-train/Sniper-domain 비교 결과
  - Arch2 기준선 결과
  - Arch0 full eval 요약
- `artifacts/`
  - pre-train / post-train 대표 케이스 디버그 산출물

이 자료만으로도 외부 전문가가 현재 상태를 충분히 이해하고 다음 토론을 진행할 수 있다.
