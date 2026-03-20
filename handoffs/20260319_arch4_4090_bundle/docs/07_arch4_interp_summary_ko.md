# [2026-03-18-07] Arch4 Sniper Checkpoint Interpolation 평가 요약

## 1. 목적

`crop-ft` Sniper는 recall 우세,
`hard-negative ft` Sniper는 precision 우세였다.

그래서 두 checkpoint를 선형 보간해서 중간 지점을 만들고, Arch4 full-val 성능이 더 좋아지는지 확인했다.

## 2. direct 결과

| 모델 | Precision@0.5 | Recall@0.5 | F1@0.5 | ms/img |
| --- | ---: | ---: | ---: | ---: |
| crop-ft baseline | 0.6738 | 0.7398 | 0.7052 | 66.74 |
| hard-negative ft | 0.7291 | 0.7061 | 0.7174 | 70.43 |
| interp `a07` | 0.7353 | 0.7036 | 0.7191 | 61.60 |
| interp `a05` | 0.7640 | 0.6804 | 0.7198 | 60.97 |
| interp `a03` | 0.7645 | 0.6873 | 0.7238 | 59.16 |

핵심:
- interpolation은 실제로 먹혔다.
- 세 개 모두 hard-negative 단독보다 높았다.
- 최적은 `alpha=0.3`였다.

## 3. best alpha (`a03`) mAP

| 모델 | Precision | Recall | mAP50 | mAP50-95 | ms/img |
| --- | ---: | ---: | ---: | ---: | ---: |
| crop-ft baseline | 0.7417 | 0.6756 | 0.7462 | 0.5900 | 60.02 |
| hard-negative ft | 0.7486 | 0.6725 | 0.7496 | 0.5956 | 67.59 |
| interp `a03` | 0.7437 | 0.6904 | 0.7561 | 0.6026 | 51.01 |

핵심:
- `a03`가 mAP도 최고였다.
- direct와 mAP가 모두 동시에 좋아졌다.

## 4. 해석

1. `crop-ft`와 `hard-negative ft`는 보완 관계였다.
   - recall을 주는 쪽과 precision을 주는 쪽이 달랐다.

2. hard-negative 쪽 비중이 더 큰 `a03`가 최적이었다.
   - precision을 유지하면서 recall을 일부 회복했다.

3. 따라서 현재 Arch4의 가장 좋은 Sniper 후보는 interpolation 모델이다.

## 5. 현재 결론

현재 Arch4 새 canonical 후보:

- **Sniper = `interp_a03.pt`**
- direct F1@0.5 = **0.7238**
- mAP50 = **0.7561**

즉 단순 crop-ft나 hard-negative 단독보다, 둘을 보간한 checkpoint가 더 좋았다.
