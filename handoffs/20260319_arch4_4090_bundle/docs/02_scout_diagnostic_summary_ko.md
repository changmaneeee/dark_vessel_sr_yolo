# [2026-03-19-02] Scout Recall Diagnostic 결과 및 해석

## 요약
- 현재 Arch4 canonical config의 실제 `pass1_conf`는 `0.0075`였다.
- 이 기준에서 Scout recall은:
  - `IoU>=0.5`: **0.7280**
  - `IoU>=0.3`: **0.8051**
- 따라서 현재 판단은 **Scout가 큰 병목이지만, 완전 붕괴는 아니며 downstream 병목도 함께 존재한다**는 것이다.

## 왜 이 진단이 중요했나
- 이전에는 stage 통계에서 Scout box가 평균 `4.45/img`로 보였는데,
- 단순 standalone 진단을 `conf=0.1`로 돌리면 `0.91/img`가 나와 축이 안 맞았다.
- 원인을 확인해보니 current canonical config의 실제 `pass1_conf`가 `0.0075`였다.
- `conf=0.0075`로 다시 측정하자 avg scout boxes/img가 `4.46`으로 stage 통계와 맞게 정렬됐다.

## current config 기준 결과

### `scout_conf=0.0075`, `match_iou=0.5`
- 파일:
  - `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260319_scout_diagnostic/scout_recall_conf00075.json`
- 결과:
  - GT total `6418`
  - Scout found `4672`
  - Scout missed `1746`
  - recall `0.7280`
  - avg scout boxes/img `4.46`

### `scout_conf=0.0075`, `match_iou=0.3`
- 파일:
  - `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260319_scout_diagnostic/scout_recall_conf00075_miou030.json`
- 결과:
  - Scout found `5167`
  - Scout missed `1251`
  - recall `0.8051`

## 해석
- GT의 약 `27.2%`는 Scout가 IoU 0.5 기준으로 아예 커버하지 못한다.
- 동시에 IoU 0.3 완화 기준으로는 recall이 `0.8051`까지 올라간다.
- 즉 문제는 두 가지가 섞여 있다.
  1. **Scout가 아예 놓치는 GT**
  2. **Scout가 대충 근처는 보지만 localization quality가 부족한 GT**

## matched score 분포
- `<0.25`: `1223`
- `0.25~0.45`: `710`
- `0.45~0.60`: `920`
- `0.60~0.80`: `1678`
- `>=0.80`: `141`

의미:
- GT를 맞춘 Scout box 중에서도 `0.45` 아래가 많다.
- 즉 많은 GT가 uncertain 경로로 넘어가며, downstream이 이를 복원하지 못하면 최종 FN으로 이어진다.

## 전략 결론
- 이 결과는 사용자가 제안한 3가지 분기 중 **결과 B (`Scout recall 0.70~0.85`)** 에 해당한다.
- 따라서 다음 1주 우선순위는:
  1. Scout 개선 실험 준비
  2. Scout를 바꾼 뒤 current best Sniper(`interp_a03`)로 Arch4 full-val 재평가
  3. 그 결과를 보고 Sniper 재학습 필요 여부를 다시 판단
