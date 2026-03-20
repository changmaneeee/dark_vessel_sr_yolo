# Arch4 Representative Case Diagnosis

기준 산출물: `20260313_113641_arch4_case_debug`

## 000d42241
- GT: `[480.0, 583.0, 488.0, 588.0]`
- sr: TP=0, FP=1, FN=1, preds=1
  - pred0: score=0.2966, IoU=0.1456, box=[463.52, 581.86, 485.42, 589.77]
- bilinear: TP=0, FP=0, FN=1, preds=0
  - final box 없음
- hr_ref: TP=1, FP=0, FN=0, preds=1
  - pred0: score=0.2588, IoU=0.6401, box=[477.33, 582.72, 487.98, 588.56]
- 진단: `sr`는 ROI를 활성화하지만 GT 근처 작은 선박에서 박스 위치/크기가 흔들려 IoU 0.5를 넘지 못함. `hr_ref`는 같은 ROI에서 TP를 회복하므로 wiring 문제가 아니라 crop 품질/도메인 문제로 해석하는 것이 타당함.

## 025c5fdca
- GT: `[120.0, 690.0, 137.0, 700.0]`
- sr: TP=0, FP=1, FN=1, preds=1
  - pred0: score=0.3684, IoU=0.0379, box=[114.56, 669.69, 229.12, 708.79]
- bilinear: TP=0, FP=0, FN=1, preds=0
  - final box 없음
- hr_ref: TP=1, FP=0, FN=0, preds=1
  - pred0: score=0.2805, IoU=0.6499, box=[118.0, 692.02, 138.25, 699.45]
- 진단: `sr`는 ROI를 활성화하지만 GT 근처 작은 선박에서 박스 위치/크기가 흔들려 IoU 0.5를 넘지 못함. `hr_ref`는 같은 ROI에서 TP를 회복하므로 wiring 문제가 아니라 crop 품질/도메인 문제로 해석하는 것이 타당함.

## 016ec07ac
- GT: `[197.0, 681.0, 625.0, 745.0]`
- sr: TP=1, FP=2, FN=0, preds=3
  - pred0: score=0.4065, IoU=0.8524, box=[182.22, 675.12, 626.59, 747.44]
  - pred1: score=0.2805, IoU=0.0050, box=[202.05, 679.87, 219.4, 688.93]
  - pred2: score=0.2644, IoU=0.0063, box=[608.75, 734.33, 625.75, 746.52]
- bilinear: TP=1, FP=0, FN=0, preds=1
  - pred0: score=0.2805, IoU=0.7895, box=[202.0, 678.0, 623.0, 758.0]
- hr_ref: TP=1, FP=2, FN=0, preds=3
  - pred0: score=0.3953, IoU=0.0060, box=[605.69, 736.16, 625.25, 744.69]
  - pred1: score=0.3948, IoU=0.9150, box=[198.13, 677.49, 623.06, 746.97]
  - pred2: score=0.3857, IoU=0.0037, box=[201.56, 681.56, 218.5, 687.57]
- 진단: `sr`는 큰 선박 본체는 맞추지만 작은 조각 박스를 추가로 생성해 FP가 늘어남. `bilinear`는 fallback만 남아 깔끔하게 끝나므로 이 케이스는 over-detection 또는 fragment FP 패턴에 가깝다.
