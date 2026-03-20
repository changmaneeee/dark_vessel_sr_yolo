# Arch5 Codex Handoff

이 문서는 **다른 Codex가 Arch5 개발을 바로 시작할 수 있도록** 만든 상세 handoff 문서다.  
전제는 다음과 같다.

- Arch0~4는 이미 장기간 실험이 진행되었고, 핵심 병목과 숫자가 어느 정도 정리되어 있다.
- Arch0~4의 후속 검증과 Jetson 이관은 현재 메인 Codex/사용자가 계속 진행한다.
- **새 Codex의 임무는 Arch5를 설계·구현·최적화**하는 것이다.

이 문서는 단순 아이디어가 아니라, **현재 무엇이 사실이고 무엇이 이미 실패했는지**를 최대한 명확하게 전달하는 데 목적이 있다.

---

## 1. Arch5의 한 줄 목표

**Arch2의 실전성(높은 F1, 빠른 latency)과 Arch4의 selective ROI refinement 아이디어를 결합해서, full-image SR 계산량을 줄이면서도 Arch2 이상의 정확도를 노리는 새 구조를 만든다.**

조금 더 정확히 말하면:

- Arch2는 현재 가장 실전적이다.
- Arch4는 selective ROI-SR 방향성은 맞지만, 구조적 병목 때문에 아직 Arch2를 못 넘었다.
- 따라서 Arch5는 “Arch4를 더 미세 튜닝”이 아니라,
  **Arch2의 이미지-level gating과 Arch4의 ROI-level refinement를 계층적으로 결합**하는 방향으로 가야 한다.

---

## 2. 현재까지의 팩트 요약

### 2-1. 현재 아키텍처별 위치

#### Arch0
- full-image SR 후 detector
- full6418 direct 기준:
  - F1 `0.7971`
- full6418 mAP50:
  - `0.8903`
- 강점:
  - 높은 정확도
- 약점:
  - full-image SR 비용

참조:
- `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/arch0_direct_probe_full6418.json`
- `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/arch0_eval_full6418/arch0_eval_full6418_rfdnyolo_summary.json`

#### Arch2
- image-level gate + selective SR skip
- 현재 메인 실전 baseline
- full6418 direct 기준:
  - `thr=0.5`: F1 `0.7538`, 약 `18.39 ms/img`
  - `full_blend`: F1 `0.7584`
- 강점:
  - accuracy / latency 균형이 좋음
- 약점:
  - 이미지 단위 gating이라 지역적 refinement는 못 함

참조:
- `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/arch2_direct_probe_full6418.json`
- `/home/changmin/dark_vessel_sr_yolo/src/models/pipelines/arch2_softgate.py`

#### Arch4
- Scout(LR) -> uncertain ROI -> ROI SR -> Sniper -> merge
- selective ROI-SR라는 아이디어는 유효했음
- 하지만 구조적 병목이 큼

현재 best 경로:
1. crop-ft Sniper
2. hard-negative Sniper
3. crop-ft / hard-neg interpolation

현재 best:
- `interp_a03`
- full6418 direct:
  - F1 `0.7238`
  - P `0.7645`
  - R `0.6873`
  - TP `4411`
  - FP `1359`
  - FN `2007`
- full6418 mAP50:
  - `0.7561`

overnight retuning 최고치:
- `0.7246`
- 즉 기존 best 대비 사실상 미세 개선만 존재

참조:
- `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260318_arch4_interp_eval/arch4_interp_a03_direct_full6418.json`
- `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260318_arch4_interp_eval/arch4_interp_a03_eval_full6418.json`
- `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260318_223633_overnight_optimization/overnight_summary_ko.md`

---

## 3. Arch4에서 이미 확인된 것

Arch5 설계 전에 이걸 이해하지 못하면 같은 실수를 반복한다.

### 3-1. ROI-SR 자체는 의미가 있다
- ROI-RFDN 재학습 후 Arch4 `sr` branch는 실제로 개선되었다.
- `sr > bilinear`도 만들었다.
- 즉 “ROI-SR 자체가 무의미”한 건 아니다.

### 3-2. Sniper detector domain mismatch도 실제 병목이었다
- ROI crop 전용 Sniper fine-tuning(crop-ft)은 실제로 Arch4를 올렸다.
- hard-negative도 실제로 FP를 줄였다.
- interpolation까지 하면서 Arch4 best가 형성되었다.

즉 Arch4는:
- SR domain adaptation
- detector domain adaptation
모두 어느 정도 먹혔다.

### 3-3. 그런데도 Arch2를 못 넘었다
- 이건 Arch4의 남은 병목이 더 구조적이라는 뜻이다.
- merge/verifier/bonus/pass2_conf retuning까지 했지만 개선 폭은 제한적이었다.

### 3-4. Scout가 실제 병목이다

중요:
- current canonical config의 실제 `pass1_conf`는 `0.1`이 아니라 **`0.0075`**

current config 기준 Scout 진단:
- recall@IoU0.5 = `0.7280`
- recall@IoU0.3 = `0.8051`

해석:
- GT의 약 27%는 Scout가 IoU 0.5 기준으로 아예 커버하지 못한다.
- 동시에 IoU 0.3 기준으로는 0.805까지 올라가므로,
  - 아예 못 보는 문제
  - 근처는 보지만 localization quality가 부족한 문제
가 함께 존재한다.

참조:
- `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260319_scout_diagnostic/scout_recall_conf00075.json`
- `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260319_scout_diagnostic/scout_recall_conf00075_miou030.json`

---

## 4. Arch5가 필요한 이유

현재 상태를 냉정하게 요약하면:

- Arch2는 충분히 강하다
- Arch4는 선택적 ROI refinement 방향은 좋지만, Scout+merge 구조 때문에 ceiling이 낮다
- Arch0는 정확도는 높지만 full-image SR 비용이 크다

즉 Arch5는 다음 질문에 대한 답이어야 한다.

> full-image SR을 무조건 돌리지 않으면서도, Arch2의 안정성과 Arch4의 지역적 refinement를 함께 가져갈 수 있는가?

이 관점에서 Arch5의 자연스러운 방향은:

**Arch2 상위 gating + Arch4 ROI refinement cascade**

이다.

---

## 5. Arch5의 추천 설계

### 5-1. 이름

권장 이름:
- `Arch5HybridCascade`
- 또는 `Arch5 Gate-ROI Cascade`

코드 파일 제안:
- `/home/changmin/dark_vessel_sr_yolo/src/models/pipelines/arch5_hybrid_cascade.py`

### 5-2. 핵심 아이디어

이미지 전체에 대해 먼저 Arch2식 gate를 본다.

그리고 gate score에 따라 3갈래로 분기한다.

#### Branch A: high-confidence high-need SR image
- gate score가 높으면
- Arch2 full-SR path
- 즉 full-image SR + detector

#### Branch B: low-need SR image
- gate score가 낮으면
- bilinear / bypass detector

#### Branch C: ambiguous image
- gate score가 중간이면
- full-image SR을 하지 않고
- LR detector로 scout-like first detection
- uncertain region만 ROI refinement
- 즉 Arch4-style selective ROI path 사용

요약:

```text
image-level gate
  -> high: Arch2 full-SR
  -> low: bypass
  -> mid: Arch4 ROI-refine
```

이 설계의 장점:
- 쉬운 이미지는 싸게 처리
- 매우 어려운 이미지는 full-image SR로 안정성 확보
- 애매한 이미지만 selective ROI refinement

즉 Arch4의 “모든 이미지를 Scout부터 가는 구조”를 피하고,
Arch2의 strong prior를 상위 레벨에서 먼저 적용한다.

---

## 6. 왜 이 방향이 합리적인가

### 6-1. Arch4의 큰 문제는 모든 이미지를 Scout 중심으로 풀려 한다는 점
- Scout quality가 충분하지 않으면 초반에 놓친다
- 이후 ROI path가 아무리 좋아도 복구가 안 된다

### 6-2. Arch2는 global image-level decision이 강하다
- gate만으로도 꽤 좋은 분기가 가능함
- selective skip이 실제로 latency 이점을 줌

### 6-3. Arch4의 ROI refinement는 “중간 난이도 구간”에서만 쓰는 것이 더 자연스럽다
- 전 이미지에 대해 Scout-first는 과하다
- 불확실한 이미지 subset에서만 selective ROI-SR를 쓰면 구조적 부담이 줄어든다

---

## 7. 반드시 알아야 할 코드 레퍼런스

### Arch2 레퍼런스
- `/home/changmin/dark_vessel_sr_yolo/src/models/pipelines/arch2_softgate.py`

특히 봐야 할 부분:
- gate network 생성
- selective inference path
- gate threshold handling

### Arch4 레퍼런스
- `/home/changmin/dark_vessel_sr_yolo/src/models/pipelines/arch4_roi_awareNMS_ablation.py`
- `/home/changmin/dark_vessel_sr_yolo/src/models/pipelines/arch4_adaptive.py`

특히 봐야 할 부분:
- Scout predict
- ROI grouping
- crop extraction
- Sniper global remap
- merge policy

### base pipeline
- `/home/changmin/dark_vessel_sr_yolo/src/models/pipelines/base_pipeline.py`

### 주의: 기존 `arch5b_fusion.py`
- `/home/changmin/dark_vessel_sr_yolo/src/models/pipelines/arch5b_fusion.py`

이 파일은 **이미 존재하지만, 현재 원하는 Arch5와 방향이 다를 가능성이 높다.**

이 파일은:
- feature fusion 계열
- detector feature + SR feature fusion
중심으로 보인다.

즉 이번 Arch5 handoff의 핵심인
- image-level gating
- ROI refinement cascade
와는 다른 설계 축이다.

따라서:
- 참고는 가능
- 하지만 **이 파일을 메인 시작점으로 삼는 것은 비추천**

---

## 8. Arch5 구현 시 피해야 할 것

### 8-1. Arch4를 그대로 더 복잡하게 만들지 말 것
- 지금 Arch4는 이미 복잡하고, 미세 조정 이득이 작아졌다
- Arch5는 “복잡한 Arch4++”가 아니라 **분기 구조를 새로 짜는 것**이어야 한다

### 8-2. feature fusion으로 바로 가지 말 것
- 현재 병목은 fusion expressive power 부족보다
  - gating
  - scout recall
  - ROI selection
  - global/local decision hierarchy
쪽에 더 가깝다

### 8-3. 처음부터 end-to-end training까지 욕심내지 말 것
- 1차 Arch5는 inference-time hybrid prototype이 맞다
- 즉 기존 weight들을 최대한 재사용해서 forward path를 검증해야 한다

---

## 9. Arch5 1차 구현 목표

### 목표
먼저 **train-free prototype**을 만든다.

즉:
- Arch2 gate weights 그대로 사용
- Arch4 Scout/ROI/Sniper weights 그대로 사용
- RFDN weights 그대로 사용
- Sniper는 current best `interp_a03` 사용

이걸로 “구조가 먹히는지” 먼저 본다.

### 1차에서 필요한 것
- 새 pipeline class
- config schema
- full6418 direct eval 가능

### 1차에서 굳이 안 필요한 것
- 새 학습
- 새 loss
- end-to-end gradient flow

---

## 10. Arch5 1차 동작 제안

### 입력
- LR image

### Step 1. gate inference
- Arch2 gate network로 score `g` 계산

### Step 2. 분기
- `g >= t_high`
  - full-image SR path
  - detector result 사용
- `g <= t_low`
  - bypass path
  - detector result 사용
- `t_low < g < t_high`
  - ROI path
  - Arch4-style scout -> ROI -> SR -> Sniper

### Step 3. output unification
- 세 branch 모두 최종 output 포맷을 통일
- direct eval 가능하도록 `boxes/scores/classes` 형태 유지

---

## 11. 추천 초기 threshold

초기 제안:
- `t_low = 0.30`
- `t_high = 0.70`

이건 시작점일 뿐이다.  
나중에 grid search 대상이 될 수 있다.

---

## 12. 1차 성공 기준

### 절대 기준
- Arch5 prototype이 깨지지 않고 full-val direct eval이 돌아갈 것

### 상대 기준
- Arch4 best `0.7238`보다 높을 것

### 강한 성공 기준
- Arch2 `0.7538`에 근접하거나 넘을 것

즉 우선순위는:
1. Arch4보다 높다
2. Arch2에 근접한다

---

## 13. 1차 실험 순서

1. Arch5 hybrid pipeline 구현
2. 기존 weights들 연결
3. 소량 smoke eval
4. full6418 direct eval
5. branch usage 통계 출력
   - full-SR path 비율
   - bypass path 비율
   - ROI path 비율

이 branch usage 통계가 중요하다.  
왜냐하면 Arch5가 실제로 어떻게 동작하는지 설명 가능해야 하기 때문이다.

---

## 14. 출력해야 할 추가 통계

Arch5는 단순 F1만 보면 안 된다.  
반드시 아래를 같이 출력해야 한다.

- `num_images_full_sr`
- `num_images_bypass`
- `num_images_roi`
- `ratio_full_sr`
- `ratio_bypass`
- `ratio_roi`
- path별 평균 latency
- overall avg ms/img

---

## 15. 새로운 Codex에게 주는 실무 지시

### 첫 번째 할 일
- 위 레퍼런스 파일들을 읽고
- `arch5_hybrid_cascade.py` 1차 버전을 만든다

### 두 번째 할 일
- existing weights를 그대로 써서 돌아가게 만든다

### 세 번째 할 일
- full6418 direct eval이 가능하도록 평가 루틴을 연결한다

### 네 번째 할 일
- branch ratio 통계를 출력한다

### 다섯 번째 할 일
- current Arch4 / Arch2와 비교한 결과를 문서화한다

---

## 16. 최종 메시지

Arch5는 “새로운 SR 모델”을 만드는 프로젝트가 아니다.  
Arch5는 **이미 확보한 강한 부품들**:

- Arch2 gate
- Arch4 ROI refinement
- ROI-RFDN
- crop-ft/hard-neg/interp Sniper

를 **더 나은 계층 구조로 재배치**하는 프로젝트다.

즉, 이 문제의 본질은 component quality보다 **decision hierarchy redesign**에 있다.

---

## 17. 참고 문서

### 가장 중요한 결과 요약
- `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260318_arch4_interp_eval/arch4_interp_summary_ko.md`
- `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260319_scout_diagnostic/scout_diagnostic_summary_ko.md`
- `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260318_223633_overnight_optimization/overnight_summary_ko.md`

### Notion-ready 요약
- `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260319_notion_ready/01_arch4_overnight_result_summary_ko.md`
- `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260319_notion_ready/02_scout_diagnostic_summary_ko.md`
- `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260319_notion_ready/03_scout_retrain_launch_ko.md`

---

## 18. 다른 Codex가 처음 2시간 안에 해야 할 일

이 handoff를 받은 Codex는 아래 순서로 시작하는 것이 가장 안전하다.

1. 코드 읽기
- `/home/changmin/dark_vessel_sr_yolo/src/models/pipelines/arch2_softgate.py`
- `/home/changmin/dark_vessel_sr_yolo/src/models/pipelines/arch4_roi_awareNMS_ablation.py`
- `/home/changmin/dark_vessel_sr_yolo/src/models/pipelines/base_pipeline.py`

2. Arch5 파일 생성
- 새 파일:
  - `/home/changmin/dark_vessel_sr_yolo/src/models/pipelines/arch5_hybrid_cascade.py`

3. 최소 config 생성
- 새 config 예시:
  - `/home/changmin/dark_vessel_sr_yolo/configs/experiment/arch5_hybrid_cascade.yaml`

4. smoke path 구현
- gate 추론
- 3-branch 분기
- branch별 detector 결과를 공통 포맷으로 반환

5. smoke eval
- `max_images=20` 수준으로 깨지지 않는지 확인

6. full6418 direct eval
- 깨지지 않으면 바로 full eval

즉, 첫날 목표는 “성능 최적화”가 아니라 **구조가 실제로 돌아가고 숫자가 나오는 상태**를 만드는 것이다.

---

## 19. Arch5 1차 구현에서 반드시 필요한 config 필드

최소한 아래 필드는 있어야 한다.

```yaml
model:
  arch5:
    gate_low: 0.30
    gate_high: 0.70
    use_arch2_full_sr_path: true
    use_arch2_bypass_path: true
    use_arch4_roi_path: true
    record_branch_stats: true
```

추가로 기존 weight 경로들을 그대로 받을 수 있어야 한다.

```yaml
weights:
  gate_arch2: <arch2 gate weights path>
  scout_lr: <yolo_lr path>
  sniper_hr: <interp_a03 path>
  sr_roi: <rfdn_arch4 path>
```

중요:
- 1차에서는 새 weight를 학습하지 않는다.
- **기존 weight를 꽂을 수 있게 plumbing을 만드는 것**이 핵심이다.

---

## 20. 권장 forward 흐름 의사코드

```python
def forward(lr_images):
    gate_scores = arch2_gate(lr_images)

    results = []
    branch_stats = init_stats()

    for img, g in zip(lr_images, gate_scores):
        if g >= gate_high:
            det = run_arch2_full_sr_path(img)
            branch_stats["full_sr"] += 1
        elif g <= gate_low:
            det = run_arch2_bypass_path(img)
            branch_stats["bypass"] += 1
        else:
            det = run_arch4_roi_refine_path(img)
            branch_stats["roi"] += 1

        results.append(normalize_output(det))

    return {
        "results": results,
        "branch_stats": branch_stats,
    }
```

여기서 중요한 건:
- branch별 내부 구현이 달라도
- **최종 출력 포맷은 완전히 동일해야 한다**는 점이다.

즉 `boxes/scores/classes`를 동일 구조로 맞춰야 evaluator 연결이 쉽다.

---

## 21. 다른 Codex가 피해야 할 구현 함정

### 함정 1. Arch2 전체 코드를 그대로 복붙해서 얹는 것
- 이렇게 하면 branch별 weight 관리가 꼬이기 쉽다.
- gate만 재사용하고 path는 명시적으로 분리하는 편이 낫다.

### 함정 2. Arch4 path를 수정하면서 기존 canonical을 깨뜨리는 것
- Arch5 구현 때문에 Arch4 canonical 동작이 변하면 안 된다.
- **새 파일에서 wrapping**하는 방식이 맞다.

### 함정 3. 초반부터 latency까지 완벽히 맞추려는 것
- 1차 목표는 정확도/구조 검증이다.
- latency는 branch stats가 나온 뒤 2차에서 본다.

### 함정 4. 바로 fusion/feature-level 결합으로 빠지는 것
- 그건 Arch5b류로 흘러가며, 지금 필요한 질문과 다르다.

---

## 22. 1차 평가에서 반드시 남겨야 할 산출물

다른 Codex는 최소 아래 파일을 남겨야 한다.

1. Arch5 config
- `/home/changmin/dark_vessel_sr_yolo/configs/experiment/arch5_hybrid_cascade.yaml`

2. smoke 결과 json
- `iac_runs/.../arch5_smoke.json`

3. full6418 direct 결과 json
- `iac_runs/.../arch5_direct_full6418.json`

4. branch usage summary
- `iac_runs/.../arch5_branch_stats.json`

5. 설명 문서
- `iac_runs/.../arch5_first_result_summary_ko.md`

즉 “코드만 남기고 끝”이 아니라, **다음 사람이 바로 비교할 수 있는 결과 패키지**를 남겨야 한다.

---

## 23. Arch5가 성공했다고 볼 최소 기준

### 실패
- full eval이 안 돈다
- Arch4 best `0.7238`보다 낮다

### 부분 성공
- full eval이 돌고
- Arch4 best를 넘는다

### 강한 성공
- Arch2 `0.7538` 근처까지 간다
- 또는 branch ratio가 설득력 있게 나와서 “왜 이 구조가 합리적인지” 설명 가능하다

즉 Arch5는 단순히 숫자 하나만 보는 게 아니다.
**성능 + branch 사용 패턴의 설명 가능성**까지 함께 봐야 한다.
