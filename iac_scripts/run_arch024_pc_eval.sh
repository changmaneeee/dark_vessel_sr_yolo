#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Arch0 / Arch2 / Arch4 PC 평가 일괄 실행 스크립트
# ------------------------------------------------------------
# 사용 예시:
#   EXP_NAME=yolo_swap_rfdn500 \
#   YOLO_HR_WEIGHTS=/home/changmin/dark_vessel_sr_yolo/weights/yolohr/8s/best.pt \
#   YOLO_LR_WEIGHTS=/home/changmin/dark_vessel_sr_yolo/weights/yolo_lr/8s/best.pt \
#   YOLO_SR_WEIGHTS=/home/changmin/dark_vessel_sr_yolo/weights/yolo_8s_rfdn/best.pt \
#   bash run_arch024_pc_eval.sh
#
<< 'COMMENT'
EXP_NAME=yolo_swap_rfdn500 \
YOLO_HR_WEIGHTS=/home/changmin/dark_vessel_sr_yolo/weights/yolohr/8s/best.pt \
YOLO_LR_WEIGHTS=/home/changmin/dark_vessel_sr_yolo/weights/yolo_lr/8s/best.pt \
YOLO_SR_WEIGHTS=/home/changmin/dark_vessel_sr_yolo/weights/yolo_8s_rfdn/best.pt \
bash run_arch024_pc_eval.sh
COMMENT

# 핵심 포인트:
# - 새 실험 폴더를 자동 생성
# - config 복사/패치 후 원본은 건드리지 않음
# - HR/LR pure YOLO baseline JSON 저장
# - Arch0 / Arch2 / Arch4(full + deploy-like 200) 결과 저장
#
# 주의:
# - evaluate_arch0.py / evaluate_arch2.py 는 단일 --yolo_weights 만 받음
# - 따라서 Arch0/2 내부 HR/LR/SR 비교는 YOLO_SR_WEIGHTS 기준으로 계산됨
# - 공정 baseline은 이 스크립트가 따로 저장하는 baseline_hr.json / baseline_lr.json 을 보세요.
# ============================================================

# ---------- 반드시 repo root 에서 실행 ----------
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "$REPO_ROOT"

# ---------- 실험명 / 저장 경로 ----------
EXP_NAME="${EXP_NAME:-pc_eval_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-pc_eval_runs/${EXP_NAME}}"
CFG_DIR="$RUN_ROOT/configs"
RES_DIR="$RUN_ROOT/results"
LOG_DIR="$RUN_ROOT/logs"
TMP_DIR="$RUN_ROOT/tmp"
mkdir -p "$CFG_DIR" "$RES_DIR" "$LOG_DIR" "$TMP_DIR"

# ---------- 데이터 ----------
HR_DATA_YAML="${HR_DATA_YAML:-/home/changmin/smart_airbus_data/data.yaml}"
LR_DATA_YAML="${LR_DATA_YAML:-/home/changmin/smart_airbus_data_lr/data.yaml}"

# ---------- 모델 경로 ----------
YOLO_HR_WEIGHTS="${YOLO_HR_WEIGHTS:-}"
YOLO_LR_WEIGHTS="${YOLO_LR_WEIGHTS:-}"
YOLO_SR_WEIGHTS="${YOLO_SR_WEIGHTS:-}"

# 선택: arch 전체 weights (없으면 비워두면 됨)
ARCH0_WEIGHTS="${ARCH0_WEIGHTS:-}"
ARCH2_WEIGHTS="${ARCH2_WEIGHTS:-}"

# ---------- base config ----------
ARCH0_BASE_CFG="${ARCH0_BASE_CFG:-configs/experiment/arch0_sequential.yaml}"
ARCH2_BASE_CFG="${ARCH2_BASE_CFG:-configs/experiment/arch2_softgate.yaml}"
ARCH4_BAL_BASE_CFG="${ARCH4_BAL_BASE_CFG:-configs/experiment/arch4_stage2_balanced_best.yaml}"
ARCH4_REC_BASE_CFG="${ARCH4_REC_BASE_CFG:-configs/experiment/arch4_stage2_recall_best.yaml}"

# ---------- script path ----------
ARCH0_EVAL_SCRIPT="${ARCH0_EVAL_SCRIPT:-/home/changmin/dark_vessel_sr_yolo/scripts/evaluate_arch0.py}"
ARCH2_EVAL_SCRIPT="${ARCH2_EVAL_SCRIPT:-/home/changmin/dark_vessel_sr_yolo/scripts/evaluate_arch2.py}"
ARCH4_EVAL_SCRIPT="${ARCH4_EVAL_SCRIPT:-iac_scripts/arch4_eval_ultralytics.py}"

# ---------- 공통 평가 파라미터 ----------
DEVICE="${DEVICE:-cuda}"
CONF="${CONF:-0.001}"
IOU="${IOU:-0.6}"
MAX_DET="${MAX_DET:-300}"
YOLO_BATCH="${YOLO_BATCH:-16}"
WORKERS="${WORKERS:-8}"

# Arch0/2: 0 이면 전체 이미지로 동작 (스크립트 내부 if max_images: ... 로 처리)
ARCH02_MAX_IMAGES="${ARCH02_MAX_IMAGES:-0}"
# Arch4 full: 0 이면 --max_images 를 안 넘겨서 전체 평가
ARCH4_MAX_IMAGES="${ARCH4_MAX_IMAGES:-0}"
# Arch4 deploy-like: 보통 200 추천
ARCH4_DEPLOY_MAX_IMAGES="${ARCH4_DEPLOY_MAX_IMAGES:-200}"

# ---------- sanity check ----------
if [[ -z "$YOLO_HR_WEIGHTS" ]]; then
  echo "[ERROR] YOLO_HR_WEIGHTS is empty"
  exit 1
fi
if [[ -z "$YOLO_LR_WEIGHTS" ]]; then
  YOLO_LR_WEIGHTS="$YOLO_HR_WEIGHTS"
fi
if [[ -z "$YOLO_SR_WEIGHTS" ]]; then
  YOLO_SR_WEIGHTS="$YOLO_HR_WEIGHTS"
fi

for p in \
  "$HR_DATA_YAML" "$LR_DATA_YAML" \
  "$YOLO_HR_WEIGHTS" "$YOLO_LR_WEIGHTS" "$YOLO_SR_WEIGHTS" \
  "$ARCH0_BASE_CFG" "$ARCH2_BASE_CFG" "$ARCH4_BAL_BASE_CFG" "$ARCH4_REC_BASE_CFG" \
  "$ARCH0_EVAL_SCRIPT" "$ARCH2_EVAL_SCRIPT" "$ARCH4_EVAL_SCRIPT"
  do
  if [[ ! -e "$p" ]]; then
    echo "[ERROR] Missing path: $p"
    exit 1
  fi
done

echo "============================================================"
echo "[RUN] $EXP_NAME"
echo "[ROOT] $RUN_ROOT"
echo "============================================================"

echo "EXP_NAME=$EXP_NAME"                  >  "$RUN_ROOT/run_env.txt"
echo "RUN_ROOT=$RUN_ROOT"                  >> "$RUN_ROOT/run_env.txt"
echo "HR_DATA_YAML=$HR_DATA_YAML"          >> "$RUN_ROOT/run_env.txt"
echo "LR_DATA_YAML=$LR_DATA_YAML"          >> "$RUN_ROOT/run_env.txt"
echo "YOLO_HR_WEIGHTS=$YOLO_HR_WEIGHTS"    >> "$RUN_ROOT/run_env.txt"
echo "YOLO_LR_WEIGHTS=$YOLO_LR_WEIGHTS"    >> "$RUN_ROOT/run_env.txt"
echo "YOLO_SR_WEIGHTS=$YOLO_SR_WEIGHTS"    >> "$RUN_ROOT/run_env.txt"
echo "ARCH0_WEIGHTS=$ARCH0_WEIGHTS"        >> "$RUN_ROOT/run_env.txt"
echo "ARCH2_WEIGHTS=$ARCH2_WEIGHTS"        >> "$RUN_ROOT/run_env.txt"
echo "DEVICE=$DEVICE"                      >> "$RUN_ROOT/run_env.txt"
echo "CONF=$CONF"                          >> "$RUN_ROOT/run_env.txt"
echo "IOU=$IOU"                            >> "$RUN_ROOT/run_env.txt"
echo "MAX_DET=$MAX_DET"                    >> "$RUN_ROOT/run_env.txt"
echo "YOLO_BATCH=$YOLO_BATCH"              >> "$RUN_ROOT/run_env.txt"
echo "WORKERS=$WORKERS"                    >> "$RUN_ROOT/run_env.txt"
echo "ARCH02_MAX_IMAGES=$ARCH02_MAX_IMAGES" >> "$RUN_ROOT/run_env.txt"
echo "ARCH4_MAX_IMAGES=$ARCH4_MAX_IMAGES"  >> "$RUN_ROOT/run_env.txt"
echo "ARCH4_DEPLOY_MAX_IMAGES=$ARCH4_DEPLOY_MAX_IMAGES" >> "$RUN_ROOT/run_env.txt"

# ---------- helper: 실행 + 로그 ----------
run_logged() {
  local name="$1"
  shift
  echo
  echo "============================================================"
  echo "[STEP] $name"
  echo "============================================================"
  echo "CMD: $*"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

# ---------- helper: pure YOLO val -> JSON ----------
run_baseline_json() {
  local tag="$1"
  local model_path="$2"
  local data_yaml="$3"
  local out_json="$4"
  python - <<PY
import json
from ultralytics import YOLO

model_path = r'''$model_path'''
data_yaml = r'''$data_yaml'''
out_json = r'''$out_json'''
conf = float(r'''$CONF''')
iou = float(r'''$IOU''')
max_det = int(r'''$MAX_DET''')
device = r'''$DEVICE'''
batch = int(r'''$YOLO_BATCH''')
workers = int(r'''$WORKERS''')

y = YOLO(model_path)
res = y.val(
    data=data_yaml,
    imgsz=640,
    conf=conf,
    iou=iou,
    max_det=max_det,
    device=device,
    batch=batch,
    workers=workers,
    verbose=False,
    plots=False,
    save_json=False,
)
out = {
    "tag": "$tag",
    "model": model_path,
    "data": data_yaml,
    "results_dict": {k: (float(v) if hasattr(v, 'item') else v) for k, v in res.results_dict.items()}
}
with open(out_json, "w") as f:
    json.dump(out, f, indent=2)
print(f"saved: {out_json}")
print(out["results_dict"])
PY
}

# ---------- config 복사/패치 ----------
python - <<PY
import yaml
from pathlib import Path

def load_yaml(p):
    with open(p, 'r') as f:
        return yaml.safe_load(f) or {}

def save_yaml(obj, p):
    with open(p, 'w') as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def set_path(d, keys, value):
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

cfg_dir = Path(r'''$CFG_DIR''')
arch0 = load_yaml(r'''$ARCH0_BASE_CFG''')
arch2 = load_yaml(r'''$ARCH2_BASE_CFG''')
arch4b = load_yaml(r'''$ARCH4_BAL_BASE_CFG''')
arch4r = load_yaml(r'''$ARCH4_REC_BASE_CFG''')

yolo_hr = r'''$YOLO_HR_WEIGHTS'''
yolo_lr = r'''$YOLO_LR_WEIGHTS'''

# Arch0 / Arch2 -> single detector path
for cfg in [arch0, arch2]:
    set_path(cfg, ["model", "weights", "detector"], yolo_hr)
    set_path(cfg, ["model", "yolo", "weights_path"], yolo_hr)
    set_path(cfg, ["model", "yolo", "weights"], yolo_hr)

# Arch4 -> dual detector
for cfg in [arch4b, arch4r]:
    set_path(cfg, ["model", "yolo", "weights_hr"], yolo_hr)
    set_path(cfg, ["model", "yolo", "weights_lr"], yolo_lr)
    # alias도 같이 맞춰두기
    arch4 = cfg.setdefault("model", {}).setdefault("arch4", {})
    if "high_conf" in arch4 and "pass2_conf" not in arch4:
        arch4["pass2_conf"] = arch4["high_conf"]

save_yaml(arch0, cfg_dir / "arch0.yaml")
save_yaml(arch2, cfg_dir / "arch2.yaml")
save_yaml(arch4b, cfg_dir / "arch4_balanced.yaml")
save_yaml(arch4r, cfg_dir / "arch4_recall.yaml")

# deploy 버전도 같이 생성
for src_name, dst_name in [
    ("arch4_balanced.yaml", "arch4_balanced_deploy.yaml"),
    ("arch4_recall.yaml", "arch4_recall_deploy.yaml"),
]:
    p = cfg_dir / src_name
    cfg = load_yaml(p)
    arch4 = cfg.setdefault("model", {}).setdefault("arch4", {})
    arch4["sniper_conf"] = 0.001
    arch4["final_conf"] = 0.25
    save_yaml(cfg, cfg_dir / dst_name)

print("saved configs under", cfg_dir)
PY

# ---------- 1) baseline ----------
echo
echo "============================================================"
echo "[STEP] baseline_hr"
echo "============================================================"
run_baseline_json baseline_hr "$YOLO_HR_WEIGHTS" "$HR_DATA_YAML" "$RES_DIR/baseline_hr.json" \
  2>&1 | tee "$LOG_DIR/baseline_hr.log"

echo
echo "============================================================"
echo "[STEP] baseline_lr"
echo "============================================================"
run_baseline_json baseline_lr "$YOLO_LR_WEIGHTS" "$LR_DATA_YAML" "$RES_DIR/baseline_lr.json" \
  2>&1 | tee "$LOG_DIR/baseline_lr.log"

# ---------- 2) Arch0 ----------
ARCH0_CMD=(python "$ARCH0_EVAL_SCRIPT" \
  --config "$CFG_DIR/arch0.yaml" \
  --yolo_weights "$YOLO_SR_WEIGHTS" \
  --hr_data_yaml "$HR_DATA_YAML" \
  --lr_data_yaml "$LR_DATA_YAML" \
  --sr_output_dir "$TMP_DIR/arch0_sr_cache" \
  --output "$RES_DIR/arch0_eval.json" \
  --conf "$CONF" \
  --iou "$IOU" \
  --device "$DEVICE" \
  --max_images "$ARCH02_MAX_IMAGES")
if [[ -n "$ARCH0_WEIGHTS" && -e "$ARCH0_WEIGHTS" ]]; then
  ARCH0_CMD+=(--weights "$ARCH0_WEIGHTS")
fi
run_logged "arch0_eval" "${ARCH0_CMD[@]}"

# ---------- 3) Arch2 ----------
ARCH2_CMD=(python "$ARCH2_EVAL_SCRIPT" \
  --config "$CFG_DIR/arch2.yaml" \
  --yolo_weights "$YOLO_SR_WEIGHTS" \
  --hr_data_yaml "$HR_DATA_YAML" \
  --lr_data_yaml "$LR_DATA_YAML" \
  --sr_output_dir "$TMP_DIR/arch2_sr_cache" \
  --output "$RES_DIR/arch2_eval.json" \
  --conf "$CONF" \
  --iou "$IOU" \
  --device "$DEVICE" \
  --max_images "$ARCH02_MAX_IMAGES")
if [[ -n "$ARCH2_WEIGHTS" && -e "$ARCH2_WEIGHTS" ]]; then
  ARCH2_CMD+=(--weights "$ARCH2_WEIGHTS")
fi
run_logged "arch2_eval" "${ARCH2_CMD[@]}"

# ---------- 4) Arch4 full ----------
ARCH4_BAL_CMD=(python "$ARCH4_EVAL_SCRIPT" \
  --arch4_config "$CFG_DIR/arch4_balanced.yaml" \
  --hr_data_yaml "$HR_DATA_YAML" \
  --lr_data_yaml "$LR_DATA_YAML" \
  --eval_space hr \
  --batch 1 \
  --device "$DEVICE" \
  --out_json "$RES_DIR/arch4_balanced_full.json")
if [[ "$ARCH4_MAX_IMAGES" != "0" ]]; then
  ARCH4_BAL_CMD+=(--max_images "$ARCH4_MAX_IMAGES")
fi
run_logged "arch4_balanced_full" "${ARCH4_BAL_CMD[@]}"

ARCH4_REC_CMD=(python "$ARCH4_EVAL_SCRIPT" \
  --arch4_config "$CFG_DIR/arch4_recall.yaml" \
  --hr_data_yaml "$HR_DATA_YAML" \
  --lr_data_yaml "$LR_DATA_YAML" \
  --eval_space hr \
  --batch 1 \
  --device "$DEVICE" \
  --out_json "$RES_DIR/arch4_recall_full.json")
if [[ "$ARCH4_MAX_IMAGES" != "0" ]]; then
  ARCH4_REC_CMD+=(--max_images "$ARCH4_MAX_IMAGES")
fi
run_logged "arch4_recall_full" "${ARCH4_REC_CMD[@]}"

# ---------- 5) Arch4 deploy-like 200 ----------
run_logged "arch4_balanced_deploy_200" python "$ARCH4_EVAL_SCRIPT" \
  --arch4_config "$CFG_DIR/arch4_balanced_deploy.yaml" \
  --hr_data_yaml "$HR_DATA_YAML" \
  --lr_data_yaml "$LR_DATA_YAML" \
  --eval_space hr \
  --batch 1 \
  --max_images "$ARCH4_DEPLOY_MAX_IMAGES" \
  --device "$DEVICE" \
  --out_json "$RES_DIR/arch4_balanced_deploy_200.json"

run_logged "arch4_recall_deploy_200" python "$ARCH4_EVAL_SCRIPT" \
  --arch4_config "$CFG_DIR/arch4_recall_deploy.yaml" \
  --hr_data_yaml "$HR_DATA_YAML" \
  --lr_data_yaml "$LR_DATA_YAML" \
  --eval_space hr \
  --batch 1 \
  --max_images "$ARCH4_DEPLOY_MAX_IMAGES" \
  --device "$DEVICE" \
  --out_json "$RES_DIR/arch4_recall_deploy_200.json"

# ---------- 6) summary ----------
cat > "$RUN_ROOT/README_result_paths.txt" <<TXT
[Experiment]
EXP_NAME=$EXP_NAME
RUN_ROOT=$RUN_ROOT

[Main result JSON]
- $RES_DIR/baseline_hr.json
- $RES_DIR/baseline_lr.json
- $RES_DIR/arch0_eval.json
- $RES_DIR/arch2_eval.json
- $RES_DIR/arch4_balanced_full.json
- $RES_DIR/arch4_recall_full.json
- $RES_DIR/arch4_balanced_deploy_200.json
- $RES_DIR/arch4_recall_deploy_200.json

[Patched configs]
- $CFG_DIR/arch0.yaml
- $CFG_DIR/arch2.yaml
- $CFG_DIR/arch4_balanced.yaml
- $CFG_DIR/arch4_recall.yaml
- $CFG_DIR/arch4_balanced_deploy.yaml
- $CFG_DIR/arch4_recall_deploy.yaml

[Notes]
- Arch0/Arch2 내부 비교는 YOLO_SR_WEIGHTS 기준으로 계산됨
- pure baseline 비교는 baseline_hr.json / baseline_lr.json 을 우선 참고
TXT

echo
echo "============================================================"
echo "DONE: $RUN_ROOT"
echo "============================================================"
cat "$RUN_ROOT/README_result_paths.txt"
