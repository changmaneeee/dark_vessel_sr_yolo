#!/usr/bin/env bash
set -euo pipefail

<< comment
SR_SNIPER_WEIGHTS=/home/changmin/dark_vessel_sr_yolo/weights/yolo_8s_rfdn/best.pt \
EXP_TAG=arch4_sr_sniper_check \
bash run_arch4_sr_sniper_baseline.sh
comment

# =========================================================
# Arch4 Sniper swap baseline test
# - keeps Scout detector unchanged
# - replaces Sniper detector with SR-domain YOLO
# - runs eval-like and deploy-like tests
# =========================================================

PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-cuda}
HR_DATA_YAML=${HR_DATA_YAML:-/home/changmin/smart_airbus_data/data.yaml}
LR_DATA_YAML=${LR_DATA_YAML:-/home/changmin/smart_airbus_data_lr/data.yaml}
BASE_EVAL_CFG=${BASE_EVAL_CFG:-configs/experiment/arch4_roi_awareNMS_eval.yaml}
BASE_DEPLOY_CFG=${BASE_DEPLOY_CFG:-configs/experiment/arch4_roi_awareNMS_deploy.yaml}
MAX_IMAGES_EVAL=${MAX_IMAGES_EVAL:-200}
MAX_IMAGES_DEPLOY=${MAX_IMAGES_DEPLOY:-200}
EXP_TAG=${EXP_TAG:-arch4_sr_sniper_baseline}
ARCH4_EVAL_SCRIPT=${ARCH4_EVAL_SCRIPT:-iac_scripts/arch4_eval_ultralytics.py}

# REQUIRED: path to the YOLO detector trained on RFDN-SR images
SR_SNIPER_WEIGHTS=${SR_SNIPER_WEIGHTS:-}
if [[ -z "$SR_SNIPER_WEIGHTS" ]]; then
  echo "[ERROR] SR_SNIPER_WEIGHTS is empty."
  echo "Example:"
  echo "  SR_SNIPER_WEIGHTS=/ABS/PATH/TO/rfdn_yolo_epoch500.pt bash $0"
  exit 1
fi

RUN_DIR="iac_runs/${EXP_TAG}"
CFG_DIR="configs/experiment/${EXP_TAG}"
LOG_DIR="${RUN_DIR}/logs"
RESULT_DIR="${RUN_DIR}/results"
mkdir -p "$CFG_DIR" "$LOG_DIR" "$RESULT_DIR"

EVAL_CFG_OUT="${CFG_DIR}/arch4_sr_sniper_eval.yaml"
DEPLOY_CFG_OUT="${CFG_DIR}/arch4_sr_sniper_deploy.yaml"

# ---------------------------------------------------------
# 1) Patch configs: only replace Sniper detector path
# ---------------------------------------------------------
$PYTHON_BIN - <<PY
import yaml, os
from pathlib import Path

for src, dst in [
    (Path(${BASE_EVAL_CFG@Q}), Path(${EVAL_CFG_OUT@Q})),
    (Path(${BASE_DEPLOY_CFG@Q}), Path(${DEPLOY_CFG_OUT@Q})),
]:
    with open(src, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault('model', {}).setdefault('yolo', {})
    cfg['model']['yolo']['weights_hr'] = ${SR_SNIPER_WEIGHTS@Q}
    with open(dst, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"saved: {dst}")
PY

# ---------------------------------------------------------
# 2) Run eval-like test
# ---------------------------------------------------------
EVAL_JSON="${RESULT_DIR}/arch4_sr_sniper_eval_${MAX_IMAGES_EVAL}.json"
DEPLOY_JSON="${RESULT_DIR}/arch4_sr_sniper_deploy_${MAX_IMAGES_DEPLOY}.json"

echo "[RUN] Eval-like test"
EVAL_CMD=(
  "$PYTHON_BIN" "$ARCH4_EVAL_SCRIPT"
  --arch4_config "$EVAL_CFG_OUT"
  --hr_data_yaml "$HR_DATA_YAML"
  --lr_data_yaml "$LR_DATA_YAML"
  --eval_space hr
  --batch 1
  --device "$DEVICE"
  --out_json "$EVAL_JSON"
)
if [[ "$MAX_IMAGES_EVAL" != "0" ]]; then
  EVAL_CMD+=(--max_images "$MAX_IMAGES_EVAL")
fi
"${EVAL_CMD[@]}" | tee "$LOG_DIR/arch4_sr_sniper_eval.log"

# ---------------------------------------------------------
# 3) Run deploy-like test
# ---------------------------------------------------------
echo "[RUN] Deploy-like test"
DEPLOY_CMD=(
  "$PYTHON_BIN" "$ARCH4_EVAL_SCRIPT"
  --arch4_config "$DEPLOY_CFG_OUT"
  --hr_data_yaml "$HR_DATA_YAML"
  --lr_data_yaml "$LR_DATA_YAML"
  --eval_space hr
  --batch 1
  --device "$DEVICE"
  --out_json "$DEPLOY_JSON"
)
if [[ "$MAX_IMAGES_DEPLOY" != "0" ]]; then
  DEPLOY_CMD+=(--max_images "$MAX_IMAGES_DEPLOY")
fi
"${DEPLOY_CMD[@]}" | tee "$LOG_DIR/arch4_sr_sniper_deploy.log"

# ---------------------------------------------------------
# 4) Print compact summary
# ---------------------------------------------------------
$PYTHON_BIN - <<PY
import json
from pathlib import Path

for tag, p in [
    ('eval', Path(${EVAL_JSON@Q})),
    ('deploy', Path(${DEPLOY_JSON@Q})),
]:
    with open(p, 'r') as f:
        data = json.load(f)
    rd = data['runs'][0]['results_dict']
    meta = data.get('meta', {})
    print(f"\n[{tag.upper()} SUMMARY]")
    print(f"  mAP50-95      : {rd.get('metrics/mAP50-95(B)')}")
    print(f"  mAP50         : {rd.get('metrics/mAP50(B)')}")
    print(f"  Precision     : {rd.get('metrics/precision(B)')}")
    print(f"  Recall        : {rd.get('metrics/recall(B)')}")
    print(f"  direct TP50   : {rd.get('direct/tp50')}")
    print(f"  direct FP50   : {rd.get('direct/fp50')}")
    print(f"  direct FN50   : {rd.get('direct/fn50')}")
    print(f"  direct P50    : {rd.get('direct/precision50')}")
    print(f"  direct R50    : {rd.get('direct/recall50')}")
    print(f"  avg ms/img    : {meta.get('avg_ms_per_image')}")
    print(f"  json          : {p}")
PY

echo "\n[DONE] Results saved under: ${RUN_DIR}"
