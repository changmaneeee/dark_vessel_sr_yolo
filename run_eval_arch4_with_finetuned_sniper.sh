#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------
# Patch Arch4 config with a new Sniper checkpoint and evaluate it.
# -----------------------------------------------------------------
# Usage example:
#   BASE_CONFIG=configs/experiment/arch4_roi_awareNMS_eval.yaml \
#   NEW_SNIPER_WEIGHTS=iac_runs/sniper_finetune_train/sniper_ft_a11/weights/best.pt \
#   OUT_TAG=arch4_a11_ftsniper_full \
#   bash run_eval_arch4_with_finetuned_sniper.sh
# -----------------------------------------------------------------

PYTHON_BIN=${PYTHON_BIN:-python}
BASE_CONFIG=${BASE_CONFIG:-configs/experiment/arch4_roi_awareNMS_eval.yaml}
NEW_SNIPER_WEIGHTS=${NEW_SNIPER_WEIGHTS:-}
if [[ -z "$NEW_SNIPER_WEIGHTS" ]]; then
  echo "[ERROR] NEW_SNIPER_WEIGHTS is required"
  exit 1
fi

OUT_TAG=${OUT_TAG:-arch4_a11_ftsniper}
TMP_CONFIG_DIR=${TMP_CONFIG_DIR:-iac_runs/tmp_arch4_ftsniper_configs}
OUT_JSON=${OUT_JSON:-iac_runs/${OUT_TAG}.json}
HR_DATA_YAML=${HR_DATA_YAML:-/home/changmin/smart_airbus_data/data.yaml}
LR_DATA_YAML=${LR_DATA_YAML:-/home/changmin/smart_airbus_data_lr/data.yaml}
DEVICE=${DEVICE:-cuda}
MAX_IMAGES=${MAX_IMAGES:-0}
BATCH=${BATCH:-1}

mkdir -p "$TMP_CONFIG_DIR"
PATCHED_CFG="$TMP_CONFIG_DIR/${OUT_TAG}.yaml"

$PYTHON_BIN - <<PY
import yaml
from pathlib import Path
src = Path('$BASE_CONFIG')
dst = Path('$PATCHED_CFG')
with open(src, 'r') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('model', {}).setdefault('yolo', {})['weights_hr'] = '$NEW_SNIPER_WEIGHTS'
with open(dst, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(dst)
PY

CMD=(
  $PYTHON_BIN iac_scripts/arch4_eval_ultralytics.py
  --arch4_config "$PATCHED_CFG"
  --hr_data_yaml "$HR_DATA_YAML"
  --lr_data_yaml "$LR_DATA_YAML"
  --eval_space hr
  --batch "$BATCH"
  --device "$DEVICE"
  --out_json "$OUT_JSON"
)
if [[ "$MAX_IMAGES" != "0" ]]; then
  CMD+=(--max_images "$MAX_IMAGES")
fi

printf '[EVAL CMD] %q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
