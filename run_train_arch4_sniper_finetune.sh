#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------
# Train / fine-tune the Arch4 Sniper YOLO on the generated SR-crop set
# -----------------------------------------------------------------
# Usage example:
#   DATA_YAML=iac_runs/sniper_ft_dataset_a11/data.yaml \
#   BASE_WEIGHTS=/ABS/PATH/TO/rfdn_sr_yolo500.pt \
#   EXP_NAME=sniper_ft_a11_e100 \
#   bash run_train_arch4_sniper_finetune.sh
# -----------------------------------------------------------------

DATA_YAML=${DATA_YAML:-iac_runs/sniper_ft_dataset_a11/data.yaml}
BASE_WEIGHTS=${BASE_WEIGHTS:-}
if [[ -z "$BASE_WEIGHTS" ]]; then
  echo "[ERROR] BASE_WEIGHTS is required"
  exit 1
fi

EXP_NAME=${EXP_NAME:-sniper_ft_a11}
PROJECT_DIR=${PROJECT_DIR:-iac_runs/sniper_finetune_train}
IMGSZ=${IMGSZ:-256}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-64}
DEVICE=${DEVICE:-0}
WORKERS=${WORKERS:-8}
PATIENCE=${PATIENCE:-20}
CLOSE_MOSAIC=${CLOSE_MOSAIC:-10}
LR0=${LR0:-0.001}
LRF=${LRF:-0.01}
FREEZE=${FREEZE:-0}

mkdir -p "$PROJECT_DIR"
LOG_PATH="$PROJECT_DIR/${EXP_NAME}.log"

CMD=(
  yolo detect train
  model="$BASE_WEIGHTS"
  data="$DATA_YAML"
  imgsz="$IMGSZ"
  epochs="$EPOCHS"
  batch="$BATCH"
  device="$DEVICE"
  workers="$WORKERS"
  patience="$PATIENCE"
  close_mosaic="$CLOSE_MOSAIC"
  lr0="$LR0"
  lrf="$LRF"
  freeze="$FREEZE"
  project="$PROJECT_DIR"
  name="$EXP_NAME"
)

printf '[TRAIN CMD] %q ' "${CMD[@]}" | tee "$LOG_PATH"
printf '\n' | tee -a "$LOG_PATH"

"${CMD[@]}" 2>&1 | tee -a "$LOG_PATH"

echo "[DONE] training complete" | tee -a "$LOG_PATH"
