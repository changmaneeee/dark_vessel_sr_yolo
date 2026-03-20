#!/usr/bin/env bash
set -euo pipefail

# Quick wrapper for arch4_wiring_check.py
#
# Example:
#   PROJECT_ROOT=/path/to/repo \
#   ARCH4_CONFIG=/path/to/repo/configs/experiment/arch4_roi_awareNMS_deploy.yaml \
#   ARCH4_PY=/path/to/repo/src/models/pipelines/arch4_roi_awareNMS_ablation.py \
#   LR_IMAGES_DIR=/path/to/lr/images/val \
#   HR_IMAGES_DIR=/path/to/hr/images/val \
#   HR_LABELS_DIR=/path/to/hr/labels/val \
#   MODES=sr,bilinear,hr_ref \
#   MAX_IMAGES=100 \
#   DEVICE=cuda \
#   OUT_JSON=/tmp/arch4_wiring_check.json \
#   bash run_arch4_wiring_check.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_ROOT="${PROJECT_ROOT:-}"
ARCH4_WIRING_SCRIPT="${ARCH4_WIRING_SCRIPT:-${PROJECT_ROOT:+$PROJECT_ROOT/iac_jetson/arch4_wiring_check.py}}"
ARCH4_CONFIG="${ARCH4_CONFIG:-}"
ARCH4_PY="${ARCH4_PY:-}"
LR_IMAGES_DIR="${LR_IMAGES_DIR:-}"
HR_IMAGES_DIR="${HR_IMAGES_DIR:-}"
HR_LABELS_DIR="${HR_LABELS_DIR:-}"
MAX_IMAGES="${MAX_IMAGES:-100}"
DEVICE="${DEVICE:-cuda}"
HALF_FLAG="${HALF_FLAG:-}"
MODES="${MODES:-sr,bilinear,hr_ref}"
EVAL_SPACE="${EVAL_SPACE:-hr}"
SNIPER_IMGSZ_MODE="${SNIPER_IMGSZ_MODE:-}"
SNIPER_IMGSZ_FIXED="${SNIPER_IMGSZ_FIXED:-}"
OUT_JSON="${OUT_JSON:-${PROJECT_ROOT:-.}/arch4_wiring_check.json}"
PRINT_EVERY="${PRINT_EVERY:-25}"
SAVE_EXAMPLES="${SAVE_EXAMPLES:-10}"
SR_WEIGHTS="${SR_WEIGHTS:-}"
YOLO_WEIGHTS_LR="${YOLO_WEIGHTS_LR:-}"
YOLO_WEIGHTS_HR="${YOLO_WEIGHTS_HR:-}"

required=(ARCH4_WIRING_SCRIPT ARCH4_CONFIG LR_IMAGES_DIR HR_LABELS_DIR OUT_JSON)
for var in "${required[@]}"; do
  if [[ -z "${!var}" ]]; then
    echo "[ERROR] $var must be set." >&2
    exit 1
  fi
done

cmd=(
  "$PYTHON_BIN" "$ARCH4_WIRING_SCRIPT"
  --arch4_config "$ARCH4_CONFIG"
  --lr_images_dir "$LR_IMAGES_DIR"
  --hr_labels_dir "$HR_LABELS_DIR"
  --max_images "$MAX_IMAGES"
  --device "$DEVICE"
  --modes "$MODES"
  --eval_space "$EVAL_SPACE"
  --out_json "$OUT_JSON"
  --print_every "$PRINT_EVERY"
  --save_examples "$SAVE_EXAMPLES"
)

if [[ -n "$PROJECT_ROOT" ]]; then
  cmd+=(--project_root "$PROJECT_ROOT")
fi
if [[ -n "$ARCH4_PY" ]]; then
  cmd+=(--arch4_py "$ARCH4_PY")
fi
if [[ -n "$HR_IMAGES_DIR" ]]; then
  cmd+=(--hr_images_dir "$HR_IMAGES_DIR")
fi
if [[ "$HALF_FLAG" == "--half" || "$HALF_FLAG" == "1" ]]; then
  cmd+=(--half)
fi
if [[ -n "$SNIPER_IMGSZ_MODE" ]]; then
  cmd+=(--sniper_imgsz_mode "$SNIPER_IMGSZ_MODE")
fi
if [[ -n "$SNIPER_IMGSZ_FIXED" ]]; then
  cmd+=(--sniper_imgsz_fixed "$SNIPER_IMGSZ_FIXED")
fi
if [[ -n "$SR_WEIGHTS" ]]; then
  cmd+=(--sr_weights "$SR_WEIGHTS")
fi
if [[ -n "$YOLO_WEIGHTS_LR" ]]; then
  cmd+=(--yolo_weights_lr "$YOLO_WEIGHTS_LR")
fi
if [[ -n "$YOLO_WEIGHTS_HR" ]]; then
  cmd+=(--yolo_weights_hr "$YOLO_WEIGHTS_HR")
fi

printf '[RUN] %q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
