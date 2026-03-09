#!/usr/bin/env bash
set -euo pipefail

# Quick wrapper for arch4_crop_mode_probe.py
#
# Example:
<< COMMENT
   PROJECT_ROOT=/home/octolab/dark_vessel_sr_yolo \
   ARCH4_CONFIG=/home/octolab/dark_vessel_sr_yolo/configs/experiment/arch4_roi_awareNMS_deploy.yaml \
   ARCH4_PY=/home/octolab/dark_vessel_sr_yolo/src/models/pipelines/arch4_roi_awareNMS.py \
   LR_IMAGES_DIR=/home/octolab/dark_vessel_sr_yolo/dataset/smart_airbus_data_lr/images/val \
   HR_IMAGES_DIR=/home/octolab/dark_vessel_sr_yolo/dataset/smart_airbus_data/images/val \
   HR_LABELS_DIR=/home/octolab/dark_vessel_sr_yolo/dataset/smart_airbus_data/labels/val \
   MODES=sr,bilinear,hr_ref \
   MAX_IMAGES=200 \
   DEVICE=cuda \
   OUT_JSON=/home/octolab/dark_vessel_sr_yolo/iac_runs/jetson_runs/arch4_0308.json \
   bash jetson_runs/arch4_0308/run_arch4_crop_mode_probe.sh
COMMENT

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_ROOT="${PROJECT_ROOT:-}"
ARCH4_PROBE_SCRIPT="${ARCH4_PROBE_SCRIPT:-${PROJECT_ROOT:+$PROJECT_ROOT/jetson_runs/arch4_0308/arch4_crop_mode_probe.py}}"
ARCH4_CONFIG="${ARCH4_CONFIG:-}"
ARCH4_PY="${ARCH4_PY:-}"
LR_IMAGES_DIR="${LR_IMAGES_DIR:-}"
HR_IMAGES_DIR="${HR_IMAGES_DIR:-}"
HR_LABELS_DIR="${HR_LABELS_DIR:-}"
MAX_IMAGES="${MAX_IMAGES:-200}"
DEVICE="${DEVICE:-cuda}"
HALF_FLAG="${HALF_FLAG:-}"
MODES="${MODES:-sr,bilinear,hr_ref}"
EVAL_SPACE="${EVAL_SPACE:-hr}"
SNIPER_IMGSZ_MODE="${SNIPER_IMGSZ_MODE:-}"
SNIPER_IMGSZ_FIXED="${SNIPER_IMGSZ_FIXED:-}"
OUT_JSON="${OUT_JSON:-${PROJECT_ROOT:-.}/arch4_crop_mode_probe.json}"
PRINT_EVERY="${PRINT_EVERY:-25}"
SAVE_EXAMPLES="${SAVE_EXAMPLES:-0}"
DEBUG_DIR="${DEBUG_DIR:-}"
SR_WEIGHTS="${SR_WEIGHTS:-}"
YOLO_WEIGHTS_LR="${YOLO_WEIGHTS_LR:-}"
YOLO_WEIGHTS_HR="${YOLO_WEIGHTS_HR:-}"

required=(ARCH4_PROBE_SCRIPT ARCH4_CONFIG LR_IMAGES_DIR HR_IMAGES_DIR HR_LABELS_DIR OUT_JSON)
for var in "${required[@]}"; do
  if [[ -z "${!var}" ]]; then
    echo "[ERROR] $var must be set." >&2
    exit 1
  fi
done

cmd=(
  "$PYTHON_BIN" "$ARCH4_PROBE_SCRIPT"
  --arch4_config "$ARCH4_CONFIG"
  --lr_images_dir "$LR_IMAGES_DIR"
  --hr_images_dir "$HR_IMAGES_DIR"
  --hr_labels_dir "$HR_LABELS_DIR"
  --max_images "$MAX_IMAGES"
  --device "$DEVICE"
  --modes "$MODES"
  --eval_space "$EVAL_SPACE"
  --out_json "$OUT_JSON"
  --print_every "$PRINT_EVERY"
)

if [[ -n "$PROJECT_ROOT" ]]; then
  cmd+=(--project_root "$PROJECT_ROOT")
fi
if [[ -n "$ARCH4_PY" ]]; then
  cmd+=(--arch4_py "$ARCH4_PY")
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
if [[ -n "$DEBUG_DIR" ]]; then
  cmd+=(--debug_dir "$DEBUG_DIR" --save_examples "$SAVE_EXAMPLES")
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
