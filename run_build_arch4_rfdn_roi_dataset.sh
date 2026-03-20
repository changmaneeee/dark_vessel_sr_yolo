#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/changmin/dark_vessel_sr_yolo}"
ARCH4_CONFIG="${ARCH4_CONFIG:-$PROJECT_ROOT/configs/experiment/arch4_roi_awareNMS_eval.yaml}"
HR_DATA_YAML="${HR_DATA_YAML:-/home/changmin/smart_airbus_data/data.yaml}"
LR_DATA_YAML="${LR_DATA_YAML:-/home/changmin/smart_airbus_data_lr/data.yaml}"
DEVICE="${DEVICE:-cuda}"
MAX_IMAGES="${MAX_IMAGES:-0}"
MAX_ROIS_PER_IMAGE="${MAX_ROIS_PER_IMAGE:-0}"
NEG_KEEP_PROB="${NEG_KEEP_PROB:-0.30}"
ROI_EXPANSION_OVERRIDE="${ROI_EXPANSION_OVERRIDE:-0}"
CROP_SIZE_LR_OVERRIDE="${CROP_SIZE_LR_OVERRIDE:-0}"
UNCERTAIN_MIN_CONF="${UNCERTAIN_MIN_CONF:-}"
UNCERTAIN_MAX_CONF="${UNCERTAIN_MAX_CONF:-}"

RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/iac_runs/$(date +%Y%m%d_%H%M%S)_arch4_rfdn_roi_dataset}"
mkdir -p "$RUN_ROOT"

CMD=(
  python "$PROJECT_ROOT/build_arch4_rfdn_roi_dataset.py"
  --arch4_config "$ARCH4_CONFIG"
  --hr_data_yaml "$HR_DATA_YAML"
  --lr_data_yaml "$LR_DATA_YAML"
  --out_root "$RUN_ROOT"
  --splits train val
  --device "$DEVICE"
  --max_images "$MAX_IMAGES"
  --max_rois_per_image "$MAX_ROIS_PER_IMAGE"
  --neg_keep_prob "$NEG_KEEP_PROB"
  --roi_expansion_override "$ROI_EXPANSION_OVERRIDE"
  --crop_size_lr_override "$CROP_SIZE_LR_OVERRIDE"
)

if [[ -n "$UNCERTAIN_MIN_CONF" ]]; then
  CMD+=(--uncertain_min_conf "$UNCERTAIN_MIN_CONF")
fi

if [[ -n "$UNCERTAIN_MAX_CONF" ]]; then
  CMD+=(--uncertain_max_conf "$UNCERTAIN_MAX_CONF")
fi

printf '[RUN] %q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "[DONE] out_root=$RUN_ROOT"
