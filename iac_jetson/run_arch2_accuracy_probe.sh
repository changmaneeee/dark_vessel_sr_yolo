#!/usr/bin/env bash
set -euo pipefail

# Quick wrapper for arch2_accuracy_probe.py
#
# Example:
#   PROJECT_ROOT=/path/to/repo \
#   ARCH2_CONFIG=/path/to/repo/configs/experiment/arch2_softgate.yaml \
#   ARCH2_PY=/path/to/repo/src/models/pipelines/arch2_softgate.py \
#   LR_IMAGES_DIR=/path/to/lr/images/val \
#   HR_LABELS_DIR=/path/to/hr/labels/val \
#   MODES=full_blend,thr=0.5 \
#   MAX_IMAGES=500 \
#   DEVICE=cuda \
#   OUT_JSON=/tmp/arch2_accuracy_probe.json \
#   bash run_arch2_accuracy_probe.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_ROOT="${PROJECT_ROOT:-}"
ARCH2_PROBE_SCRIPT="${ARCH2_PROBE_SCRIPT:-${PROJECT_ROOT:+$PROJECT_ROOT/iac_jetson/arch2_accuracy_probe.py}}"
ARCH2_CONFIG="${ARCH2_CONFIG:-}"
ARCH2_PY="${ARCH2_PY:-}"
LR_IMAGES_DIR="${LR_IMAGES_DIR:-}"
HR_LABELS_DIR="${HR_LABELS_DIR:-}"
MAX_IMAGES="${MAX_IMAGES:-500}"
DEVICE="${DEVICE:-cuda}"
HALF_FLAG="${HALF_FLAG:-}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.45}"
MODES="${MODES:-full_blend,thr=0.5}"
OUT_JSON="${OUT_JSON:-${PROJECT_ROOT:-.}/arch2_accuracy_probe.json}"
PRINT_EVERY="${PRINT_EVERY:-50}"
DEBUG_DIR="${DEBUG_DIR:-}"
SAVE_EXAMPLES="${SAVE_EXAMPLES:-0}"
SR_WEIGHTS="${SR_WEIGHTS:-}"
GATE_WEIGHTS="${GATE_WEIGHTS:-}"
YOLO_SR_WEIGHTS="${YOLO_SR_WEIGHTS:-}"
BLEND_SELECTED="${BLEND_SELECTED:-}"

required=(ARCH2_PROBE_SCRIPT ARCH2_CONFIG LR_IMAGES_DIR HR_LABELS_DIR OUT_JSON)
for var in "${required[@]}"; do
  if [[ -z "${!var}" ]]; then
    echo "[ERROR] $var must be set." >&2
    exit 1
  fi
done

cmd=(
  "$PYTHON_BIN" "$ARCH2_PROBE_SCRIPT"
  --arch2_config "$ARCH2_CONFIG"
  --lr_images_dir "$LR_IMAGES_DIR"
  --hr_labels_dir "$HR_LABELS_DIR"
  --max_images "$MAX_IMAGES"
  --device "$DEVICE"
  --conf "$CONF"
  --iou "$IOU"
  --modes "$MODES"
  --out_json "$OUT_JSON"
  --print_every "$PRINT_EVERY"
)

if [[ -n "$PROJECT_ROOT" ]]; then
  cmd+=(--project_root "$PROJECT_ROOT")
fi
if [[ -n "$ARCH2_PY" ]]; then
  cmd+=(--arch2_py "$ARCH2_PY")
fi
if [[ "$HALF_FLAG" == "--half" || "$HALF_FLAG" == "1" ]]; then
  cmd+=(--half)
fi
if [[ -n "$DEBUG_DIR" ]]; then
  cmd+=(--debug_dir "$DEBUG_DIR" --save_examples "$SAVE_EXAMPLES")
fi
if [[ -n "$SR_WEIGHTS" ]]; then
  cmd+=(--sr_weights "$SR_WEIGHTS")
fi
if [[ -n "$GATE_WEIGHTS" ]]; then
  cmd+=(--gate_weights "$GATE_WEIGHTS")
fi
if [[ -n "$YOLO_SR_WEIGHTS" ]]; then
  cmd+=(--yolo_weights "$YOLO_SR_WEIGHTS")
fi
if [[ -n "$BLEND_SELECTED" ]]; then
  cmd+=(--blend_selected "$BLEND_SELECTED")
fi

printf '[RUN] %q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
