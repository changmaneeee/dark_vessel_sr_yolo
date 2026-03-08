#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------
# Run only Arch2 selective-skip benchmark on Jetson/local
# -----------------------------------------------------
# This wrapper is intentionally narrow so you can iterate on Arch2 quickly
# without re-running Arch0/Arch4.
#
# Required env / overrides:
#   PROJECT_ROOT, SR_WEIGHTS, GATE_WEIGHTS, YOLO_SR_WEIGHTS, LR_IMAGES_DIR
# Optional:
#   ARCH2_CONFIG, ARCH2_PY, RUN_TAG, DEVICE, MAX_IMAGES, WARMUP,
#   CONF, IOU, HALF_FLAG, ARCH2_GATE_THRESHOLD, ARCH2_BLEND_SELECTED,
#   DISABLE_SELECTIVE, TEGR_INTERVAL_MS

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-arch2_selective_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/jetson_runs/$RUN_TAG}"
mkdir -p "$OUT_ROOT" "$OUT_ROOT/logs" "$OUT_ROOT/results"

DEVICE="${DEVICE:-cuda}"
MAX_IMAGES="${MAX_IMAGES:-200}"
WARMUP="${WARMUP:-20}"
HALF_FLAG="${HALF_FLAG:---half}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.45}"
TEGR_INTERVAL_MS="${TEGR_INTERVAL_MS:-500}"

LR_IMAGES_DIR="${LR_IMAGES_DIR:-$PROJECT_ROOT/dataset/smart_airbus_data_lr/images/val}"
SR_WEIGHTS="${SR_WEIGHTS:-$PROJECT_ROOT/weights/rfdn/model_best.pt}"
YOLO_SR_WEIGHTS="${YOLO_SR_WEIGHTS:-$PROJECT_ROOT/weights/yolo_8s_rfdn/best.pt}"
GATE_WEIGHTS="${GATE_WEIGHTS:-$PROJECT_ROOT/training/gate_arch2/checkpoints/gate_gt/gate_best.pt}"

ARCH2_CONFIG="${ARCH2_CONFIG:-$PROJECT_ROOT/configs/experiment/arch2_softgate.yaml}"
ARCH2_PY="${ARCH2_PY:-$PROJECT_ROOT/src/models/pipelines/arch2_softgate.py}"
ARCH2_SCRIPT="${ARCH2_SCRIPT:-$SCRIPT_DIR/arch2_bench_selective_skip.py}"
MEASURE_SH="${MEASURE_SH:-$SCRIPT_DIR/measure_jetson_job.sh}"

ARCH2_GATE_THRESHOLD="${ARCH2_GATE_THRESHOLD:-0.5}"
ARCH2_BLEND_SELECTED="${ARCH2_BLEND_SELECTED:-0}"
DISABLE_SELECTIVE="${DISABLE_SELECTIVE:-0}"

OUT_JSON="$OUT_ROOT/results/arch2_bench.json"

EXTRA_ARGS=()
if [[ "$ARCH2_BLEND_SELECTED" == "1" ]]; then
  EXTRA_ARGS+=(--blend_selected)
fi
if [[ "$DISABLE_SELECTIVE" == "1" ]]; then
  EXTRA_ARGS+=(--disable_selective)
fi

"$MEASURE_SH" arch2_selective "$OUT_ROOT/logs" "$OUT_JSON" "$TEGR_INTERVAL_MS" -- \
  "$PYTHON_BIN" "$ARCH2_SCRIPT" \
    --project_root "$PROJECT_ROOT" \
    --arch2_config "$ARCH2_CONFIG" \
    --arch2_py "$ARCH2_PY" \
    --sr_weights "$SR_WEIGHTS" \
    --gate_weights "$GATE_WEIGHTS" \
    --yolo_weights "$YOLO_SR_WEIGHTS" \
    --images_dir "$LR_IMAGES_DIR" \
    --max_images "$MAX_IMAGES" \
    --warmup "$WARMUP" \
    --device "$DEVICE" \
    $HALF_FLAG \
    --conf "$CONF" \
    --iou "$IOU" \
    --gate_threshold "$ARCH2_GATE_THRESHOLD" \
    --out_json "$OUT_JSON" \
    "${EXTRA_ARGS[@]}"

SUMMARY_JSON="$OUT_ROOT/logs/arch2_selective.summary.json"
if [[ -f "$SUMMARY_JSON" ]]; then
  echo "[DONE] summary json -> $SUMMARY_JSON"
fi

echo "[DONE] metrics json -> $OUT_JSON"
