#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------
# Run Arch0 / Arch2 / Arch4 on Jetson from iac_jetson/
# -----------------------------------------------------
# Assumes this script lives in: <PROJECT_ROOT>/iac_jetson/
# and PROJECT_ROOT contains src/, iac_scripts/, configs/, weights/, dataset/
#
# Usage example:

<< COMMENT
   RUN_TAG=jetson_suite_run1_fix4 \
   SR_WEIGHTS=/home/octolab/dark_vessel_sr_yolo/models/rfdn/model_best.pt \
   YOLO_SR_WEIGHTS=/home/octolab/dark_vessel_sr_yolo/models/yolo_8s_rfdn/best.pt \
   YOLO_LR_WEIGHTS=/home/octolab/dark_vessel_sr_yolo/models/yolo8s_lr/best.pt \
   GATE_WEIGHTS=/home/octolab/dark_vessel_sr_yolo/models/arch2_gate/gate_best.pt \
   ARCH4_BASE_CONFIG=/home/octolab/dark_vessel_sr_yolo/configs/experiment/arch4_roi_awareNMS_deploy.yaml \
   HR_DATA_YAML=/home/octolab/dark_vessel_sr_yolo/dataset/smart_airbus_data/data.yaml \
   LR_DATA_YAML=/home/octolab/dark_vessel_sr_yolo/dataset/smart_airbus_data_lr/data.yaml \
   LR_IMAGES_DIR=/home/octolab/dark_vessel_sr_yolo/dataset/smart_airbus_data_lr/images/val \
   bash iac_jetson/run_jetson_arch024_suite_iac.sh
COMMENT


IAC_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$IAC_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-jetson_arch024_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/jetson_runs/$RUN_TAG}"
mkdir -p "$OUT_ROOT" "$OUT_ROOT/logs" "$OUT_ROOT/results" "$OUT_ROOT/configs"

# Device/perf
DEVICE="${DEVICE:-cuda}"
MAX_IMAGES="${MAX_IMAGES:-200}"
WARMUP="${WARMUP:-20}"
HALF_FLAG="${HALF_FLAG:---half}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.45}"
TEGR_INTERVAL_MS="${TEGR_INTERVAL_MS:-500}"
USE_JETSON_CLOCKS="${USE_JETSON_CLOCKS:-0}"
NVP_MODE_ID="${NVP_MODE_ID:-}"

# Paths that almost always change on Jetson
HR_DATA_YAML="${HR_DATA_YAML:-$PROJECT_ROOT/dataset/smart_airbus_data/data.yaml}"
LR_DATA_YAML="${LR_DATA_YAML:-$PROJECT_ROOT/dataset/smart_airbus_data_lr/data.yaml}"
LR_IMAGES_DIR="${LR_IMAGES_DIR:-$PROJECT_ROOT/dataset/smart_airbus_data_lr/images/val}"

SR_WEIGHTS="${SR_WEIGHTS:-$PROJECT_ROOT/weights/rfdn/model_best.pt}"
YOLO_SR_WEIGHTS="${YOLO_SR_WEIGHTS:-$PROJECT_ROOT/weights/yolo_8s_rfdn/best.pt}"
YOLO_LR_WEIGHTS="${YOLO_LR_WEIGHTS:-$PROJECT_ROOT/models/yolo8s_lr/best.pt}"
GATE_WEIGHTS="${GATE_WEIGHTS:-$PROJECT_ROOT/training/gate_arch2/checkpoints/gate_gt/gate_best.pt}"

ARCH0_CONFIG="${ARCH0_CONFIG:-$PROJECT_ROOT/configs/experiment/arch0_sequential.yaml}"
ARCH2_CONFIG="${ARCH2_CONFIG:-$PROJECT_ROOT/configs/experiment/arch2_softgate.yaml}"
ARCH4_BASE_CONFIG="${ARCH4_BASE_CONFIG:-$PROJECT_ROOT/configs/experiment/arch4_roi_awareNMS_deploy.yaml}"

ARCH0_SCRIPT="${ARCH0_SCRIPT:-$PROJECT_ROOT/iac_scripts/arch0_bench_jetson.py}"
ARCH2_SCRIPT="${ARCH2_SCRIPT:-$PROJECT_ROOT/iac_scripts/arch2_bench_jetson.py}"
ARCH4_SCRIPT="${ARCH4_SCRIPT:-$PROJECT_ROOT/iac_scripts/arch4_eval_ultralytics.py}"
MEASURE_SH="$IAC_DIR/measure_jetson_job.sh"

snapshot() {
  {
    echo '==== DATE ===='; date
    echo '==== USER ===='; whoami
    echo '==== HOST ===='; hostname
    echo '==== PROJECT_ROOT ===='; echo "$PROJECT_ROOT"
    echo '==== OS ===='; uname -a
    echo '==== JetPack/L4T ===='; cat /etc/nv_tegra_release 2>/dev/null || true
    echo '==== nvpmodel -q ===='; sudo nvpmodel -q 2>/dev/null || nvpmodel -q 2>/dev/null || true
    echo '==== jetson_clocks --show ===='; sudo jetson_clocks --show 2>/dev/null || jetson_clocks --show 2>/dev/null || true
    echo '==== CUDA test ===='; python3 - <<PY
import torch
print('torch:', torch.__version__)
print('cuda_available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
    echo '==== DISK ===='; df -h
    echo '==== MEM ===='; free -h
  } | tee "$OUT_ROOT/00_system_snapshot.txt"
}

apply_perf_mode() {
  if [[ -n "$NVP_MODE_ID" ]]; then
    echo "[INFO] Setting nvpmodel mode: $NVP_MODE_ID" | tee -a "$OUT_ROOT/00_system_snapshot.txt"
    sudo nvpmodel -m "$NVP_MODE_ID" || true
  fi
  if [[ "$USE_JETSON_CLOCKS" == "1" ]]; then
    echo "[INFO] Enabling jetson_clocks" | tee -a "$OUT_ROOT/00_system_snapshot.txt"
    sudo jetson_clocks || true
  fi
}

patch_arch4_config() {
  local dst="$OUT_ROOT/configs/arch4_patched.yaml"
  ARCH4_PATCHED_CONFIG="$dst"
  python3 - <<PY
import yaml
from pathlib import Path
src = Path(r'''$ARCH4_BASE_CONFIG''')
dst = Path(r'''$dst''')
with open(src, 'r') as f:
    cfg = yaml.safe_load(f)
model = cfg.setdefault('model', {})
y = model.setdefault('yolo', {})
sr = model.setdefault('sr', {})
arch4 = model.setdefault('arch4', {})
y['weights_lr'] = r'''$YOLO_LR_WEIGHTS'''
y['weights_hr'] = r'''$YOLO_SR_WEIGHTS'''
y['weights_path'] = r'''$YOLO_SR_WEIGHTS'''
sr['weights'] = r'''$SR_WEIGHTS'''
# deploy defaults if not present
arch4.setdefault('sniper_conf', 0.001)
arch4.setdefault('final_conf', 0.25)
with open(dst, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(dst)
PY
}

append_row() {
  local summary_json="$1"
  python3 - <<PY >> "$OUT_ROOT/suite_summary.tsv"
import json
p = r'''$summary_json'''
with open(p, 'r') as f:
    d = json.load(f)
job = d['job_name']
m = d.get('metrics', {})
pw = d.get('power', {})
der = d.get('derived', {})
cols = [
    job,
    str(m.get('precision')),
    str(m.get('recall')),
    str(m.get('map50')),
    str(m.get('map5095')),
    str(m.get('tp50')),
    str(m.get('fp50')),
    str(m.get('fn50')),
    str(m.get('precision50_direct')),
    str(m.get('recall50_direct')),
    str(m.get('avg_ms_per_image')),
    str(m.get('fps')),
    str(pw.get('power_source')),
    str(pw.get('avg_power_mw')),
    str(pw.get('max_power_mw')),
    str(der.get('energy_per_image_j')),
    p,
]
print("\t".join(cols))
PY
}

run_arch0() {
  local out_json="$OUT_ROOT/results/arch0_bench.json"
  "$MEASURE_SH" arch0 "$OUT_ROOT/logs" "$out_json" "$TEGR_INTERVAL_MS" -- \
    "$PYTHON_BIN" "$ARCH0_SCRIPT" \
      --arch0_config "$ARCH0_CONFIG" \
      --sr_weights "$SR_WEIGHTS" \
      --yolo_weights "$YOLO_SR_WEIGHTS" \
      --images_dir "$LR_IMAGES_DIR" \
      --max_images "$MAX_IMAGES" \
      --warmup "$WARMUP" \
      --device "$DEVICE" \
      $HALF_FLAG \
      --conf "$CONF" \
      --iou "$IOU" \
      --out_json "$out_json"
  append_row "$OUT_ROOT/logs/arch0.summary.json"
}

run_arch2() {
  local out_json="$OUT_ROOT/results/arch2_bench.json"
  "$MEASURE_SH" arch2 "$OUT_ROOT/logs" "$out_json" "$TEGR_INTERVAL_MS" -- \
    "$PYTHON_BIN" "$ARCH2_SCRIPT" \
      --arch2_config "$ARCH2_CONFIG" \
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
      --out_json "$out_json"
  append_row "$OUT_ROOT/logs/arch2.summary.json"
}

run_arch4() {
  patch_arch4_config
  local out_json="$OUT_ROOT/results/arch4_eval.json"
  "$MEASURE_SH" arch4 "$OUT_ROOT/logs" "$out_json" "$TEGR_INTERVAL_MS" -- \
    "$PYTHON_BIN" "$ARCH4_SCRIPT" \
      --arch4_config "$ARCH4_PATCHED_CONFIG" \
      --hr_data_yaml "$HR_DATA_YAML" \
      --lr_data_yaml "$LR_DATA_YAML" \
      --eval_space hr \
      --batch 1 \
      --max_images "$MAX_IMAGES" \
      --device "$DEVICE" \
      --out_json "$out_json"
  append_row "$OUT_ROOT/logs/arch4.summary.json"
}

main() {
  snapshot
  apply_perf_mode
  printf "job_name\tprecision\trecall\tmap50\tmap5095\ttp50\tfp50\tfn50\tprecision50_direct\trecall50_direct\tavg_ms_per_image\tfps\tpower_source\tavg_power_mw\tmax_power_mw\tenergy_per_image_j\tsummary_json\n" > "$OUT_ROOT/suite_summary.tsv"

  run_arch0
  run_arch2
  run_arch4

  echo "[DONE] suite_summary.tsv -> $OUT_ROOT/suite_summary.tsv"
  echo "[DONE] system snapshot   -> $OUT_ROOT/00_system_snapshot.txt"
}

main "$@"
