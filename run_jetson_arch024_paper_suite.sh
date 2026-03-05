#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------
# Jetson paper suite for Arch0 / Arch2 / Arch4
# ---------------------------------------------
# 목적:
# - Jetson에서 arch0, arch2, arch4를 각각 실행
# - tegrastats로 전력 로그 수집
# - 결과 JSON + 전력 로그를 합쳐 Joule/image 계산
# - 논문용 표에 바로 넣기 쉬운 summary.tsv 생성
#
# 사용 전 반드시 아래 경로 변수만 점검하세요.
# ---------------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-jetson_arch024_paper_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/jetson_runs/$RUN_TAG}"
mkdir -p "$OUT_ROOT" "$OUT_ROOT/logs" "$OUT_ROOT/results" "$OUT_ROOT/configs"

# ---------- Jetson system settings ----------
TEGR_INTERVAL_MS="${TEGR_INTERVAL_MS:-500}"
NVP_MODE_ID="${NVP_MODE_ID:-}"          # 예: 0 (device별 다를 수 있으니 비워두면 변경 안 함)
USE_JETSON_CLOCKS="${USE_JETSON_CLOCKS:-0}" # 1이면 sudo jetson_clocks 실행

# ---------- Dataset / device ----------
DEVICE="${DEVICE:-cuda}"
MAX_IMAGES="${MAX_IMAGES:-200}"
WARMUP="${WARMUP:-20}"
HALF_FLAG="${HALF_FLAG:---half}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.45}"

HR_DATA_YAML="${HR_DATA_YAML:-/home/octolab/dark_vessel_sr_yolo/dataset/smart_airbus_data/data.yaml}"
LR_DATA_YAML="${LR_DATA_YAML:-/home/octolab/dark_vessel_sr_yolo/dataset/smart_airbus_data_lr/data.yaml}"
LR_IMAGES_DIR="${LR_IMAGES_DIR:-/home/octolab/dark_vessel_sr_yolo/dataset/smart_airbus_data_lr/images/val}"

# ---------- Common weights ----------
SCOUT_WEIGHTS="${SCOUT_WEIGHTS:-/home/octolab/dark_vessel_sr_yolo/models/yolo8s_lr/best.pt}"
SNIPER_WEIGHTS="${SNIPER_WEIGHTS:-/home/octolab/dark_vessel_sr_yolo/models/yolo8s_hr/best.pt}"
SR_SNIPER_WEIGHTS="${SR_SNIPER_WEIGHTS:-$SNIPER_WEIGHTS}"
SR_WEIGHTS="${SR_WEIGHTS:-/home/octolab/dark_vessel_sr_yolo/models/rfdn/model_best.pt}"
GATE_WEIGHTS="${GATE_WEIGHTS:-/home/octolab/dark_vessel_sr_yolo/training/gate_arch2/checkpoints/gate_gt/gate_best.pt}"

# ---------- Configs / scripts ----------
ARCH0_CONFIG="${ARCH0_CONFIG:-$PROJECT_ROOT/configs/experiment/arch0_sequential.yaml}"
ARCH2_CONFIG="${ARCH2_CONFIG:-$PROJECT_ROOT/configs/experiment/arch2_softgate.yaml}"
ARCH4_CONFIG="${ARCH4_CONFIG:-$PROJECT_ROOT/configs/experiment/arch4_run037_like_deploy.yaml}"

ARCH0_SCRIPT="${ARCH0_SCRIPT:-$PROJECT_ROOT/iac_scripts/arch0_bench_jetson.py}"
ARCH2_SCRIPT="${ARCH2_SCRIPT:-$PROJECT_ROOT/iac_scripts/arch2_bench_jetson.py}"
ARCH4_SCRIPT="${ARCH4_SCRIPT:-$PROJECT_ROOT/iac_scripts/arch4_eval_ultralytics.py}"

# ---------- Helpers ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEASURE_SH="$SCRIPT_DIR/measure_jetson_job.sh"

snapshot() {
  {
    echo '==== DATE ===='; date
    echo '==== USER ===='; whoami
    echo '==== HOST ===='; hostname
    echo '==== OS ===='; uname -a
    echo '==== nv_tegra_release ===='; cat /etc/nv_tegra_release 2>/dev/null || true
    echo '==== nvpmodel -q ===='; sudo nvpmodel -q 2>/dev/null || nvpmodel -q 2>/dev/null || true
    echo '==== jetson_clocks --show ===='; sudo jetson_clocks --show 2>/dev/null || jetson_clocks --show 2>/dev/null || true
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

copy_and_patch_arch4_config() {
  local dst="$OUT_ROOT/configs/arch4_patched.yaml"
  ARCH4_PATCHED_CONFIG="$dst"
  python3 - <<PY
import yaml
from pathlib import Path
src = Path(r'''$ARCH4_CONFIG''')
dst = Path(r'''$dst''')
with open(src, 'r') as f:
    cfg = yaml.safe_load(f)
y = cfg.setdefault('model', {}).setdefault('yolo', {})
y['weights_lr'] = r'''$SCOUT_WEIGHTS'''
y['weights_hr'] = r'''$SR_SNIPER_WEIGHTS'''
with open(dst, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(dst)
PY
}

append_row() {
  local summary_json="$1"
  python3 - <<PY >> "$OUT_ROOT/suite_summary.tsv"
import json, sys
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
      --yolo_weights "$SNIPER_WEIGHTS" \
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
      --yolo_weights "$SNIPER_WEIGHTS" \
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
  copy_and_patch_arch4_config
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
}

main "$@"
