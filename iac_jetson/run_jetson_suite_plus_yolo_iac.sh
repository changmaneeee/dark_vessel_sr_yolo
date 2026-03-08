#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Jetson paper suite (Arch0/2/4 + YOLO baselines)
# ============================================================
#
# This is an "extended" version of run_jetson_arch024_suite_iac.sh.
# It adds YOLO-only benchmarks so you can compare:
#   - Baseline LR YOLO
#   - Baseline SR/HR YOLO
#   - (Optional) heavier YOLO weights
# alongside Arch0/2/4.
#
# Required env vars:
#   RUN_TAG
#   SR_WEIGHTS
#   YOLO_SR_WEIGHTS   (used as SR/HR baseline + Arch0/4 Sniper by default)
#   YOLO_LR_WEIGHTS   (baseline LR + Arch4 Scout)
#   GATE_WEIGHTS      (Arch2)
#   ARCH4_BASE_CONFIG (roi-aware deploy yaml recommended)
#   HR_DATA_YAML
#   LR_DATA_YAML
#   LR_IMAGES_DIR
#
# Optional:
#   YOLO_HR_WEIGHTS        (if you have a separate HR-only baseline)
#   YOLO_HEAVY_WEIGHTS     (space-separated list, e.g. "models/yolo8m.pt models/yolo8l.pt")
#   MAX_IMAGES (default 2000), ARCH4_MAX_IMAGES (default 200)
#   DEVICE (default cuda), HALF (default 1), IMG_SZ_LR (default 640)
#
# Output:
#   iac_runs/jetson_suite_plus_yolo/${RUN_TAG}/
#     00_system_snapshot.txt
#     results/*.json
#     summary.tsv
#
# NOTE: To avoid "permission denied", we always call measure_jetson_job.sh via bash.
# NOTE: To avoid sudo password prompts during overnight runs, we use sudo -n (non-interactive),
#       and skip if not permitted.
#

RUN_TAG="${RUN_TAG:-jetson_suite_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT_ROOT="$PROJECT_ROOT/iac_runs/jetson_suite_plus_yolo/${RUN_TAG}"
mkdir -p "$OUT_ROOT/results"

MEASURE_SH="$PROJECT_ROOT/iac_jetson/measure_jetson_job.sh"
SUMMARY_PY="$PROJECT_ROOT/iac_jetson/jetson_job_summary.py"

ARCH0_SCRIPT="${ARCH0_SCRIPT:-$PROJECT_ROOT/iac_scripts/arch0_bench_jetson.py}"
ARCH2_SCRIPT="${ARCH2_SCRIPT:-$PROJECT_ROOT/iac_scripts/arch2_bench_jetson.py}"
ARCH4_SCRIPT="${ARCH4_SCRIPT:-$PROJECT_ROOT/iac_scripts/arch4_eval_ultralytics.py}"
YOLO_BENCH_SCRIPT="${YOLO_BENCH_SCRIPT:-$PROJECT_ROOT/iac_scripts/yolo_bench_jetson.py}"

# Required vars (paths may differ on Jetson)
SR_WEIGHTS="${SR_WEIGHTS:?Missing SR_WEIGHTS}"
YOLO_SR_WEIGHTS="${YOLO_SR_WEIGHTS:?Missing YOLO_SR_WEIGHTS}"
YOLO_LR_WEIGHTS="${YOLO_LR_WEIGHTS:?Missing YOLO_LR_WEIGHTS}"
GATE_WEIGHTS="${GATE_WEIGHTS:?Missing GATE_WEIGHTS}"
ARCH4_BASE_CONFIG="${ARCH4_BASE_CONFIG:?Missing ARCH4_BASE_CONFIG}"
HR_DATA_YAML="${HR_DATA_YAML:?Missing HR_DATA_YAML}"
LR_DATA_YAML="${LR_DATA_YAML:?Missing LR_DATA_YAML}"
LR_IMAGES_DIR="${LR_IMAGES_DIR:?Missing LR_IMAGES_DIR}"

YOLO_HR_WEIGHTS="${YOLO_HR_WEIGHTS:-}"
YOLO_HEAVY_WEIGHTS="${YOLO_HEAVY_WEIGHTS:-}"

MAX_IMAGES="${MAX_IMAGES:-2000}"
ARCH4_MAX_IMAGES="${ARCH4_MAX_IMAGES:-200}"
DEVICE="${DEVICE:-cuda}"
HALF="${HALF:-1}"
IMG_SZ_LR="${IMG_SZ_LR:-640}"

# ---------- Helpers ----------
snapshot () {
  local f="$OUT_ROOT/00_system_snapshot.txt"
  {
    echo "==== DATE ===="
    date
    echo "==== USER ===="
    whoami
    echo "==== HOST ===="
    hostname
    echo "==== PROJECT_ROOT ===="
    echo "$PROJECT_ROOT"
    echo "==== OS ===="
    uname -a
    echo "==== JetPack/L4T ===="
    cat /etc/nv_tegra_release 2>/dev/null || true
    echo "==== nvpmodel -q ===="
    # avoid sudo password prompts
    sudo -n nvpmodel -q 2>/dev/null || nvpmodel -q 2>/dev/null || true
    echo "==== jetson_clocks --show ===="
    sudo -n jetson_clocks --show 2>/dev/null || jetson_clocks --show 2>/dev/null || true
    echo "==== CUDA test ===="
    python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
    echo "==== DISK ===="
    df -h || true
    echo "==== MEM ===="
    free -h || true
  } > "$f"
  echo "[SNAPSHOT] saved: $f"
}

patch_arch4_yaml () {
  # Patch weights into a temp yaml under OUT_ROOT
  local out_yaml="$OUT_ROOT/arch4_patched.yaml"
  python - "$ARCH4_BASE_CONFIG" "$out_yaml" "$SR_WEIGHTS" "$YOLO_LR_WEIGHTS" "$YOLO_SR_WEIGHTS" <<'PY'
import sys, yaml, pathlib
base, out, sr_w, scout_w, sniper_w = sys.argv[1:]
cfg = yaml.safe_load(open(base, "r")) or {}
cfg.setdefault("model", {})
cfg["model"].setdefault("sr", {})
cfg["model"].setdefault("yolo", {})
cfg["model"]["sr"]["weights"] = sr_w
cfg["model"]["yolo"]["weights_lr"] = scout_w
cfg["model"]["yolo"]["weights_hr"] = sniper_w
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
print(out)
PY
  echo "$out_yaml"
}

run_job () {
  local job="$1"
  local metrics_json="$2"
  shift 2
  # measure_jetson_job.sh creates summary json + tegrastats log
  bash "$MEASURE_SH" "$job" "$OUT_ROOT" "$metrics_json" 500 -- "$@"
}

# ---------- Main ----------
snapshot

# (A) YOLO baselines (speed only)
echo "==== [A] YOLO-only baselines ===="
run_job "yolo_lr_bench" "$OUT_ROOT/results/yolo_lr_bench.json" \
  python "$YOLO_BENCH_SCRIPT" \
    --yolo_weights "$YOLO_LR_WEIGHTS" \
    --images_dir "$LR_IMAGES_DIR" \
    --max_images "$MAX_IMAGES" \
    --device "$DEVICE" \
    --imgsz "$IMG_SZ_LR" \
    --conf 0.25 --iou 0.45 --max_det 300 \
    $( [[ "$HALF" == "1" ]] && echo "--half" ) \
    --out_json "$OUT_ROOT/results/yolo_lr_bench.json"

run_job "yolo_sr_bench" "$OUT_ROOT/results/yolo_sr_bench.json" \
  python "$YOLO_BENCH_SCRIPT" \
    --yolo_weights "$YOLO_SR_WEIGHTS" \
    --images_dir "$LR_IMAGES_DIR" \
    --max_images "$MAX_IMAGES" \
    --device "$DEVICE" \
    --imgsz "$IMG_SZ_LR" \
    --conf 0.25 --iou 0.45 --max_det 300 \
    $( [[ "$HALF" == "1" ]] && echo "--half" ) \
    --out_json "$OUT_ROOT/results/yolo_sr_bench.json"

if [[ -n "$YOLO_HR_WEIGHTS" ]]; then
  run_job "yolo_hr_bench" "$OUT_ROOT/results/yolo_hr_bench.json" \
    python "$YOLO_BENCH_SCRIPT" \
      --yolo_weights "$YOLO_HR_WEIGHTS" \
      --images_dir "$LR_IMAGES_DIR" \
      --max_images "$MAX_IMAGES" \
      --device "$DEVICE" \
      --imgsz "$IMG_SZ_LR" \
      --conf 0.25 --iou 0.45 --max_det 300 \
      $( [[ "$HALF" == "1" ]] && echo "--half" ) \
      --out_json "$OUT_ROOT/results/yolo_hr_bench.json"
fi

if [[ -n "$YOLO_HEAVY_WEIGHTS" ]]; then
  i=0
  for w in $YOLO_HEAVY_WEIGHTS; do
    i=$((i+1))
    bn="$(basename "$w")"
    run_job "yolo_heavy_${i}_${bn}" "$OUT_ROOT/results/yolo_heavy_${i}_${bn}.json" \
      python "$YOLO_BENCH_SCRIPT" \
        --yolo_weights "$w" \
        --images_dir "$LR_IMAGES_DIR" \
        --max_images "$MAX_IMAGES" \
        --device "$DEVICE" \
        --imgsz "$IMG_SZ_LR" \
        --conf 0.25 --iou 0.45 --max_det 300 \
        $( [[ "$HALF" == "1" ]] && echo "--half" ) \
        --out_json "$OUT_ROOT/results/yolo_heavy_${i}_${bn}.json"
  done
fi

# (B) Arch0/2 benchmarks
echo "==== [B] Arch0/2 benchmarks ===="
run_job "arch0" "$OUT_ROOT/results/arch0_bench.json" \
  python "$ARCH0_SCRIPT" \
    --sr_weights "$SR_WEIGHTS" \
    --yolo_weights "$YOLO_SR_WEIGHTS" \
    --images_dir "$LR_IMAGES_DIR" \
    --max_images "$MAX_IMAGES" \
    --device "$DEVICE" \
    $( [[ "$HALF" == "1" ]] && echo "--half" ) \
    --out_json "$OUT_ROOT/results/arch0_bench.json"

run_job "arch2" "$OUT_ROOT/results/arch2_bench.json" \
  python "$ARCH2_SCRIPT" \
    --gate_weights "$GATE_WEIGHTS" \
    --sr_weights "$SR_WEIGHTS" \
    --yolo_weights "$YOLO_SR_WEIGHTS" \
    --images_dir "$LR_IMAGES_DIR" \
    --max_images "$MAX_IMAGES" \
    --device "$DEVICE" \
    $( [[ "$HALF" == "1" ]] && echo "--half" ) \
    --out_json "$OUT_ROOT/results/arch2_bench.json"

# (C) Arch4 eval on a small subset (accuracy + speed)
echo "==== [C] Arch4 eval (subset) ===="
PATCHED_ARCH4="$(patch_arch4_yaml)"
run_job "arch4_eval_${ARCH4_MAX_IMAGES}" "$OUT_ROOT/results/arch4_eval.json" \
  python "$ARCH4_SCRIPT" \
    --arch4_config "$PATCHED_ARCH4" \
    --hr_data_yaml "$HR_DATA_YAML" \
    --lr_data_yaml "$LR_DATA_YAML" \
    --eval_space hr \
    --batch 1 \
    --max_images "$ARCH4_MAX_IMAGES" \
    --device "$DEVICE" \
    --out_json "$OUT_ROOT/results/arch4_eval.json"

# Summary table (TSV)
python "$SUMMARY_PY" --run_dir "$OUT_ROOT" --out_tsv "$OUT_ROOT/summary.tsv"
echo
echo "=== DONE ==="
echo "Run dir: $OUT_ROOT"
echo "Summary : $OUT_ROOT/summary.tsv"
