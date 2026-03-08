#!/usr/bin/env bash
set -euo pipefail

# =========================
# Arch4 (ROI-aware NMS) conf sweep for a new Sniper weight (PC)
# =========================
#
# What this does
#  1) Takes a "base" Arch4 YAML (usually deploy.yaml or eval.yaml)
#  2) Patches only a few fields (sniper weights + conf thresholds)
#  3) Runs iac_scripts/arch4_eval_ultralytics.py repeatedly
#  4) Collects a single CSV summary (easy to paste into Notion)
#
# Required:
#   - SNIPER_WEIGHTS : path to finetuned sniper best.pt
#   - ARCH4_BASE_YAML: base yaml to patch (roi-aware deploy/eval yaml)
#   - HR_DATA_YAML   : HR dataset yaml (for labels)
#   - LR_DATA_YAML   : LR dataset yaml (for images)
#
# Optional:
#   - RUN_TAG        : name suffix (default: date-based)
#   - MAX_IMAGES     : subset size (default: 200)
#   - DEVICE         : cuda / cpu (default: cuda)
#   - EVAL_SPACE     : hr or lr (default: hr)
#   - PASS1_CONF     : override scout pass1_conf (default: keep base yaml)
#   - HIGH_CONF      : override pass2_conf/high_conf (default: keep base yaml)
#   - SNIPER_CONF    : override sniper_conf (default: 0.001)
#   - FINAL_CONF_LIST: whitespace list (default: "0.05 0.1 0.15 0.2 0.25 0.3")
#
# Example:
#   SNIPER_WEIGHTS=runs/detect/.../weights/best.pt \
#   ARCH4_BASE_YAML=configs/experiment/arch4_roi_awareNMS_deploy.yaml \
#   HR_DATA_YAML=/home/changmin/smart_airbus_data/data.yaml \
#   LR_DATA_YAML=/home/changmin/smart_airbus_data_lr/data.yaml \
#   RUN_TAG=sniper_ft_e250 \
#   MAX_IMAGES=200 \
#   DEVICE=cuda \
#   bash iac_scripts/pc_arch4_sniper_ft_conf_sweep.sh
#

<< COMMENT
export SNIPER_WEIGHTS=/home/changmin/dark_vessel_sr_yolo/runs/detect/iac_runs/sniper_finetune_train/sniper_ft_a11_e100_safe13/weights/best.pt

export ARCH4_BASE_YAML=configs/experiment/arch4_roi_awareNMS_deploy.yaml
export HR_DATA_YAML=/home/changmin/smart_airbus_data/data.yaml
export LR_DATA_YAML=/home/changmin/smart_airbus_data_lr/data.yaml

export RUN_TAG=sniper_ft_e250
export MAX_IMAGES=200
export DEVICE=cuda

# deploy conf 후보(원하는대로 바꿔도 됨)
export FINAL_CONF_LIST="0.10 0.15 0.20 0.25 0.30"

bash iac_scripts/pc_arch4_sniper_ft_conf_sweep.sh

COMMENT



SNIPER_WEIGHTS="${SNIPER_WEIGHTS:-}"
ARCH4_BASE_YAML="${ARCH4_BASE_YAML:-}"
HR_DATA_YAML="${HR_DATA_YAML:-}"
LR_DATA_YAML="${LR_DATA_YAML:-}"

if [[ -z "$SNIPER_WEIGHTS" || -z "$ARCH4_BASE_YAML" || -z "$HR_DATA_YAML" || -z "$LR_DATA_YAML" ]]; then
  echo "[ERR] Missing required env vars."
  echo "  SNIPER_WEIGHTS=... ARCH4_BASE_YAML=... HR_DATA_YAML=... LR_DATA_YAML=..."
  exit 1
fi

RUN_TAG="${RUN_TAG:-sniper_ft_$(date +%Y%m%d_%H%M%S)}"
MAX_IMAGES="${MAX_IMAGES:-200}"
DEVICE="${DEVICE:-cuda}"
EVAL_SPACE="${EVAL_SPACE:-hr}"

SNIPER_CONF="${SNIPER_CONF:-0.001}"
FINAL_CONF_LIST="${FINAL_CONF_LIST:-0.05 0.1 0.15 0.2 0.25 0.3}"

OUT_DIR="iac_runs/arch4_sniper_ft_sweep/${RUN_TAG}"
mkdir -p "$OUT_DIR/yamls" "$OUT_DIR/json"

CSV="$OUT_DIR/summary.csv"
echo "tag,pass1_conf,high_conf,sniper_conf,final_conf,precision,recall,map50,map50_95,avg_ms_per_image,source_json,patched_yaml" > "$CSV"

patch_yaml () {
  local base_yaml="$1"
  local out_yaml="$2"
  local final_conf="$3"

  python - "$base_yaml" "$out_yaml" "$SNIPER_WEIGHTS" "$SNIPER_CONF" "$final_conf" "${PASS1_CONF:-}" "${HIGH_CONF:-}" <<'PY'
import sys, yaml, pathlib
base_yaml, out_yaml, sniper_w, sniper_conf, final_conf, pass1_conf, high_conf = sys.argv[1:]
cfg = yaml.safe_load(open(base_yaml, "r")) or {}

# Ensure tree
cfg.setdefault("model", {})
cfg["model"].setdefault("yolo", {})
cfg["model"].setdefault("arch4", {})

# 1) Replace sniper weights (HR YOLO)
cfg["model"]["yolo"]["weights_hr"] = sniper_w

# 2) sniper_conf + final_conf (important!)
cfg["model"]["arch4"]["sniper_conf"] = float(sniper_conf)
cfg["model"]["arch4"]["final_conf"] = float(final_conf)

# Optional overrides
if pass1_conf:
    cfg["model"]["arch4"]["pass1_conf"] = float(pass1_conf)
if high_conf:
    # some yml uses high_conf alias, some uses pass2_conf
    cfg["model"]["arch4"]["pass2_conf"] = float(high_conf)
    cfg["model"]["arch4"]["high_conf"] = float(high_conf)

pathlib.Path(out_yaml).parent.mkdir(parents=True, exist_ok=True)
yaml.safe_dump(cfg, open(out_yaml, "w"), sort_keys=False)
print(f"[PATCH] wrote {out_yaml}")
PY
}

extract_metrics () {
  local json_path="$1"
  python - "$json_path" <<'PY'
import sys, json
p=sys.argv[1]
j=json.load(open(p))
rd=j["runs"][0]["results_dict"]
meta=j.get("meta", {})
avg=meta.get("avg_ms_per_image", None)
print(rd.get("metrics/precision(B)", 0.0),
      rd.get("metrics/recall(B)", 0.0),
      rd.get("metrics/mAP50(B)", 0.0),
      rd.get("metrics/mAP50-95(B)", 0.0),
      avg if avg is not None else "")
PY
}

echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] Base YAML: $ARCH4_BASE_YAML"
echo "[INFO] Sniper weights: $SNIPER_WEIGHTS"

for FINAL_CONF in $FINAL_CONF_LIST; do
  TAG="final_${FINAL_CONF}"
  PATCHED_YAML="$OUT_DIR/yamls/arch4_${TAG}.yaml"
  OUT_JSON="$OUT_DIR/json/arch4_${TAG}.json"

  patch_yaml "$ARCH4_BASE_YAML" "$PATCHED_YAML" "$FINAL_CONF"

  echo "[RUN] $TAG"
  python iac_scripts/arch4_eval_ultralytics.py \
    --arch4_config "$PATCHED_YAML" \
    --hr_data_yaml "$HR_DATA_YAML" \
    --lr_data_yaml "$LR_DATA_YAML" \
    --eval_space "$EVAL_SPACE" \
    --batch 1 \
    --max_images "$MAX_IMAGES" \
    --device "$DEVICE" \
    --out_json "$OUT_JSON"

  read P R M50 M5095 AVGMS < <(extract_metrics "$OUT_JSON")

  # best effort: record pass1/high actually used (from yaml)
  PASS1_USED="$(python -c "import yaml; c=yaml.safe_load(open('$PATCHED_YAML')); print(c.get('model',{}).get('arch4',{}).get('pass1_conf',''))")"
  HIGH_USED="$(python -c "import yaml; c=yaml.safe_load(open('$PATCHED_YAML')); a=c.get('model',{}).get('arch4',{}); print(a.get('pass2_conf', a.get('high_conf','')))")"

  echo "$TAG,$PASS1_USED,$HIGH_USED,$SNIPER_CONF,$FINAL_CONF,$P,$R,$M50,$M5095,$AVGMS,$OUT_JSON,$PATCHED_YAML" >> "$CSV"
done

echo
echo "=== DONE ==="
echo "Saved CSV: $CSV"
