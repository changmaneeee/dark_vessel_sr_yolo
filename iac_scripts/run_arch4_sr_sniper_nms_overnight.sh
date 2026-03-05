#!/usr/bin/env bash
set -euo pipefail


<< COMMENT

SR_SNIPER_WEIGHTS=/home/changmin/dark_vessel_sr_yolo/weights/yolo_8s_rfdn/best.pt \
EXP_TAG=arch4_sr_sniper_nms_overnight \
bash run_arch4_sr_sniper_nms_overnight.sh

COMMENT



# =========================================================
# Arch4 SR-Sniper + ROI-aware NMS overnight sweep
# 1) Patch base configs to use SR-domain Sniper detector
# 2) Run a baseline eval/deploy check
# 3) Sweep ROI-aware NMS parameters on deploy-like setting
# =========================================================

PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-cuda}
HR_DATA_YAML=${HR_DATA_YAML:-/home/changmin/smart_airbus_data/data.yaml}
LR_DATA_YAML=${LR_DATA_YAML:-/home/changmin/smart_airbus_data_lr/data.yaml}
ARCH4_EVAL_SCRIPT=${ARCH4_EVAL_SCRIPT:-iac_scripts/arch4_eval_ultralytics.py}

BASE_EVAL_CFG=${BASE_EVAL_CFG:-configs/experiment/arch4_roi_awareNMS_eval.yaml}
BASE_DEPLOY_CFG=${BASE_DEPLOY_CFG:-configs/experiment/arch4_roi_awareNMS_deploy.yaml}

# REQUIRED: detector trained on RFDN-SR images
SR_SNIPER_WEIGHTS=${SR_SNIPER_WEIGHTS:-}
if [[ -z "$SR_SNIPER_WEIGHTS" ]]; then
  echo "[ERROR] SR_SNIPER_WEIGHTS is empty."
  echo "Example:"
  echo "  SR_SNIPER_WEIGHTS=/ABS/PATH/TO/rfdn_yolo_epoch500.pt nohup bash $0 > overnight.log 2>&1 &"
  exit 1
fi

# Baseline sanity runs before sweep
BASELINE_EVAL_MAX_IMAGES=${BASELINE_EVAL_MAX_IMAGES:-200}
BASELINE_DEPLOY_MAX_IMAGES=${BASELINE_DEPLOY_MAX_IMAGES:-200}

# Sweep setting: tune on deploy-like mode by default
TUNE_MAX_IMAGES=${TUNE_MAX_IMAGES:-200}
EXP_TAG=${EXP_TAG:-arch4_sr_sniper_nms_overnight}

# Sweep grids (space-separated)
SCOUT_NMS_LIST=${SCOUT_NMS_LIST:-"0.45 0.50 0.55"}
ROI_MERGE_LIST=${ROI_MERGE_LIST:-"0.25 0.30 0.35"}
FINAL_NMS_LIST=${FINAL_NMS_LIST:-"0.45 0.50 0.55"}
SNIPER_BONUS_LIST=${SNIPER_BONUS_LIST:-"0.0 0.05"}

# fixed parameters for this sweep
DROP_UNCERTAIN_IF_SNIPER_HITS=${DROP_UNCERTAIN_IF_SNIPER_HITS:-true}
SNIPER_NMS_IOU=${SNIPER_NMS_IOU:-0.45}
ROI_CENTER_RATIO=${ROI_CENTER_RATIO:-0.35}

RUN_DIR="iac_runs/${EXP_TAG}"
CFG_DIR="configs/experiment/${EXP_TAG}"
LOG_DIR="${RUN_DIR}/logs"
RESULT_DIR="${RUN_DIR}/results"
mkdir -p "$CFG_DIR" "$LOG_DIR" "$RESULT_DIR"

BASELINE_EVAL_CFG="${CFG_DIR}/arch4_sr_sniper_eval.yaml"
BASELINE_DEPLOY_CFG="${CFG_DIR}/arch4_sr_sniper_deploy.yaml"
BASELINE_EVAL_JSON="${RESULT_DIR}/baseline_eval_${BASELINE_EVAL_MAX_IMAGES}.json"
BASELINE_DEPLOY_JSON="${RESULT_DIR}/baseline_deploy_${BASELINE_DEPLOY_MAX_IMAGES}.json"
MANIFEST_TSV="${RUN_DIR}/manifest.tsv"
CSV_SUMMARY="${RUN_DIR}/summary.csv"
JSON_SUMMARY="${RUN_DIR}/summary.json"

# ---------------------------------------------------------
# 1) Patch base configs to use SR-domain Sniper detector
# ---------------------------------------------------------
$PYTHON_BIN - <<PY
import yaml
from pathlib import Path

for src, dst in [
    (Path(${BASE_EVAL_CFG@Q}), Path(${BASELINE_EVAL_CFG@Q})),
    (Path(${BASE_DEPLOY_CFG@Q}), Path(${BASELINE_DEPLOY_CFG@Q})),
]:
    with open(src, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault('model', {}).setdefault('yolo', {})
    cfg['model']['yolo']['weights_hr'] = ${SR_SNIPER_WEIGHTS@Q}
    with open(dst, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"saved: {dst}")
PY

# ---------------------------------------------------------
# 2) Baseline sanity runs (quick)
# ---------------------------------------------------------
run_eval () {
  local cfg_path="$1"
  local out_json="$2"
  local max_images="$3"
  local log_path="$4"

  local -a cmd=(
    "$PYTHON_BIN" "$ARCH4_EVAL_SCRIPT"
    --arch4_config "$cfg_path"
    --hr_data_yaml "$HR_DATA_YAML"
    --lr_data_yaml "$LR_DATA_YAML"
    --eval_space hr
    --batch 1
    --device "$DEVICE"
    --out_json "$out_json"
  )
  if [[ "$max_images" != "0" ]]; then
    cmd+=(--max_images "$max_images")
  fi
  echo "[CMD] ${cmd[*]}" | tee "$log_path"
  "${cmd[@]}" | tee -a "$log_path"
}

echo "[BASELINE] Eval-like SR-Sniper check"
run_eval "$BASELINE_EVAL_CFG" "$BASELINE_EVAL_JSON" "$BASELINE_EVAL_MAX_IMAGES" "$LOG_DIR/baseline_eval.log"

echo "[BASELINE] Deploy-like SR-Sniper check"
run_eval "$BASELINE_DEPLOY_CFG" "$BASELINE_DEPLOY_JSON" "$BASELINE_DEPLOY_MAX_IMAGES" "$LOG_DIR/baseline_deploy.log"

# ---------------------------------------------------------
# 3) Build manifest for NMS sweep (deploy-like tuning)
# ---------------------------------------------------------
echo -e "run_id\tscout_nms_iou\troi_merge_iou\tfinal_nms_iou\tsniper_score_bonus\tcfg\tjson\tlog" > "$MANIFEST_TSV"
run_id=0
for scout_nms in $SCOUT_NMS_LIST; do
  for roi_merge in $ROI_MERGE_LIST; do
    for final_nms in $FINAL_NMS_LIST; do
      for bonus in $SNIPER_BONUS_LIST; do
        run_id=$((run_id + 1))
        run_name=$(printf "run_%03d_scout%s_roi%s_final%s_bonus%s" "$run_id" "$scout_nms" "$roi_merge" "$final_nms" "$bonus")
        cfg_path="${CFG_DIR}/${run_name}.yaml"
        json_path="${RESULT_DIR}/${run_name}.json"
        log_path="${LOG_DIR}/${run_name}.log"

        $PYTHON_BIN - <<PY
import yaml
from pathlib import Path
src = Path(${BASELINE_DEPLOY_CFG@Q})
dst = Path(${cfg_path@Q})
with open(src, 'r') as f:
    cfg = yaml.safe_load(f)
arch4 = cfg.setdefault('model', {}).setdefault('arch4', {})
arch4['scout_nms_iou'] = float(${scout_nms@Q})
arch4['roi_merge_iou'] = float(${roi_merge@Q})
arch4['final_nms_iou'] = float(${final_nms@Q})
arch4['sniper_score_bonus'] = float(${bonus@Q})
arch4['drop_uncertain_if_sniper_hits'] = ${DROP_UNCERTAIN_IF_SNIPER_HITS@Q}.lower() == 'true'
arch4['sniper_nms_iou'] = float(${SNIPER_NMS_IOU@Q})
arch4['roi_center_ratio'] = float(${ROI_CENTER_RATIO@Q})
with open(dst, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"saved: {dst}")
PY

        echo -e "${run_id}\t${scout_nms}\t${roi_merge}\t${final_nms}\t${bonus}\t${cfg_path}\t${json_path}\t${log_path}" >> "$MANIFEST_TSV"
      done
    done
  done
done

echo "[SWEEP] manifest: ${MANIFEST_TSV}"
echo "[SWEEP] total runs: $run_id"

# ---------------------------------------------------------
# 4) Execute sweep sequentially
# ---------------------------------------------------------
tail -n +2 "$MANIFEST_TSV" | while IFS=$'\t' read -r rid scout_nms roi_merge final_nms bonus cfg_path json_path log_path; do
  echo "================================================================================"
  echo "[${rid}/${run_id}] scout=${scout_nms}, roi=${roi_merge}, final=${final_nms}, bonus=${bonus}"
  echo "================================================================================"
  run_eval "$cfg_path" "$json_path" "$TUNE_MAX_IMAGES" "$log_path"
done

# ---------------------------------------------------------
# 5) Build summary CSV/JSON
# ---------------------------------------------------------
$PYTHON_BIN - <<PY
import csv, json
from pathlib import Path

manifest = Path(${MANIFEST_TSV@Q})
csv_out = Path(${CSV_SUMMARY@Q})
json_out = Path(${JSON_SUMMARY@Q})

rows = []
with open(manifest, 'r') as f:
    next(f)  # header
    for line in f:
        rid, scout_nms, roi_merge, final_nms, bonus, cfg_path, json_path, log_path = line.strip().split('\t')
        p = Path(json_path)
        row = {
            'run_id': int(rid),
            'scout_nms_iou': float(scout_nms),
            'roi_merge_iou': float(roi_merge),
            'final_nms_iou': float(final_nms),
            'sniper_score_bonus': float(bonus),
            'config_path': cfg_path,
            'json_path': json_path,
            'log_path': log_path,
            'status': 'ok' if p.exists() else 'missing',
        }
        if p.exists():
            with open(p, 'r') as jf:
                data = json.load(jf)
            rd = data['runs'][0]['results_dict']
            meta = data.get('meta', {})
            row.update({
                'precision': rd.get('metrics/precision(B)'),
                'recall': rd.get('metrics/recall(B)'),
                'map50': rd.get('metrics/mAP50(B)'),
                'map5095': rd.get('metrics/mAP50-95(B)'),
                'tp50': rd.get('direct/tp50'),
                'fp50': rd.get('direct/fp50'),
                'fn50': rd.get('direct/fn50'),
                'precision50_direct': rd.get('direct/precision50'),
                'recall50_direct': rd.get('direct/recall50'),
                'avg_ms_per_image': meta.get('avg_ms_per_image'),
            })
        rows.append(row)

# recall-first, then fp, then mAP, then speed
rows_sorted = sorted(
    [r for r in rows if r['status'] == 'ok'],
    key=lambda r: (
        -(r['recall50_direct'] if r.get('recall50_direct') is not None else -1),
        (r['fp50'] if r.get('fp50') is not None else 10**9),
        -(r['map5095'] if r.get('map5095') is not None else -1),
        (r['avg_ms_per_image'] if r.get('avg_ms_per_image') is not None else 10**9),
    )
)

fieldnames = [
    'run_id', 'scout_nms_iou', 'roi_merge_iou', 'final_nms_iou', 'sniper_score_bonus',
    'precision', 'recall', 'map50', 'map5095',
    'tp50', 'fp50', 'fn50', 'precision50_direct', 'recall50_direct', 'avg_ms_per_image',
    'status', 'config_path', 'json_path', 'log_path'
]
with open(csv_out, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows_sorted:
        writer.writerow(r)
    for r in rows:
        if r['status'] != 'ok':
            writer.writerow(r)

summary = {
    'baseline_eval_json': ${BASELINE_EVAL_JSON@Q},
    'baseline_deploy_json': ${BASELINE_DEPLOY_JSON@Q},
    'top10': rows_sorted[:10],
    'all_runs': rows,
}
with open(json_out, 'w') as f:
    json.dump(summary, f, indent=2)

print('\n[TOP 10] recall-first ranking')
for i, r in enumerate(rows_sorted[:10], 1):
    print(
        f"{i:02d}. run={r['run_id']} | scout={r['scout_nms_iou']} roi={r['roi_merge_iou']} "
        f"final={r['final_nms_iou']} bonus={r['sniper_score_bonus']} | "
        f"R50={r['recall50_direct']:.4f} FP50={r['fp50']} FN50={r['fn50']} "
        f"mAP95={r['map5095']:.4f} ms={r['avg_ms_per_image']:.2f}"
    )
print(f"\nCSV: {csv_out}")
print(f"JSON: {json_out}")
PY

echo "\n[DONE] Overnight run finished. Results in: ${RUN_DIR}"
