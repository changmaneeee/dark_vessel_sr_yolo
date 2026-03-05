#!/usr/bin/env bash
set -euo pipefail

<< COMMENT
EXP_TAG=arch4_2x2_full \
OLD_SCOUT_WEIGHTS=/home/changmin/dark_vessel_sr_yolo/weights/yolo_lr/8s/best.pt \
OLD_SNIPER_WEIGHTS=/home/changmin/dark_vessel_sr_yolo/weights/yolohr/8s/best.pt \
SR_SNIPER_WEIGHTS=/home/changmin/dark_vessel_sr_yolo/weights/yolo_8s_rfdn/best.pt \
HR_DATA_YAML=/home/changmin/smart_airbus_data/data.yaml \
LR_DATA_YAML=/home/changmin/smart_airbus_data_lr/data.yaml \
bash run_arch4_2x2_ablation.sh

COMMENT


# -----------------------------------------------------------------------------
# Arch4 2x2 ablation runner
# -----------------------------------------------------------------------------
# Goal:
#   Factorize the gains of current Arch4 into:
#   (A) ROI-aware NMS effect
#   (B) SR-domain Sniper effect
#
# 4 cells:
#   1) old Sniper + old merge        = A00_old_old
#   2) old Sniper + ROI-aware NMS    = A01_old_roi
#   3) SR Sniper  + old merge        = A10_sr_old
#   4) SR Sniper  + ROI-aware NMS    = A11_sr_roi
#
# Assumptions:
#   - You already have:
#       src/models/pipelines/arch4_adaptive.py
#       src/models/pipelines/arch4_roi_awareNMS.py
#       iac_scripts/arch4_eval_ultralytics.py
#   - The eval script currently works for one pipeline import.
#   - This script creates TEMP eval copies and patches pipeline import/class.
#
# Usage example:
#   EXP_TAG=arch4_2x2_full \
#   OLD_SCOUT_WEIGHTS=/abs/path/lr_scout.pt \
#   OLD_SNIPER_WEIGHTS=/abs/path/hr_sniper.pt \
#   SR_SNIPER_WEIGHTS=/abs/path/rfdn_sr_yolo500.pt \
#   HR_DATA_YAML=/home/changmin/smart_airbus_data/data.yaml \
#   LR_DATA_YAML=/home/changmin/smart_airbus_data_lr/data.yaml \
#   bash run_arch4_2x2_ablation.sh
#
# Optional:
#   MAX_IMAGES=200   # 0 or unset means full validation
#   DEVICE=cuda
#   BATCH=1
# -----------------------------------------------------------------------------

EXP_TAG="${EXP_TAG:-arch4_2x2_ablation}"
DEVICE="${DEVICE:-cuda}"
BATCH="${BATCH:-1}"
MAX_IMAGES="${MAX_IMAGES:-0}"

export HR_DATA_YAML="${HR_DATA_YAML:-/home/changmin/smart_airbus_data/data.yaml}"
export LR_DATA_YAML="${LR_DATA_YAML:-/home/changmin/smart_airbus_data_lr/data.yaml}"

export OLD_SCOUT_WEIGHTS="${OLD_SCOUT_WEIGHTS:-/home/changmin/yolov8s+airbus_smartdata/weights/best.pt}"
export OLD_SNIPER_WEIGHTS="${OLD_SNIPER_WEIGHTS:-/home/changmin/yolov8s+HR_airbus_smartdata/weights/best.pt}"
export SR_SNIPER_WEIGHTS="${SR_SNIPER_WEIGHTS:?Please set SR_SNIPER_WEIGHTS=/abs/path/to/rfdn_sr_yolo500.pt}"
export SR_WEIGHTS="${SR_WEIGHTS:-/home/changmin/dark_vessel_sr_yolo/weights/rfdn/model_best.pt}"

export BASE_OLD_CFG="${BASE_OLD_CFG:-configs/experiment/arch4_stage2_balanced_best.yaml}"
export BASE_ROI_CFG="${BASE_ROI_CFG:-configs/experiment/arch4_roi_awareNMS_eval.yaml}"
export BASE_EVAL_SCRIPT="${BASE_EVAL_SCRIPT:-iac_scripts/arch4_eval_ultralytics.py}"

export OUT_ROOT="iac_runs/${EXP_TAG}"
export CFG_DIR="configs/experiment/${EXP_TAG}"
export LOG_DIR="${OUT_ROOT}/logs"
export RES_DIR="${OUT_ROOT}/results"
export TMP_DIR="${OUT_ROOT}/tmp"

mkdir -p "$OUT_ROOT" "$CFG_DIR" "$LOG_DIR" "$RES_DIR" "$TMP_DIR"

# -----------------------------------------------------------------------------
# helper: patch eval script import/class for pipeline mode
# -----------------------------------------------------------------------------
python - <<'PY'
from pathlib import Path
import os, re

src = Path(os.environ['BASE_EVAL_SCRIPT'])
out_old = Path(os.environ['TMP_DIR']) / 'arch4_eval_oldmerge.py'
out_roi = Path(os.environ['TMP_DIR']) / 'arch4_eval_roi.py'
text = src.read_text()

# Create OLD MERGE version
text_old = text
text_old = re.sub(r'from\s+src\.models\.pipelines\.arch4_roi_awareNMS\s+import\s+Arch4RoiAwareNMS',
                  'from src.models.pipelines.arch4_adaptive import Arch4Adaptive', text_old)
text_old = re.sub(r'from\s+src\.models\.pipelines\.arch4_adaptive\s+import\s+Arch4Adaptive',
                  'from src.models.pipelines.arch4_adaptive import Arch4Adaptive', text_old)
text_old = re.sub(r'Arch4RoiAwareNMS\(', 'Arch4Adaptive(', text_old)
text_old = re.sub(r'pipeline\s*=\s*Arch4Adaptive\(', 'pipeline = Arch4Adaptive(', text_old)
out_old.write_text(text_old)

# Create ROI version
text_roi = text
text_roi = re.sub(r'from\s+src\.models\.pipelines\.arch4_adaptive\s+import\s+Arch4Adaptive',
                  'from src.models.pipelines.arch4_roi_awareNMS import Arch4RoiAwareNMS', text_roi)
text_roi = re.sub(r'from\s+src\.models\.pipelines\.arch4_roi_awareNMS\s+import\s+Arch4RoiAwareNMS',
                  'from src.models.pipelines.arch4_roi_awareNMS import Arch4RoiAwareNMS', text_roi)
text_roi = re.sub(r'Arch4Adaptive\(', 'Arch4RoiAwareNMS(', text_roi)
text_roi = re.sub(r'pipeline\s*=\s*Arch4RoiAwareNMS\(', 'pipeline = Arch4RoiAwareNMS(', text_roi)
out_roi.write_text(text_roi)

print('patched:', out_old)
print('patched:', out_roi)
PY

# -----------------------------------------------------------------------------
# helper: patch config YAMLs
# -----------------------------------------------------------------------------
python - <<'PY'
import os, yaml, copy
from pathlib import Path

cfg_dir = Path(os.environ['CFG_DIR'])
base_old = Path(os.environ['BASE_OLD_CFG'])
base_roi = Path(os.environ['BASE_ROI_CFG'])

old_scout = os.environ['OLD_SCOUT_WEIGHTS']
old_sniper = os.environ['OLD_SNIPER_WEIGHTS']
sr_sniper = os.environ['SR_SNIPER_WEIGHTS']
sr_weights = os.environ['SR_WEIGHTS']

# best ROI-aware params from run_037
best = {
    'pass1_conf': 0.0075,
    'high_conf': 0.45,
    'pass2_conf': 0.45,
    'sniper_conf': 0.001,
    'final_conf': 0.25,
    'merge_iou': 0.5,
    'roi_expansion': 1.75,
    'crop_size_lr': 64,
    'batch_size_sr': 32,
    'scout_nms_iou': 0.55,
    'roi_merge_iou': 0.25,
    'roi_center_ratio': 0.35,
    'sniper_nms_iou': 0.45,
    'final_nms_iou': 0.45,
    'drop_uncertain_if_sniper_hits': True,
    'sniper_score_bonus': 0.0,
}

def load(p):
    with open(p,'r') as f:
        return yaml.safe_load(f)

def save(obj,p):
    with open(p,'w') as f:
        yaml.safe_dump(obj,f,sort_keys=False)

def patch_common(cfg, sniper_path):
    cfg = copy.deepcopy(cfg)
    cfg.setdefault('data', {})
    cfg.setdefault('model', {})
    cfg['data']['upscale_factor'] = cfg['data'].get('upscale_factor', 4)
    cfg['model'].setdefault('sr', {})
    cfg['model']['sr']['type'] = cfg['model']['sr'].get('type', 'rfdn')
    cfg['model']['sr']['weights'] = sr_weights
    cfg['model']['sr'].setdefault('rfdn', {})
    cfg['model']['sr']['rfdn']['nf'] = cfg['model']['sr']['rfdn'].get('nf', 50)
    cfg['model']['sr']['rfdn']['num_modules'] = cfg['model']['sr']['rfdn'].get('num_modules', 4)
    cfg['model'].setdefault('yolo', {})
    cfg['model']['yolo']['weights_lr'] = old_scout
    cfg['model']['yolo']['weights_hr'] = sniper_path
    cfg['model']['yolo']['classes'] = cfg['model']['yolo'].get('classes', 1)
    cfg['model']['yolo']['num_classes'] = cfg['model']['yolo'].get('num_classes', cfg['model']['yolo']['classes'])
    cfg['model'].setdefault('arch4', {})
    for k,v in best.items():
        cfg['model']['arch4'][k] = v
    return cfg

# A00 old sniper + old merge
cfg = patch_common(load(base_old), old_sniper)
save(cfg, cfg_dir / 'A00_old_old.yaml')

# A01 old sniper + ROI-aware NMS
cfg = patch_common(load(base_roi), old_sniper)
save(cfg, cfg_dir / 'A01_old_roi.yaml')

# A10 SR sniper + old merge
cfg = patch_common(load(base_old), sr_sniper)
save(cfg, cfg_dir / 'A10_sr_old.yaml')

# A11 SR sniper + ROI-aware NMS
cfg = patch_common(load(base_roi), sr_sniper)
save(cfg, cfg_dir / 'A11_sr_roi.yaml')

print('saved 4 configs in', cfg_dir)
PY

# -----------------------------------------------------------------------------
# helper: run one evaluation
# -----------------------------------------------------------------------------
run_one() {
  local tag="$1"
  local script_py="$2"
  local cfg="$3"
  local out_json="$4"
  local log_txt="$5"

  echo "[RUN] ${tag}"
  local cmd=(python "$script_py" \
    --arch4_config "$cfg" \
    --hr_data_yaml "$HR_DATA_YAML" \
    --lr_data_yaml "$LR_DATA_YAML" \
    --eval_space hr \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --out_json "$out_json")

  if [[ "$MAX_IMAGES" != "0" && -n "$MAX_IMAGES" ]]; then
    cmd+=(--max_images "$MAX_IMAGES")
  fi

  printf '%q ' "${cmd[@]}" > "$log_txt"
  printf '\n\n' >> "$log_txt"
  "${cmd[@]}" >> "$log_txt" 2>&1
}

OLD_EVAL_PY="$TMP_DIR/arch4_eval_oldmerge.py"
ROI_EVAL_PY="$TMP_DIR/arch4_eval_roi.py"

run_one "A00_old_old" "$OLD_EVAL_PY" "$CFG_DIR/A00_old_old.yaml" "$RES_DIR/A00_old_old.json" "$LOG_DIR/A00_old_old.log"
run_one "A01_old_roi" "$ROI_EVAL_PY" "$CFG_DIR/A01_old_roi.yaml" "$RES_DIR/A01_old_roi.json" "$LOG_DIR/A01_old_roi.log"
run_one "A10_sr_old" "$OLD_EVAL_PY" "$CFG_DIR/A10_sr_old.yaml" "$RES_DIR/A10_sr_old.json" "$LOG_DIR/A10_sr_old.log"
run_one "A11_sr_roi" "$ROI_EVAL_PY" "$CFG_DIR/A11_sr_roi.yaml" "$RES_DIR/A11_sr_roi.json" "$LOG_DIR/A11_sr_roi.log"

# -----------------------------------------------------------------------------
# summarize
# -----------------------------------------------------------------------------
python - <<'PY'
import json, os, csv
from pathlib import Path

res_dir = Path(os.environ['RES_DIR'])
out_csv = Path(os.environ['OUT_ROOT']) / 'arch4_2x2_summary.csv'
out_json = Path(os.environ['OUT_ROOT']) / 'arch4_2x2_summary.json'

rows = []
for name in ['A00_old_old','A01_old_roi','A10_sr_old','A11_sr_roi']:
    p = res_dir / f'{name}.json'
    d = json.load(open(p))
    rd = d['runs'][0]['results_dict']
    meta = d.get('meta', {})
    rows.append({
        'tag': name,
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
        'source_json': str(p),
    })

with open(out_csv,'w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

with open(out_json,'w') as f:
    json.dump({'rows': rows}, f, indent=2)

print('saved:', out_csv)
print('saved:', out_json)
for r in rows:
    print(r)
PY

echo
echo "DONE: $OUT_ROOT"
echo "See: $OUT_ROOT/arch4_2x2_summary.csv"
