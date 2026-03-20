#!/bin/bash
set -euo pipefail

source /home/changmin/miniconda3/etc/profile.d/conda.sh
conda activate dark_vessel

PROJECT_ROOT="/home/changmin/dark_vessel_sr_yolo"
DIAG_DIR="${PROJECT_ROOT}/iac_runs/$(date +%Y%m%d)_scout_diagnostic"
mkdir -p "${DIAG_DIR}"

echo "=== Scout Recall Diagnostic ==="

python "${PROJECT_ROOT}/iac_jetson/scout_recall_diagnostic.py" \
  --project_root "${PROJECT_ROOT}" \
  --scout_weights "${PROJECT_ROOT}/weights/yolo_lr/8s/best.pt" \
  --lr_images_dir /home/changmin/smart_airbus_data_lr/images/val \
  --hr_labels_dir /home/changmin/smart_airbus_data/labels/val \
  --upscale_factor 4.0 \
  --scout_conf 0.0075 \
  --match_iou 0.5 \
  --device cuda \
  --out_json "${DIAG_DIR}/scout_recall_conf00075.json"

echo ""
echo "=== 추가: Scout conf=0.05 ==="

python "${PROJECT_ROOT}/iac_jetson/scout_recall_diagnostic.py" \
  --project_root "${PROJECT_ROOT}" \
  --scout_weights "${PROJECT_ROOT}/weights/yolo_lr/8s/best.pt" \
  --lr_images_dir /home/changmin/smart_airbus_data_lr/images/val \
  --hr_labels_dir /home/changmin/smart_airbus_data/labels/val \
  --upscale_factor 4.0 \
  --scout_conf 0.05 \
  --match_iou 0.5 \
  --device cuda \
  --out_json "${DIAG_DIR}/scout_recall_conf005.json"

echo ""
echo "=== 추가: Scout conf=0.1 ==="

python "${PROJECT_ROOT}/iac_jetson/scout_recall_diagnostic.py" \
  --project_root "${PROJECT_ROOT}" \
  --scout_weights "${PROJECT_ROOT}/weights/yolo_lr/8s/best.pt" \
  --lr_images_dir /home/changmin/smart_airbus_data_lr/images/val \
  --hr_labels_dir /home/changmin/smart_airbus_data/labels/val \
  --upscale_factor 4.0 \
  --scout_conf 0.1 \
  --match_iou 0.5 \
  --device cuda \
  --out_json "${DIAG_DIR}/scout_recall_conf01.json"

echo ""
echo "=== 추가: current config + match_iou=0.3 ==="

python "${PROJECT_ROOT}/iac_jetson/scout_recall_diagnostic.py" \
  --project_root "${PROJECT_ROOT}" \
  --scout_weights "${PROJECT_ROOT}/weights/yolo_lr/8s/best.pt" \
  --lr_images_dir /home/changmin/smart_airbus_data_lr/images/val \
  --hr_labels_dir /home/changmin/smart_airbus_data/labels/val \
  --upscale_factor 4.0 \
  --scout_conf 0.0075 \
  --match_iou 0.3 \
  --device cuda \
  --out_json "${DIAG_DIR}/scout_recall_conf00075_miou030.json"

echo ""
echo "=== 결과 ==="
for f in "${DIAG_DIR}"/*.json; do
    echo "--- $(basename "$f") ---"
    python - <<'PY' "$f"
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    d = json.load(f)
print(f"  Scout recall: {d['scout_recall_at_iou50']:.4f}")
print(f"  GT found: {d['gt_found_by_scout']} / {d['total_gt']}")
print(f"  GT missed: {d['gt_missed_by_scout']}")
print(f"  Score dist: {json.dumps(d['matched_score_distribution'])}")
print(f"  Score mean: {d['matched_score_mean']:.4f}")
PY
done

echo ""
echo "=== 완료 ==="
echo "결과 디렉토리: ${DIAG_DIR}"
