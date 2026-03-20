#!/bin/bash
set -euo pipefail

# Rebuild Arch4 Sniper crop dataset directly on the 4090 PC
# from raw LR/HR datasets instead of copying arch4_sniper_crops.
#
# Usage:
#   bash REBUILD_SNIPER_CROPS_ON_4090.sh <project_root> <lr_root> <hr_root>
#
# Example:
#   bash REBUILD_SNIPER_CROPS_ON_4090.sh \
#     /home/$USER/dark_vessel_sr_yolo \
#     /home/$USER/smart_airbus_data_lr \
#     /home/$USER/smart_airbus_data

if [ "$#" -ne 3 ]; then
  echo "Usage: bash REBUILD_SNIPER_CROPS_ON_4090.sh <project_root> <lr_root> <hr_root>"
  exit 1
fi

PROJECT_ROOT="$1"
LR_ROOT="$2"
HR_ROOT="$3"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dark_vessel

CROP_DATA_DIR="${PROJECT_ROOT}/data/arch4_sniper_crops"
mkdir -p "${PROJECT_ROOT}/data"

AVAIL_GB=$(df -BG "${PROJECT_ROOT}" | awk 'NR==2 {print $4}' | tr -d 'G')
echo "Available disk space: ${AVAIL_GB}GB"
if [ "${AVAIL_GB}" -lt 15 ]; then
  echo "ERROR: Need at least 15GB free under ${PROJECT_ROOT}"
  exit 1
fi

echo "=== Phase 0: validate raw val pairs ==="
VAL_CHECK_DIR="${PROJECT_ROOT}/iac_runs/$(date +%Y%m%d_%H%M%S)_validate_pairs_val_4090"
mkdir -p "${VAL_CHECK_DIR}"

python "${PROJECT_ROOT}/iac_jetson/validate_paired_dataset.py" \
  --images_dir "${LR_ROOT}/images/val" \
  --labels_dir "${HR_ROOT}/labels/val" \
  --out_dir "${VAL_CHECK_DIR}" \
  --allow_empty_labels

echo "=== Phase A-1: train crop dump ==="
python "${PROJECT_ROOT}/iac_jetson/arch4_dump_sniper_crops.py" \
  --project_root "${PROJECT_ROOT}" \
  --arch4_config "${PROJECT_ROOT}/configs/experiment/arch4_roi_awareNMS_deploy.yaml" \
  --arch4_py "${PROJECT_ROOT}/src/models/pipelines/arch4_roi_awareNMS_ablation.py" \
  --lr_images_dir "${LR_ROOT}/images/train" \
  --hr_images_dir "${HR_ROOT}/images/train" \
  --hr_labels_dir "${HR_ROOT}/labels/train" \
  --out_dir "${CROP_DATA_DIR}" \
  --split train \
  --device cuda \
  --half \
  --sr_weights "${PROJECT_ROOT}/weights/rfdn_arch4_model_best.pt" \
  --yolo_weights_lr "${PROJECT_ROOT}/weights/scout_yolo_lr_best.pt" \
  --yolo_weights_hr "${PROJECT_ROOT}/weights/sniper_cropft_best.pt" \
  --print_every 500 \
  --checkpoint_every 10000

echo "=== Phase A-2: val crop dump ==="
python "${PROJECT_ROOT}/iac_jetson/arch4_dump_sniper_crops.py" \
  --project_root "${PROJECT_ROOT}" \
  --arch4_config "${PROJECT_ROOT}/configs/experiment/arch4_roi_awareNMS_deploy.yaml" \
  --arch4_py "${PROJECT_ROOT}/src/models/pipelines/arch4_roi_awareNMS_ablation.py" \
  --lr_images_dir "${LR_ROOT}/images/val" \
  --hr_images_dir "${HR_ROOT}/images/val" \
  --hr_labels_dir "${HR_ROOT}/labels/val" \
  --out_dir "${CROP_DATA_DIR}" \
  --split val \
  --device cuda \
  --half \
  --sr_weights "${PROJECT_ROOT}/weights/rfdn_arch4_model_best.pt" \
  --yolo_weights_lr "${PROJECT_ROOT}/weights/scout_yolo_lr_best.pt" \
  --yolo_weights_hr "${PROJECT_ROOT}/weights/sniper_cropft_best.pt" \
  --print_every 500 \
  --checkpoint_every 10000

echo "=== Done ==="
echo "Crop dataset root: ${CROP_DATA_DIR}"
echo "Stats:"
ls -l "${CROP_DATA_DIR}"/stats*.json 2>/dev/null || true
