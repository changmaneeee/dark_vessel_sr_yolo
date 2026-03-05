#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Build Arch4 Sniper fine-tune dataset from real Arch4 SR crops
# ------------------------------------------------------------
# Usage example:
#   OUT_ROOT=iac_runs/sniper_ft_dataset_a11 \
#   ARCH4_CONFIG=configs/experiment/arch4_roi_awareNMS_eval.yaml \
#   HR_DATA_YAML=/home/changmin/smart_airbus_data/data.yaml \
#   LR_DATA_YAML=/home/changmin/smart_airbus_data_lr/data.yaml \
#   bash run_build_arch4_sniper_dataset.sh
#
# Optional:
#   MAX_IMAGES=2000 MAX_ROIS_PER_IMAGE=8 NEG_KEEP_PROB=0.5 SPLITS="train val"
# ------------------------------------------------------------

PYTHON_BIN=${PYTHON_BIN:-python}
OUT_ROOT=${OUT_ROOT:-iac_runs/sniper_ft_dataset_a11}
ARCH4_CONFIG=${ARCH4_CONFIG:-configs/experiment/arch4_roi_awareNMS_eval.yaml}
HR_DATA_YAML=${HR_DATA_YAML:-/home/changmin/smart_airbus_data/data.yaml}
LR_DATA_YAML=${LR_DATA_YAML:-/home/changmin/smart_airbus_data_lr/data.yaml}
DEVICE=${DEVICE:-cuda}
MAX_IMAGES=${MAX_IMAGES:-0}
MAX_ROIS_PER_IMAGE=${MAX_ROIS_PER_IMAGE:-8}
NEG_KEEP_PROB=${NEG_KEEP_PROB:-0.30}
MIN_BOX_PX=${MIN_BOX_PX:-2.0}
NUM_CLASSES=${NUM_CLASSES:-1}
CLASS_NAMES=${CLASS_NAMES:-ship}
SPLITS=${SPLITS:-"train val"}

mkdir -p "$OUT_ROOT"
LOG_PATH="$OUT_ROOT/build_dataset.log"

echo "[BUILD DATASET] out_root=$OUT_ROOT" | tee "$LOG_PATH"
echo "[BUILD DATASET] config=$ARCH4_CONFIG" | tee -a "$LOG_PATH"

eval $PYTHON_BIN build_arch4_sniper_finetune_dataset.py \
  --arch4_config "$ARCH4_CONFIG" \
  --hr_data_yaml "$HR_DATA_YAML" \
  --lr_data_yaml "$LR_DATA_YAML" \
  --out_root "$OUT_ROOT" \
  --splits $SPLITS \
  --device "$DEVICE" \
  --max_images "$MAX_IMAGES" \
  --max_rois_per_image "$MAX_ROIS_PER_IMAGE" \
  --neg_keep_prob "$NEG_KEEP_PROB" \
  --min_box_px "$MIN_BOX_PX" \
  --num_classes "$NUM_CLASSES" \
  --names "$CLASS_NAMES" \
  2>&1 | tee -a "$LOG_PATH"


echo "[DONE] dataset build complete" | tee -a "$LOG_PATH"
echo "  data yaml : $OUT_ROOT/data.yaml" | tee -a "$LOG_PATH"
echo "  metadata  : $OUT_ROOT/metadata.csv" | tee -a "$LOG_PATH"
