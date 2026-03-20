#!/bin/bash
set -euo pipefail

source /home/changmin/miniconda3/etc/profile.d/conda.sh
conda activate dark_vessel

PROJECT_ROOT="/home/changmin/dark_vessel_sr_yolo"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${PROJECT_ROOT}/iac_runs/${TIMESTAMP}_scout_retrain_v2"
mkdir -p "${RUN_DIR}"

BASE_WEIGHTS="${PROJECT_ROOT}/weights/yolo_lr/8s/best.pt"
DATA_YAML="/home/changmin/smart_airbus_data_lr/data.yaml"
PROJECT_OUT="${PROJECT_ROOT}/weights/yolo_lr_improved"
NAME="${TIMESTAMP}_8s_aug_v2"
LOG_FILE="${RUN_DIR}/scout_retrain.log"

echo "========================================" | tee -a "${LOG_FILE}"
echo "Scout Retraining v2 Started" | tee -a "${LOG_FILE}"
echo "Run dir: ${RUN_DIR}" | tee -a "${LOG_FILE}"
echo "Base weights: ${BASE_WEIGHTS}" | tee -a "${LOG_FILE}"
echo "Data: ${DATA_YAML}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

python "${PROJECT_ROOT}/iac_jetson/train_scout_yolo.py" \
  --data "${DATA_YAML}" \
  --base_weights "${BASE_WEIGHTS}" \
  --imgsz 640 \
  --epochs 100 \
  --batch 16 \
  --patience 20 \
  --optimizer AdamW \
  --lr0 0.0005 \
  --lrf 0.01 \
  --warmup_epochs 5 \
  --mosaic 1.0 \
  --mixup 0.15 \
  --copy_paste 0.10 \
  --project "${PROJECT_OUT}" \
  --name "${NAME}" \
  --device 0 \
  --workers 0 \
  --save_period 10 \
  --amp false | tee -a "${LOG_FILE}"

echo "========================================" | tee -a "${LOG_FILE}"
echo "Scout Retraining v2 Complete" | tee -a "${LOG_FILE}"
echo "Weights dir: ${PROJECT_OUT}/${NAME}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
