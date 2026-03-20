#!/bin/bash
set -u -o pipefail

# ==========================================================================
# Arch4 Overnight Optimization Pipeline (corrected for current codebase)
# Notes
# - Uses current best reference: interp_a03
# - Uses EMA-aware interpolation helper
# - Uses stable training params observed on this PC: batch=16, workers=0, amp=false
# - Uses hard-negative manifest oversampling, not naive negative subsampling
# - Realistic runtime with these stable defaults is closer to ~12-16h than 9-11h
# ==========================================================================

source /home/changmin/miniconda3/etc/profile.d/conda.sh
conda activate dark_vessel

PROJECT_ROOT="/home/changmin/dark_vessel_sr_yolo"
HELPER="${PROJECT_ROOT}/iac_jetson/arch4_overnight_helper.py"
WIRING="${PROJECT_ROOT}/iac_jetson/arch4_wiring_check.py"
ARCH4_PY="${PROJECT_ROOT}/src/models/pipelines/arch4_roi_awareNMS_ablation.py"
ARCH4_BASE_CONFIG="${PROJECT_ROOT}/iac_runs/20260318_arch4_interp_eval/arch4_sizecond_interp_a03.yaml"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${PROJECT_ROOT}/iac_runs/${TIMESTAMP}_overnight_optimization"
mkdir -p "${RUN_DIR}"

# Core assets
CURRENT_BEST_SNIPER="${PROJECT_ROOT}/weights/yolo_sniper_interp/interp_a03.pt"
CROPFT_WEIGHTS="${PROJECT_ROOT}/weights/yolo_sniper_crop/yolo8s_rfdn_crop_ft_e100_w0/weights/best.pt"
HARDNEG_WEIGHTS="${PROJECT_ROOT}/weights/yolo_sniper_hardneg/20260318_023553_arch4_hardneg_ft/weights/best.pt"
INTERP_DIR="${PROJECT_ROOT}/weights/yolo_sniper_interp"

SR_WEIGHTS="${PROJECT_ROOT}/weights/rfdn_arch4/model_best.pt"
SCOUT_WEIGHTS="${PROJECT_ROOT}/weights/yolo_lr/8s/best.pt"
CROP_DATA_DIR="${PROJECT_ROOT}/data/arch4_sniper_crops"

LR_VAL="/home/changmin/smart_airbus_data_lr/images/val"
HR_VAL="/home/changmin/smart_airbus_data/images/val"
HR_LABELS="/home/changmin/smart_airbus_data/labels/val"
HR_DATA_YAML="${PROJECT_ROOT}/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_hr_data.yaml"
LR_DATA_YAML="${PROJECT_ROOT}/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_lr_data.yaml"

# Stable training defaults tuned from previous failures
TRAIN_EPOCHS="${TRAIN_EPOCHS:-12}"
TRAIN_BATCH="${TRAIN_BATCH:-16}"
TRAIN_WORKERS="${TRAIN_WORKERS:-0}"
TRAIN_AMP="${TRAIN_AMP:-false}"
TRAIN_LR0="${TRAIN_LR0:-0.0005}"
TRAIN_PATIENCE="${TRAIN_PATIENCE:-6}"

HARDNEG_THRESH="${HARDNEG_THRESH:-0.25}"
TARGET_NEG_RATIO_50="${TARGET_NEG_RATIO_50:-0.50}"
TARGET_NEG_RATIO_30="${TARGET_NEG_RATIO_30:-0.30}"
MAX_REPEATS_50="${MAX_REPEATS_50:-12}"
MAX_REPEATS_30="${MAX_REPEATS_30:-6}"

FAIL_LOG="${RUN_DIR}/failures.log"
touch "${FAIL_LOG}"

echo "========================================"
echo "Overnight Optimization Pipeline Started"
echo "Run dir: ${RUN_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "Current best sniper: ${CURRENT_BEST_SNIPER}"
echo "Stable training: epochs=${TRAIN_EPOCHS} batch=${TRAIN_BATCH} workers=${TRAIN_WORKERS} amp=${TRAIN_AMP}"
echo "========================================"

have_file() {
  [ -f "$1" ]
}

log_fail() {
  echo "[FAIL] $1" | tee -a "${FAIL_LOG}"
}

run_probe() {
    local CONFIG="$1"
    local SNIPER_W="$2"
    local OUT_JSON="$3"
    python "${WIRING}" \
        --project_root "${PROJECT_ROOT}" \
        --arch4_config "${CONFIG}" \
        --arch4_py "${ARCH4_PY}" \
        --lr_images_dir "${LR_VAL}" \
        --hr_images_dir "${HR_VAL}" \
        --hr_labels_dir "${HR_LABELS}" \
        --max_images 0 \
        --device cuda \
        --half \
        --modes sr \
        --sniper_imgsz_mode fixed \
        --sniper_imgsz_fixed 256 \
        --sr_weights "${SR_WEIGHTS}" \
        --yolo_weights_lr "${SCOUT_WEIGHTS}" \
        --yolo_weights_hr "${SNIPER_W}" \
        --out_json "${OUT_JSON}" \
        --print_every 500 \
        --save_examples 0
}

extract_f1() {
    python "${HELPER}" extract --json "$1" --mode f1
}

extract_prf() {
    python "${HELPER}" extract --json "$1" --mode prf
}

CURRENT_CONFIG="${RUN_DIR}/current_best_config.yaml"
python "${HELPER}" patch-config \
  --base "${ARCH4_BASE_CONFIG}" \
  --out "${CURRENT_CONFIG}" \
  --set "model.yolo.weights_hr=${CURRENT_BEST_SNIPER}" >/dev/null

CURRENT_BEST_JSON=""
CURRENT_BEST_F1="0.7238"

# =========================================================================
# PHASE 1: Alpha finer grid
# =========================================================================
echo ""
echo "========================================"
echo "PHASE 1: Alpha Finer Grid"
echo "========================================"

PHASE1_DIR="${RUN_DIR}/phase1_alpha_grid"
mkdir -p "${PHASE1_DIR}"

for ALPHA in 0.10 0.15 0.20 0.25 0.35 0.40; do
    TAG="a${ALPHA/./}"
    WEIGHT_OUT="${INTERP_DIR}/interp_${TAG}.pt"
    CONFIG_OUT="${PHASE1_DIR}/${TAG}.yaml"
    JSON_OUT="${PHASE1_DIR}/${TAG}.json"
    echo "--- alpha=${ALPHA} ---"

    if ! python "${HELPER}" interpolate \
      --ckpt-a "${CROPFT_WEIGHTS}" \
      --ckpt-b "${HARDNEG_WEIGHTS}" \
      --alpha "${ALPHA}" \
      --out "${WEIGHT_OUT}" >/dev/null; then
      log_fail "phase1 interpolate ${ALPHA}"
      continue
    fi

    python "${HELPER}" patch-config \
      --base "${ARCH4_BASE_CONFIG}" \
      --out "${CONFIG_OUT}" \
      --set "model.yolo.weights_hr=${WEIGHT_OUT}" >/dev/null

    if run_probe "${CONFIG_OUT}" "${WEIGHT_OUT}" "${JSON_OUT}"; then
      echo "  Result: $(extract_prf "${JSON_OUT}")"
    else
      log_fail "phase1 probe ${ALPHA}"
    fi
done

# include existing a03 in best-alpha comparison
BEST_PHASE1_JSON=$(python "${HELPER}" choose-best \
  --glob "${PHASE1_DIR}/*.json" \
  --glob "${PROJECT_ROOT}/iac_runs/20260318_arch4_interp_eval/arch4_interp_a03_direct_full6418.json" 2>/dev/null || true)

if [ -n "${BEST_PHASE1_JSON}" ] && [ -f "${BEST_PHASE1_JSON}" ]; then
  CURRENT_BEST_JSON="${BEST_PHASE1_JSON}"
  CURRENT_BEST_F1=$(extract_f1 "${CURRENT_BEST_JSON}")
  case "$(basename "${BEST_PHASE1_JSON}")" in
    arch4_interp_a03_direct_full6418.json)
      CURRENT_BEST_SNIPER="${PROJECT_ROOT}/weights/yolo_sniper_interp/interp_a03.pt"
      ;;
    a*.json)
      STEM="$(basename "${BEST_PHASE1_JSON}" .json)"
      CURRENT_BEST_SNIPER="${INTERP_DIR}/interp_${STEM}.pt"
      ;;
  esac
  python "${HELPER}" patch-config \
    --base "${ARCH4_BASE_CONFIG}" \
    --out "${CURRENT_CONFIG}" \
    --set "model.yolo.weights_hr=${CURRENT_BEST_SNIPER}" >/dev/null
fi

echo "PHASE1_BEST=${CURRENT_BEST_JSON} F1=${CURRENT_BEST_F1}"
echo "CURRENT_BEST_SNIPER=${CURRENT_BEST_SNIPER}"

# =========================================================================
# PHASE 2: Merge policy retuning
# =========================================================================
echo ""
echo "========================================"
echo "PHASE 2: Merge Policy Retuning"
echo "========================================"

PHASE2_DIR="${RUN_DIR}/phase2_merge_policy"
mkdir -p "${PHASE2_DIR}"

for BONUS in -0.05 -0.03 0.0 0.03 0.05; do
    TAG=$(echo "${BONUS}" | sed 's/-/n/; s/\./_/g; s/+//')
    CONFIG_OUT="${PHASE2_DIR}/bonus_${TAG}.yaml"
    JSON_OUT="${PHASE2_DIR}/bonus_${TAG}.json"
    echo "--- sniper_score_bonus=${BONUS} ---"
    python "${HELPER}" patch-config \
      --base "${CURRENT_CONFIG}" \
      --out "${CONFIG_OUT}" \
      --set "model.arch4.merge_policy=size_cond" \
      --set "model.arch4.sniper_score_bonus=${BONUS}" \
      --set "model.yolo.weights_hr=${CURRENT_BEST_SNIPER}" >/dev/null
    if run_probe "${CONFIG_OUT}" "${CURRENT_BEST_SNIPER}" "${JSON_OUT}"; then
      echo "  Result: $(extract_prf "${JSON_OUT}")"
    else
      log_fail "phase2 bonus ${BONUS}"
    fi
done

for POLICY in drop_true drop_false; do
    CONFIG_OUT="${PHASE2_DIR}/${POLICY}.yaml"
    JSON_OUT="${PHASE2_DIR}/${POLICY}.json"
    echo "--- ${POLICY} ---"
    if [ "${POLICY}" = "drop_true" ]; then
      python "${HELPER}" patch-config \
        --base "${CURRENT_CONFIG}" \
        --out "${CONFIG_OUT}" \
        --set "model.arch4.merge_policy=binary" \
        --set "model.arch4.drop_uncertain_if_sniper_hits=true" \
        --set "model.yolo.weights_hr=${CURRENT_BEST_SNIPER}" >/dev/null
    else
      python "${HELPER}" patch-config \
        --base "${CURRENT_CONFIG}" \
        --out "${CONFIG_OUT}" \
        --set "model.arch4.merge_policy=binary" \
        --set "model.arch4.drop_uncertain_if_sniper_hits=false" \
        --set "model.yolo.weights_hr=${CURRENT_BEST_SNIPER}" >/dev/null
    fi
    if run_probe "${CONFIG_OUT}" "${CURRENT_BEST_SNIPER}" "${JSON_OUT}"; then
      echo "  Result: $(extract_prf "${JSON_OUT}")"
    else
      log_fail "phase2 ${POLICY}"
    fi
done

BEST_PHASE2_JSON=$(python "${HELPER}" choose-best --glob "${PHASE2_DIR}/*.json" 2>/dev/null || true)
if [ -n "${BEST_PHASE2_JSON}" ] && [ -f "${BEST_PHASE2_JSON}" ]; then
  BEST_PHASE2_F1=$(extract_f1 "${BEST_PHASE2_JSON}")
  if python - <<PY
import sys
sys.exit(0 if float("${BEST_PHASE2_F1}") > float("${CURRENT_BEST_F1}") else 1)
PY
  then
    CURRENT_BEST_JSON="${BEST_PHASE2_JSON}"
    CURRENT_BEST_F1="${BEST_PHASE2_F1}"
    CURRENT_CONFIG="${PHASE2_DIR}/$(basename "${BEST_PHASE2_JSON}" .json).yaml"
  fi
fi
echo "AFTER_PHASE2_BEST=${CURRENT_BEST_JSON} F1=${CURRENT_BEST_F1}"

# =========================================================================
# PHASE 3: pass2_conf grid
# =========================================================================
echo ""
echo "========================================"
echo "PHASE 3: pass2_conf Grid"
echo "========================================"

PHASE3_DIR="${RUN_DIR}/phase3_pass2_conf"
mkdir -p "${PHASE3_DIR}"

for P2CONF in 0.40 0.45 0.50 0.55; do
    TAG="p2_$(echo "${P2CONF}" | tr -d '.')"
    CONFIG_OUT="${PHASE3_DIR}/${TAG}.yaml"
    JSON_OUT="${PHASE3_DIR}/${TAG}.json"
    echo "--- pass2_conf=${P2CONF} ---"
    python "${HELPER}" patch-config \
      --base "${CURRENT_CONFIG}" \
      --out "${CONFIG_OUT}" \
      --set "model.arch4.pass2_conf=${P2CONF}" \
      --set "model.arch4.high_conf=${P2CONF}" \
      --set "model.yolo.weights_hr=${CURRENT_BEST_SNIPER}" >/dev/null
    if run_probe "${CONFIG_OUT}" "${CURRENT_BEST_SNIPER}" "${JSON_OUT}"; then
      echo "  Result: $(extract_prf "${JSON_OUT}")"
    else
      log_fail "phase3 ${P2CONF}"
    fi
done

BEST_PHASE3_JSON=$(python "${HELPER}" choose-best --glob "${PHASE3_DIR}/*.json" 2>/dev/null || true)
if [ -n "${BEST_PHASE3_JSON}" ] && [ -f "${BEST_PHASE3_JSON}" ]; then
  BEST_PHASE3_F1=$(extract_f1 "${BEST_PHASE3_JSON}")
  if python - <<PY
import sys
sys.exit(0 if float("${BEST_PHASE3_F1}") > float("${CURRENT_BEST_F1}") else 1)
PY
  then
    CURRENT_BEST_JSON="${BEST_PHASE3_JSON}"
    CURRENT_BEST_F1="${BEST_PHASE3_F1}"
    CURRENT_CONFIG="${PHASE3_DIR}/$(basename "${BEST_PHASE3_JSON}" .json).yaml"
  fi
fi
echo "AFTER_PHASE3_BEST=${CURRENT_BEST_JSON} F1=${CURRENT_BEST_F1}"

# =========================================================================
# PHASE 4: mine hard negatives once from current best sniper
# =========================================================================
echo ""
echo "========================================"
echo "PHASE 4: Hard-Negative Mining + Manifest Build"
echo "========================================"

PHASE4_DIR="${RUN_DIR}/phase4_hardneg_datasets"
mkdir -p "${PHASE4_DIR}"

MINE_CSV="${PHASE4_DIR}/hardneg_manifest.csv"
MINE_JSON="${PHASE4_DIR}/hardneg_summary.json"

if ! python "${PROJECT_ROOT}/iac_jetson/mine_sniper_hard_negatives.py" \
  --dataset_root "${CROP_DATA_DIR}" \
  --split train \
  --weights "${CURRENT_BEST_SNIPER}" \
  --out_csv "${MINE_CSV}" \
  --out_json "${MINE_JSON}" \
  --device 0 \
  --imgsz 256 \
  --batch "${TRAIN_BATCH}" \
  --conf 0.001 \
  --iou 0.45 \
  --max_det 50 \
  --hardneg_thresh "${HARDNEG_THRESH}"; then
  log_fail "phase4 mine hard negatives"
fi

NEG50_DATA_DIR="${PHASE4_DIR}/neg50_manifest"
NEG30_DATA_DIR="${PHASE4_DIR}/neg30_manifest"

python "${PROJECT_ROOT}/iac_jetson/build_sniper_hardneg_dataset.py" \
  --base_dataset_root "${CROP_DATA_DIR}" \
  --hardneg_csv "${MINE_CSV}" \
  --out_dir "${NEG50_DATA_DIR}" \
  --hardneg_thresh "${HARDNEG_THRESH}" \
  --target_negative_ratio "${TARGET_NEG_RATIO_50}" \
  --max_extra_repeats "${MAX_REPEATS_50}" || log_fail "phase4 neg50 manifest"

python "${PROJECT_ROOT}/iac_jetson/build_sniper_hardneg_dataset.py" \
  --base_dataset_root "${CROP_DATA_DIR}" \
  --hardneg_csv "${MINE_CSV}" \
  --out_dir "${NEG30_DATA_DIR}" \
  --hardneg_thresh "${HARDNEG_THRESH}" \
  --target_negative_ratio "${TARGET_NEG_RATIO_30}" \
  --max_extra_repeats "${MAX_REPEATS_30}" || log_fail "phase4 neg30 manifest"

# =========================================================================
# PHASE 5/6: retrain
# =========================================================================
echo ""
echo "========================================"
echo "PHASE 5/6: Retraining"
echo "========================================"

NEG50_PROJECT="${PROJECT_ROOT}/weights/yolo_sniper_overnight"
NEG50_NAME="${TIMESTAMP}_neg50_ft"
NEG30_PROJECT="${PROJECT_ROOT}/weights/yolo_sniper_overnight"
NEG30_NAME="${TIMESTAMP}_neg30_ft"

if ! python "${PROJECT_ROOT}/iac_jetson/train_sniper_crop_yolo.py" \
  --data "${NEG50_DATA_DIR}/data.yaml" \
  --base_weights "${CURRENT_BEST_SNIPER}" \
  --imgsz 256 \
  --epochs "${TRAIN_EPOCHS}" \
  --batch "${TRAIN_BATCH}" \
  --patience "${TRAIN_PATIENCE}" \
  --optimizer AdamW \
  --lr0 "${TRAIN_LR0}" \
  --lrf 0.01 \
  --warmup_epochs 2 \
  --project "${NEG50_PROJECT}" \
  --name "${NEG50_NAME}" \
  --device 0 \
  --workers "${TRAIN_WORKERS}" \
  --save_period 10 \
  --amp "${TRAIN_AMP}"; then
  log_fail "phase5 neg50 train"
fi

if ! python "${PROJECT_ROOT}/iac_jetson/train_sniper_crop_yolo.py" \
  --data "${NEG30_DATA_DIR}/data.yaml" \
  --base_weights "${CURRENT_BEST_SNIPER}" \
  --imgsz 256 \
  --epochs "${TRAIN_EPOCHS}" \
  --batch "${TRAIN_BATCH}" \
  --patience "${TRAIN_PATIENCE}" \
  --optimizer AdamW \
  --lr0 "${TRAIN_LR0}" \
  --lrf 0.01 \
  --warmup_epochs 2 \
  --project "${NEG30_PROJECT}" \
  --name "${NEG30_NAME}" \
  --device 0 \
  --workers "${TRAIN_WORKERS}" \
  --save_period 10 \
  --amp "${TRAIN_AMP}"; then
  log_fail "phase6 neg30 train"
fi

NEG50_BEST="${NEG50_PROJECT}/${NEG50_NAME}/weights/best.pt"
NEG30_BEST="${NEG30_PROJECT}/${NEG30_NAME}/weights/best.pt"

# =========================================================================
# PHASE 7: retrained model direct eval
# =========================================================================
echo ""
echo "========================================"
echo "PHASE 7: Retrained Model Evaluation"
echo "========================================"

PHASE7_DIR="${RUN_DIR}/phase7_retrained_eval"
mkdir -p "${PHASE7_DIR}"

if have_file "${NEG50_BEST}"; then
  run_probe "${CURRENT_CONFIG}" "${NEG50_BEST}" "${PHASE7_DIR}/neg50_direct.json" || log_fail "phase7 neg50 direct"
fi
if have_file "${NEG30_BEST}"; then
  run_probe "${CURRENT_CONFIG}" "${NEG30_BEST}" "${PHASE7_DIR}/neg30_direct.json" || log_fail "phase7 neg30 direct"
fi

# =========================================================================
# PHASE 8: retrained + current-best interpolation
# =========================================================================
echo ""
echo "========================================"
echo "PHASE 8: Retrained + Interpolation"
echo "========================================"

PHASE8_DIR="${RUN_DIR}/phase8_retrained_interp"
mkdir -p "${PHASE8_DIR}"

interp_retrained_family() {
  local BASE_W="$1"
  local TAG_PREFIX="$2"
  if [ ! -f "${BASE_W}" ]; then
    return 0
  fi
  for ALPHA in 0.20 0.30 0.40 0.50; do
    local TAG="${TAG_PREFIX}_a${ALPHA/./}"
    local OUT_W="${PHASE8_DIR}/${TAG}.pt"
    local OUT_J="${PHASE8_DIR}/${TAG}.json"
    python "${HELPER}" interpolate \
      --ckpt-a "${CURRENT_BEST_SNIPER}" \
      --ckpt-b "${BASE_W}" \
      --alpha "${ALPHA}" \
      --out "${OUT_W}" >/dev/null || { log_fail "phase8 interpolate ${TAG}"; continue; }
    run_probe "${CURRENT_CONFIG}" "${OUT_W}" "${OUT_J}" || log_fail "phase8 probe ${TAG}"
  done
}

interp_retrained_family "${NEG50_BEST}" "neg50_interp"
interp_retrained_family "${NEG30_BEST}" "neg30_interp"

# =========================================================================
# PHASE 9: summary
# =========================================================================
echo ""
echo "========================================"
echo "PHASE 9: FINAL SUMMARY"
echo "========================================"

echo ""
echo "--- Current Reference Before Overnight ---"
echo "  CURRENT_BEST_SNIPER=${CURRENT_BEST_SNIPER}"
echo "  CURRENT_BEST_F1=${CURRENT_BEST_F1}"

for DIR in "${PHASE1_DIR}" "${PHASE2_DIR}" "${PHASE3_DIR}" "${PHASE7_DIR}" "${PHASE8_DIR}"; do
  if compgen -G "${DIR}/*.json" > /dev/null; then
    echo ""
    echo "--- $(basename "${DIR}") ---"
    for RESULT in "${DIR}"/*.json; do
      echo "  $(basename "${RESULT}" .json): $(extract_prf "${RESULT}")"
    done
  fi
done

BEST_OVERALL_JSON=$(python "${HELPER}" choose-best \
  --glob "${PHASE1_DIR}/*.json" \
  --glob "${PHASE2_DIR}/*.json" \
  --glob "${PHASE3_DIR}/*.json" \
  --glob "${PHASE7_DIR}/*.json" \
  --glob "${PHASE8_DIR}/*.json" \
  --glob "${PROJECT_ROOT}/iac_runs/20260318_arch4_interp_eval/arch4_interp_a03_direct_full6418.json" 2>/dev/null || true)

echo ""
echo "OVERALL_BEST_JSON=${BEST_OVERALL_JSON}"
if [ -n "${BEST_OVERALL_JSON}" ] && [ -f "${BEST_OVERALL_JSON}" ]; then
  echo "OVERALL_BEST=$(extract_prf "${BEST_OVERALL_JSON}")"
fi

echo ""
echo "FAILURES_LOG=${FAIL_LOG}"
echo "RUN_DIR=${RUN_DIR}"
echo "========================================"
echo "Overnight Optimization Pipeline Complete"
echo "========================================"
