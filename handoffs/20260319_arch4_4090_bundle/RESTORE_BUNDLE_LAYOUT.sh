#!/bin/bash
set -euo pipefail

# Restore transfer bundle into a repo-like layout on a new machine.
# Usage:
#   bash RESTORE_BUNDLE_LAYOUT.sh /path/to/20260319_arch4_4090_bundle /target/project/root
#
# Example:
#   bash RESTORE_BUNDLE_LAYOUT.sh \
#     /media/$USER/EXT/20260319_arch4_4090_bundle \
#     /home/$USER/dark_vessel_sr_yolo

if [ "$#" -ne 2 ]; then
  echo "Usage: bash RESTORE_BUNDLE_LAYOUT.sh <bundle_dir> <project_root>"
  exit 1
fi

BUNDLE_DIR="$(cd "$1" && pwd)"
PROJECT_ROOT="$2"

mkdir -p "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/src/models/pipelines"
mkdir -p "$PROJECT_ROOT/iac_jetson"
mkdir -p "$PROJECT_ROOT/iac_runs"
mkdir -p "$PROJECT_ROOT/configs/experiment"
mkdir -p "$PROJECT_ROOT/weights"
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/handoffs"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -f "$src" ]; then
    cp -f "$src" "$dst"
    echo "copied: $src -> $dst"
  else
    echo "skip (missing): $src"
  fi
}

# Pipelines / core code
copy_if_exists "$BUNDLE_DIR/code/arch4_roi_awareNMS_ablation.py" "$PROJECT_ROOT/src/models/pipelines/arch4_roi_awareNMS_ablation.py"
copy_if_exists "$BUNDLE_DIR/code/arch4_overnight_helper.py" "$PROJECT_ROOT/iac_jetson/arch4_overnight_helper.py"

# Jetson / evaluation / training helpers
copy_if_exists "$BUNDLE_DIR/code/arch4_wiring_check.py" "$PROJECT_ROOT/iac_jetson/arch4_wiring_check.py"
copy_if_exists "$BUNDLE_DIR/code/arch4_dump_sniper_crops.py" "$PROJECT_ROOT/iac_jetson/arch4_dump_sniper_crops.py"
copy_if_exists "$BUNDLE_DIR/code/scout_recall_diagnostic.py" "$PROJECT_ROOT/iac_jetson/scout_recall_diagnostic.py"
copy_if_exists "$BUNDLE_DIR/code/train_scout_yolo.py" "$PROJECT_ROOT/iac_jetson/train_scout_yolo.py"
copy_if_exists "$BUNDLE_DIR/code/train_sniper_crop_yolo.py" "$PROJECT_ROOT/iac_jetson/train_sniper_crop_yolo.py"
copy_if_exists "$BUNDLE_DIR/code/mine_sniper_hard_negatives.py" "$PROJECT_ROOT/iac_jetson/mine_sniper_hard_negatives.py"
copy_if_exists "$BUNDLE_DIR/code/build_sniper_hardneg_dataset.py" "$PROJECT_ROOT/iac_jetson/build_sniper_hardneg_dataset.py"
copy_if_exists "$BUNDLE_DIR/code/interpolate_sniper_checkpoints.py" "$PROJECT_ROOT/iac_jetson/interpolate_sniper_checkpoints.py"
copy_if_exists "$BUNDLE_DIR/code/validate_paired_dataset.py" "$PROJECT_ROOT/iac_jetson/validate_paired_dataset.py"

# Shell scripts
copy_if_exists "$BUNDLE_DIR/code/run_overnight_optimization.sh" "$PROJECT_ROOT/iac_runs/run_overnight_optimization.sh"
copy_if_exists "$BUNDLE_DIR/code/run_scout_diagnostic_day1.sh" "$PROJECT_ROOT/iac_runs/run_scout_diagnostic_day1.sh"
copy_if_exists "$BUNDLE_DIR/code/run_scout_retrain_v2.sh" "$PROJECT_ROOT/iac_runs/run_scout_retrain_v2.sh"

# Configs
copy_if_exists "$BUNDLE_DIR/configs/arch4_roi_awareNMS_deploy.yaml" "$PROJECT_ROOT/configs/experiment/arch4_roi_awareNMS_deploy.yaml"
copy_if_exists "$BUNDLE_DIR/configs/arch4_sizecond_hardneg.yaml" "$PROJECT_ROOT/configs/experiment/arch4_sizecond_hardneg.yaml"
copy_if_exists "$BUNDLE_DIR/configs/arch4_sizecond_interp_a03.yaml" "$PROJECT_ROOT/configs/experiment/arch4_sizecond_interp_a03.yaml"

# Weights
copy_if_exists "$BUNDLE_DIR/weights/interp_a03.pt" "$PROJECT_ROOT/weights/interp_a03.pt"
copy_if_exists "$BUNDLE_DIR/weights/rfdn_arch4_model_best.pt" "$PROJECT_ROOT/weights/rfdn_arch4_model_best.pt"
copy_if_exists "$BUNDLE_DIR/weights/scout_yolo_lr_best.pt" "$PROJECT_ROOT/weights/scout_yolo_lr_best.pt"
copy_if_exists "$BUNDLE_DIR/weights/sniper_cropft_best.pt" "$PROJECT_ROOT/weights/sniper_cropft_best.pt"
copy_if_exists "$BUNDLE_DIR/weights/sniper_hardneg_best.pt" "$PROJECT_ROOT/weights/sniper_hardneg_best.pt"

# Docs / env / results
mkdir -p "$PROJECT_ROOT/handoffs/20260319_arch4_4090_bundle"
cp -f "$BUNDLE_DIR"/*.md "$PROJECT_ROOT/handoffs/20260319_arch4_4090_bundle/" 2>/dev/null || true
cp -f "$BUNDLE_DIR/MANIFEST.txt" "$PROJECT_ROOT/handoffs/20260319_arch4_4090_bundle/" 2>/dev/null || true
cp -f "$BUNDLE_DIR/RESTORE_BUNDLE_LAYOUT.sh" "$PROJECT_ROOT/handoffs/20260319_arch4_4090_bundle/" 2>/dev/null || true
cp -f "$BUNDLE_DIR/REBUILD_SNIPER_CROPS_ON_4090.sh" "$PROJECT_ROOT/handoffs/20260319_arch4_4090_bundle/" 2>/dev/null || true
mkdir -p "$PROJECT_ROOT/handoffs/20260319_arch4_4090_bundle/docs"
cp -f "$BUNDLE_DIR/docs/"*.md "$PROJECT_ROOT/handoffs/20260319_arch4_4090_bundle/docs/" 2>/dev/null || true
mkdir -p "$PROJECT_ROOT/handoffs/20260319_arch4_4090_bundle/env"
cp -f "$BUNDLE_DIR/env/"* "$PROJECT_ROOT/handoffs/20260319_arch4_4090_bundle/env/" 2>/dev/null || true
mkdir -p "$PROJECT_ROOT/handoffs/20260319_arch4_4090_bundle/results"
cp -f "$BUNDLE_DIR/results/"* "$PROJECT_ROOT/handoffs/20260319_arch4_4090_bundle/results/" 2>/dev/null || true

echo
echo "Restore complete."
echo "Bundle dir   : $BUNDLE_DIR"
echo "Project root : $PROJECT_ROOT"
echo
echo "Next suggested checks:"
echo "  ls $PROJECT_ROOT/src/models/pipelines"
echo "  ls $PROJECT_ROOT/iac_jetson"
echo "  ls $PROJECT_ROOT/configs/experiment"
echo "  ls $PROJECT_ROOT/weights"
