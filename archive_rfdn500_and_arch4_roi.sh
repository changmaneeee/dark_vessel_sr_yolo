#!/usr/bin/env bash
set -euo pipefail

# 사용 예시:
# bash archive_rfdn500_and_arch4_roi.sh 2026-03-03_rfdn500_and_arch4_roi

TAG="${1:-2026-03-03_rfdn500_and_arch4_roi}"
ROOT="archive/${TAG}"

mkdir -p "$ROOT"/{code,configs,results,logs,notes}

copy_if_exists() {
  src="$1"
  dst="$2"
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp -r "$src" "$dst"
    echo "[OK] $src -> $dst"
  else
    echo "[MISS] $src"
  fi
}

# -----------------------
# code
# -----------------------
copy_if_exists src/models/pipelines/arch4_roi_awareNMS.py "$ROOT/code/arch4_roi_awareNMS.py"
copy_if_exists iac_scripts/arch4_eval_ultralytics.py "$ROOT/code/arch4_eval_ultralytics.py"
copy_if_exists iac_scripts/arch4_one_debug.py "$ROOT/code/arch4_one_debug.py"
copy_if_exists run_arch024_pc_eval_iac.sh "$ROOT/code/run_arch024_pc_eval_iac.sh"

# -----------------------
# configs
# -----------------------
copy_if_exists configs/experiment/arch4_roi_awareNMS_eval.yaml "$ROOT/configs/arch4_roi_awareNMS_eval.yaml"
copy_if_exists configs/experiment/arch4_roi_awareNMS_deploy.yaml "$ROOT/configs/arch4_roi_awareNMS_deploy.yaml"
copy_if_exists pc_eval_runs/yolo_swap_rfdn500/configs "$ROOT/configs/pc_eval_configs"

# -----------------------
# results
# -----------------------
copy_if_exists pc_eval_runs/yolo_swap_rfdn500/results/arch0_eval.json "$ROOT/results/arch0_eval.json"
copy_if_exists pc_eval_runs/yolo_swap_rfdn500/results/arch2_eval.json "$ROOT/results/arch2_eval.json"
copy_if_exists pc_eval_runs/yolo_swap_rfdn500/results/baseline_hr.json "$ROOT/results/baseline_hr.json"
copy_if_exists pc_eval_runs/yolo_swap_rfdn500/results/baseline_lr.json "$ROOT/results/baseline_lr.json"
copy_if_exists pc_eval_runs/yolo_swap_rfdn500/results/arch4_balanced_full.json "$ROOT/results/arch4_balanced_full.json"
copy_if_exists pc_eval_runs/yolo_swap_rfdn500/results/arch4_recall_full.json "$ROOT/results/arch4_recall_full.json"
copy_if_exists pc_eval_runs/yolo_swap_rfdn500/results/arch4_balanced_deploy_200.json "$ROOT/results/arch4_balanced_deploy_200.json"
copy_if_exists pc_eval_runs/yolo_swap_rfdn500/results/arch4_recall_deploy_200.json "$ROOT/results/arch4_recall_deploy_200.json"
copy_if_exists iac_runs/arch4_roi_eval/arch4_roi_eval_200.json "$ROOT/results/arch4_roi_eval_200.json"
copy_if_exists iac_runs/arch4_roi_eval/arch4_roi_deploy_200.json "$ROOT/results/arch4_roi_deploy_200.json"
copy_if_exists iac_runs/arch4_roi_debug_one "$ROOT/results/arch4_roi_debug_one"

# -----------------------
# logs
# -----------------------
copy_if_exists pc_eval_runs/yolo_swap_rfdn500/logs "$ROOT/logs/pc_eval_logs"
copy_if_exists iac_runs/arch4_roi_eval "$ROOT/logs/arch4_roi_eval_folder"

# -----------------------
# notes
# -----------------------
copy_if_exists /mnt/data/arch4_nms_progress_summary.md "$ROOT/notes/arch4_nms_progress_summary.md"

# archive compress

tar -czf "${ROOT}.tar.gz" -C archive "$TAG"
echo "\n[DONE] archive created: ${ROOT}.tar.gz"
