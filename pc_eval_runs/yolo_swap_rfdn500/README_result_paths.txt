[Experiment]
EXP_NAME=yolo_swap_rfdn500
RUN_ROOT=pc_eval_runs/yolo_swap_rfdn500

[Main result JSON]
- pc_eval_runs/yolo_swap_rfdn500/results/baseline_hr.json
- pc_eval_runs/yolo_swap_rfdn500/results/baseline_lr.json
- pc_eval_runs/yolo_swap_rfdn500/results/arch0_eval.json
- pc_eval_runs/yolo_swap_rfdn500/results/arch2_eval.json
- pc_eval_runs/yolo_swap_rfdn500/results/arch4_balanced_full.json
- pc_eval_runs/yolo_swap_rfdn500/results/arch4_recall_full.json
- pc_eval_runs/yolo_swap_rfdn500/results/arch4_balanced_deploy_200.json
- pc_eval_runs/yolo_swap_rfdn500/results/arch4_recall_deploy_200.json

[Patched configs]
- pc_eval_runs/yolo_swap_rfdn500/configs/arch0.yaml
- pc_eval_runs/yolo_swap_rfdn500/configs/arch2.yaml
- pc_eval_runs/yolo_swap_rfdn500/configs/arch4_balanced.yaml
- pc_eval_runs/yolo_swap_rfdn500/configs/arch4_recall.yaml
- pc_eval_runs/yolo_swap_rfdn500/configs/arch4_balanced_deploy.yaml
- pc_eval_runs/yolo_swap_rfdn500/configs/arch4_recall_deploy.yaml

[Notes]
- Arch0/Arch2 내부 비교는 YOLO_SR_WEIGHTS 기준으로 계산됨
- pure baseline 비교는 baseline_hr.json / baseline_lr.json 을 우선 참고
