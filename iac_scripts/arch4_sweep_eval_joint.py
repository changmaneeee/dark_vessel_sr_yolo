#!/usr/bin/env python3
"""
Arch4 joint sweep (recall-first)

목표:
- 상호작용이 큰 파라미터(pass1_conf, high_conf, roi_expansion, crop_size_lr)를
  한 번에 coarse sweep
- arch4_eval_ultralytics.py를 반복 실행
- direct/recall50 우선으로 정렬
- 결과를 CSV/JSON으로 저장

전제:
- arch4_eval_ultralytics.py가 아래 direct metric을 JSON에 저장해야 함
  * direct/tp50
  * direct/fp50
  * direct/fn50
  * direct/precision50
  * direct/recall50
"""

import sys
import json
import csv
import time
import copy
import itertools
import subprocess
from pathlib import Path

import yaml


def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data, path: Path):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def set_nested(d, keys, value):
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def read_result_json(path: Path):
    with open(path, "r") as f:
        data = json.load(f)

    rd = data["runs"][0]["results_dict"]
    meta = data.get("meta", {})

    return {
        "precision": rd.get("metrics/precision(B)", None),
        "recall": rd.get("metrics/recall(B)", None),
        "map50": rd.get("metrics/mAP50(B)", None),
        "map5095": rd.get("metrics/mAP50-95(B)", None),

        "tp50": rd.get("direct/tp50", None),
        "fp50": rd.get("direct/fp50", None),
        "fn50": rd.get("direct/fn50", None),
        "precision50_direct": rd.get("direct/precision50", None),
        "recall50_direct": rd.get("direct/recall50", None),

        "avg_ms_per_image": meta.get("avg_ms_per_image", None),
    }


def sort_key(row):
    """
    recall-first 정렬:
    1) direct recall50 높을수록 좋음
    2) fn50 낮을수록 좋음
    3) mAP50-95 높을수록 좋음
    4) fp50 낮을수록 좋음
    5) avg_ms 낮을수록 좋음
    """
    recall50 = row["recall50_direct"] if row["recall50_direct"] is not None else -1.0
    fn50 = row["fn50"] if row["fn50"] is not None else 10**9
    map5095 = row["map5095"] if row["map5095"] is not None else -1.0
    fp50 = row["fp50"] if row["fp50"] is not None else 10**9
    avg_ms = row["avg_ms_per_image"] if row["avg_ms_per_image"] is not None else 10**9

    # sorted(..., key=sort_key) 에서 오름차순 정렬이므로
    # 높은 게 좋은 것은 음수로 바꿈
    return (-recall50, fn50, -map5095, fp50, avg_ms)


def main():
    import argparse

    p = argparse.ArgumentParser()

    p.add_argument("--base_config", required=True,
                   help="예: configs/experiment/arch4_adaptive_eval.yaml")
    p.add_argument("--hr_data_yaml", required=True)
    p.add_argument("--lr_data_yaml", required=True)

    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--max_images", type=int, default=200)

    p.add_argument("--out_dir", default="iac_runs/arch4_sweep_joint")

    # joint sweep lists
    p.add_argument("--pass1_list", nargs="+", type=float,
                   default=[0.005, 0.01, 0.02, 0.05])
    p.add_argument("--high_conf_list", nargs="+", type=float,
                   default=[0.35, 0.45])
    p.add_argument("--roi_list", nargs="+", type=float,
                   default=[1.5, 2.0, 2.5])
    p.add_argument("--crop_list", nargs="+", type=int,
                   default=[64, 96])

    # fixed eval params
    p.add_argument("--sniper_conf", type=float, default=0.001)
    p.add_argument("--final_conf", type=float, default=0.001)
    p.add_argument("--merge_iou", type=float, default=0.5)

    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    configs_dir = out_dir / "configs"
    results_dir = out_dir / "results"
    logs_dir = out_dir / "logs"

    configs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(Path(args.base_config))

    combos = list(itertools.product(
        args.pass1_list,
        args.high_conf_list,
        args.roi_list,
        args.crop_list
    ))

    print(f"[SWEEP] total combinations: {len(combos)}")

    rows = []
    sweep_t0 = time.time()

    for idx, (pass1_conf, high_conf, roi_expansion, crop_size_lr) in enumerate(combos, 1):
        cfg = copy.deepcopy(base_cfg)

        # ---------------------------
        # Arch4 eval params set
        # ---------------------------
        set_nested(cfg, ["model", "arch4", "pass1_conf"], pass1_conf)

        # alias 혼동 방지 위해 둘 다 세팅
        set_nested(cfg, ["model", "arch4", "high_conf"], high_conf)
        set_nested(cfg, ["model", "arch4", "pass2_conf"], high_conf)

        set_nested(cfg, ["model", "arch4", "sniper_conf"], args.sniper_conf)
        set_nested(cfg, ["model", "arch4", "final_conf"], args.final_conf)
        set_nested(cfg, ["model", "arch4", "merge_iou"], args.merge_iou)

        set_nested(cfg, ["model", "arch4", "roi_expansion"], roi_expansion)
        set_nested(cfg, ["model", "arch4", "crop_size_lr"], crop_size_lr)

        run_name = (
            f"run_{idx:03d}"
            f"_p1_{pass1_conf}"
            f"_hc_{high_conf}"
            f"_roi_{roi_expansion}"
            f"_crop_{crop_size_lr}"
        )

        cfg_path = configs_dir / f"{run_name}.yaml"
        out_json = results_dir / f"{run_name}.json"
        log_path = logs_dir / f"{run_name}.log"

        save_yaml(cfg, cfg_path)

        cmd = [
            sys.executable,
            "iac_scripts/arch4_eval_ultralytics.py",
            "--arch4_config", str(cfg_path),
            "--hr_data_yaml", args.hr_data_yaml,
            "--lr_data_yaml", args.lr_data_yaml,
            "--eval_space", "hr",
            "--batch", str(args.batch),
            "--max_images", str(args.max_images),
            "--device", args.device,
            "--out_json", str(out_json),
        ]

        print("\n" + "=" * 90)
        print(f"[{idx}/{len(combos)}] {run_name}")
        print("=" * 90)

        run_t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        run_t1 = time.time()

        with open(log_path, "w") as f:
            f.write("CMD:\n")
            f.write(" ".join(cmd) + "\n\n")
            f.write("STDOUT:\n")
            f.write(proc.stdout or "")
            f.write("\n\nSTDERR:\n")
            f.write(proc.stderr or "")

        if proc.returncode != 0:
            print("[FAIL]")
            print((proc.stderr or "")[-1500:])

            rows.append({
                "run_name": run_name,
                "pass1_conf": pass1_conf,
                "high_conf": high_conf,
                "roi_expansion": roi_expansion,
                "crop_size_lr": crop_size_lr,
                "precision": None,
                "recall": None,
                "map50": None,
                "map5095": None,
                "tp50": None,
                "fp50": None,
                "fn50": None,
                "precision50_direct": None,
                "recall50_direct": None,
                "avg_ms_per_image": None,
                "wall_time_sec": run_t1 - run_t0,
                "status": "fail",
                "log_path": str(log_path),
                "json_path": str(out_json),
                "config_path": str(cfg_path),
            })
            continue

        metrics = read_result_json(out_json)

        row = {
            "run_name": run_name,
            "pass1_conf": pass1_conf,
            "high_conf": high_conf,
            "roi_expansion": roi_expansion,
            "crop_size_lr": crop_size_lr,

            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "map50": metrics["map50"],
            "map5095": metrics["map5095"],

            "tp50": metrics["tp50"],
            "fp50": metrics["fp50"],
            "fn50": metrics["fn50"],
            "precision50_direct": metrics["precision50_direct"],
            "recall50_direct": metrics["recall50_direct"],

            "avg_ms_per_image": metrics["avg_ms_per_image"],
            "wall_time_sec": run_t1 - run_t0,
            "status": "ok",
            "log_path": str(log_path),
            "json_path": str(out_json),
            "config_path": str(cfg_path),
        }
        rows.append(row)

        print(
            "[OK] "
            f"R50={row['recall50_direct']:.4f}, "
            f"FN50={row['fn50']}, "
            f"FP50={row['fp50']}, "
            f"mAP50-95={row['map5095']:.4f}, "
            f"avg_ms={row['avg_ms_per_image']:.2f}"
        )

    ok_rows = [r for r in rows if r["status"] == "ok"]
    ok_rows_sorted = sorted(ok_rows, key=sort_key)

    # CSV 저장
    csv_path = out_dir / "arch4_sweep_joint_results.csv"
    fieldnames = [
        "run_name", "pass1_conf", "high_conf", "roi_expansion", "crop_size_lr",
        "precision", "recall", "map50", "map5095",
        "tp50", "fp50", "fn50", "precision50_direct", "recall50_direct",
        "avg_ms_per_image", "wall_time_sec", "status",
        "config_path", "json_path", "log_path"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    summary = {
        "meta": {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_config": args.base_config,
            "hr_data_yaml": args.hr_data_yaml,
            "lr_data_yaml": args.lr_data_yaml,
            "device": args.device,
            "batch": args.batch,
            "max_images": args.max_images,
            "num_runs": len(combos),
            "elapsed_sec": time.time() - sweep_t0,
        },
        "grid": {
            "pass1_list": args.pass1_list,
            "high_conf_list": args.high_conf_list,
            "roi_list": args.roi_list,
            "crop_list": args.crop_list,
            "sniper_conf": args.sniper_conf,
            "final_conf": args.final_conf,
            "merge_iou": args.merge_iou,
        },
        "ranking_rule": [
            "1) higher direct/recall50",
            "2) lower direct/fn50",
            "3) higher metrics/mAP50-95(B)",
            "4) lower direct/fp50",
            "5) lower avg_ms_per_image",
        ],
        "best_runs": ok_rows_sorted[:10],
        "all_runs": rows,
    }

    summary_path = out_dir / "arch4_sweep_joint_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 90)
    print("SWEEP DONE")
    print("=" * 90)
    print(f"CSV     : {csv_path}")
    print(f"SUMMARY : {summary_path}")

    print("\nTop-10 runs (recall-first ranking):")
    for i, r in enumerate(ok_rows_sorted[:10], 1):
        print(
            f"{i:02d}. {r['run_name']} | "
            f"R50={r['recall50_direct']:.4f}, "
            f"FN50={r['fn50']}, "
            f"FP50={r['fp50']}, "
            f"mAP50-95={r['map5095']:.4f}, "
            f"avg_ms={r['avg_ms_per_image']:.2f}"
        )


if __name__ == "__main__":
    main()