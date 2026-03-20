#!/usr/bin/env python3
"""
Build a manifest-based dataset that oversamples mined hard negatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hard-negative oversampled dataset manifests.")
    parser.add_argument(
        "--base_dataset_root",
        type=str,
        default="/home/changmin/dark_vessel_sr_yolo/data/arch4_sniper_crops",
    )
    parser.add_argument("--hardneg_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--hardneg_thresh", type=float, default=0.25)
    parser.add_argument("--target_negative_ratio", type=float, default=0.30)
    parser.add_argument("--max_extra_repeats", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_root = Path(args.base_dataset_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_train_images = sorted((base_root / "images" / "train").glob("*.jpg"))
    base_val_images = sorted((base_root / "images" / "val").glob("*.jpg"))
    base_train_total = len(base_train_images)

    base_train_negative = 0
    base_train_positive = 0
    for image_path in base_train_images:
        label_path = base_root / "labels" / "train" / f"{image_path.stem}.txt"
        if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
            base_train_positive += 1
        else:
            base_train_negative += 1

    selected_hardneg: list[str] = []
    with Path(args.hardneg_csv).resolve().open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["max_conf"]) >= float(args.hardneg_thresh):
                selected_hardneg.append(row["image_path"])

    selected_count = len(selected_hardneg)
    extra_repeats = 0
    needed_repeats = 0.0
    target = float(args.target_negative_ratio)
    if selected_count > 0 and 0.0 < target < 1.0:
        needed_repeats = (
            target * base_train_total - base_train_negative
        ) / max(1e-9, selected_count * (1.0 - target))
        extra_repeats = max(0, math.ceil(needed_repeats))
        extra_repeats = min(extra_repeats, int(args.max_extra_repeats))

    train_paths = [str(p) for p in base_train_images]
    for _ in range(extra_repeats):
        train_paths.extend(selected_hardneg)

    val_paths = [str(p) for p in base_val_images]

    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    data_yaml = out_dir / "data.yaml"
    manifest_json = out_dir / "build_summary.json"

    train_txt.write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(val_paths) + "\n", encoding="utf-8")
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_dir}",
                f"train: {train_txt}",
                f"val: {val_txt}",
                "nc: 1",
                "names:",
                "  0: ship",
                "",
            ]
        ),
        encoding="utf-8",
    )

    final_total = base_train_total + selected_count * extra_repeats
    final_negative = base_train_negative + selected_count * extra_repeats
    summary = {
        "base_dataset_root": str(base_root),
        "hardneg_csv": str(Path(args.hardneg_csv).resolve()),
        "hardneg_thresh": float(args.hardneg_thresh),
        "target_negative_ratio": float(args.target_negative_ratio),
        "max_extra_repeats": int(args.max_extra_repeats),
        "base_train_total": base_train_total,
        "base_train_positive": base_train_positive,
        "base_train_negative": base_train_negative,
        "selected_hardneg": selected_count,
        "needed_repeats_unclamped": needed_repeats,
        "extra_repeats": extra_repeats,
        "final_train_total": final_total,
        "final_train_negative": final_negative,
        "final_negative_ratio": (final_negative / final_total if final_total else 0.0),
        "target_ratio_reached": (final_negative / final_total if final_total else 0.0) >= target,
        "train_txt": str(train_txt),
        "val_txt": str(val_txt),
        "data_yaml": str(data_yaml),
    }
    manifest_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
