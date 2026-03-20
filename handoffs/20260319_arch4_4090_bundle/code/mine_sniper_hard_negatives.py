#!/usr/bin/env python3
"""
Mine hard negatives from existing Arch4 Sniper crop dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard negatives from ROI crop dataset.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/changmin/dark_vessel_sr_yolo/data/arch4_sniper_crops",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max_det", type=int, default=50)
    parser.add_argument("--hardneg_thresh", type=float, default=0.25)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=500)
    return parser.parse_args()


def batched(seq: list[Path], size: int) -> Iterable[list[Path]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def is_empty_label(path: Path) -> bool:
    if not path.exists():
        return True
    return len(path.read_text(encoding="utf-8").strip()) == 0


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    img_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split

    negative_images: list[Path] = []
    for image_path in sorted(img_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if is_empty_label(label_path):
            negative_images.append(image_path)

    if args.max_images > 0:
        negative_images = negative_images[: args.max_images]

    model = YOLO(args.weights)

    out_csv = Path(args.out_csv).resolve()
    out_json = Path(args.out_json).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split",
        "image_path",
        "label_path",
        "image_stem",
        "num_preds",
        "max_conf",
        "sum_conf",
        "hard_negative",
    ]

    total = len(negative_images)
    processed = 0
    selected = 0
    max_conf_values: list[float] = []

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for batch_paths in batched(negative_images, max(1, args.batch)):
            results = model.predict(
                source=[str(p) for p in batch_paths],
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                max_det=args.max_det,
                verbose=False,
                stream=False,
            )

            for image_path, result in zip(batch_paths, results):
                label_path = label_dir / f"{image_path.stem}.txt"
                confs = result.boxes.conf.detach().cpu().tolist() if result.boxes is not None else []
                num_preds = len(confs)
                max_conf = float(max(confs)) if confs else 0.0
                sum_conf = float(sum(confs)) if confs else 0.0
                hard_negative = max_conf >= float(args.hardneg_thresh)
                selected += int(hard_negative)
                max_conf_values.append(max_conf)

                writer.writerow(
                    {
                        "split": args.split,
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "image_stem": image_path.stem,
                        "num_preds": int(num_preds),
                        "max_conf": max_conf,
                        "sum_conf": sum_conf,
                        "hard_negative": int(hard_negative),
                    }
                )
                processed += 1
                if processed % int(args.print_every) == 0 or processed == total:
                    print(
                        f"[mine_hardneg] {processed}/{total} "
                        f"selected={selected} "
                        f"ratio={(selected / processed if processed else 0.0):.4f}"
                    )

    sorted_conf = sorted(max_conf_values)
    def pct(p: float) -> float:
        if not sorted_conf:
            return 0.0
        idx = min(len(sorted_conf) - 1, max(0, math.ceil((p / 100.0) * len(sorted_conf)) - 1))
        return float(sorted_conf[idx])

    summary = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "weights": args.weights,
        "hardneg_thresh": float(args.hardneg_thresh),
        "num_negative_candidates": total,
        "num_selected_hardneg": selected,
        "selected_ratio": (selected / total if total else 0.0),
        "max_conf_stats": {
            "mean": (sum(max_conf_values) / len(max_conf_values) if max_conf_values else 0.0),
            "p50": pct(50),
            "p90": pct(90),
            "p95": pct(95),
            "p99": pct(99),
            "max": (max(sorted_conf) if sorted_conf else 0.0),
        },
        "out_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
