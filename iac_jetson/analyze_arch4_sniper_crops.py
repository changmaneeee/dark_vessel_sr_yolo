#!/usr/bin/env python3
"""
Analyze ROI-crop Sniper dataset statistics for Arch4.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Arch4 Sniper crop dataset.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/changmin/dark_vessel_sr_yolo/data/arch4_sniper_crops",
    )
    parser.add_argument("--out_json", type=str, required=True)
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((p / 100.0) * len(ordered)) - 1))
    return float(ordered[idx])


def analyze_split(dataset_root: Path, split: str, processed_images: int) -> dict[str, Any]:
    img_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    image_files = sorted(img_dir.glob("*.jpg"))
    label_files = sorted(label_dir.glob("*.txt"))

    per_source_image: dict[str, int] = {}
    boxes_per_crop: list[int] = []
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    aspects: list[float] = []
    centers_x: list[float] = []
    centers_y: list[float] = []

    positive = 0
    negative = 0
    multi_gt = 0
    boundary_close = 0

    for label_file in label_files:
        stem = label_file.stem
        source_key = stem.rsplit("_roi", 1)[0]
        per_source_image[source_key] = per_source_image.get(source_key, 0) + 1

        rows = [line for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        box_count = len(rows)
        boxes_per_crop.append(box_count)

        if box_count == 0:
            negative += 1
        else:
            positive += 1
        if box_count > 1:
            multi_gt += 1

        for row in rows:
            _, cx, cy, w, h = row.split()
            cx_f = float(cx)
            cy_f = float(cy)
            w_f = float(w)
            h_f = float(h)
            widths.append(w_f)
            heights.append(h_f)
            areas.append(w_f * h_f)
            aspects.append(w_f / max(h_f, 1e-9))
            centers_x.append(cx_f)
            centers_y.append(cy_f)
            if cx_f < 0.05 or cx_f > 0.95 or cy_f < 0.05 or cy_f > 0.95:
                boundary_close += 1

    crop_counts = list(per_source_image.values())
    image_sizes_kb = [path.stat().st_size / 1024.0 for path in image_files]

    return {
        "image_files": len(image_files),
        "label_files": len(label_files),
        "processed_images": processed_images,
        "source_images_with_crops": len(per_source_image),
        "source_coverage_ratio": len(per_source_image) / processed_images if processed_images else 0.0,
        "source_images_without_crops": processed_images - len(per_source_image),
        "positive_crops": positive,
        "negative_crops": negative,
        "positive_ratio": positive / len(label_files) if label_files else 0.0,
        "negative_ratio": negative / len(label_files) if label_files else 0.0,
        "multi_gt_crops": multi_gt,
        "multi_gt_ratio": multi_gt / len(label_files) if label_files else 0.0,
        "crop_count_per_source_image": {
            "mean": statistics.mean(crop_counts) if crop_counts else 0.0,
            "median": statistics.median(crop_counts) if crop_counts else 0.0,
            "p90": percentile(crop_counts, 90),
            "p95": percentile(crop_counts, 95),
            "max": max(crop_counts) if crop_counts else 0.0,
        },
        "boxes_per_crop": {
            "mean": statistics.mean(boxes_per_crop) if boxes_per_crop else 0.0,
            "median": statistics.median(boxes_per_crop) if boxes_per_crop else 0.0,
            "p90": percentile(boxes_per_crop, 90),
            "p95": percentile(boxes_per_crop, 95),
            "max": max(boxes_per_crop) if boxes_per_crop else 0.0,
        },
        "bbox_norm": {
            "count": len(widths),
            "width_mean": statistics.mean(widths) if widths else 0.0,
            "width_median": statistics.median(widths) if widths else 0.0,
            "width_p90": percentile(widths, 90),
            "height_mean": statistics.mean(heights) if heights else 0.0,
            "height_median": statistics.median(heights) if heights else 0.0,
            "height_p90": percentile(heights, 90),
            "area_mean": statistics.mean(areas) if areas else 0.0,
            "area_median": statistics.median(areas) if areas else 0.0,
            "area_p90": percentile(areas, 90),
            "aspect_mean": statistics.mean(aspects) if aspects else 0.0,
            "aspect_median": statistics.median(aspects) if aspects else 0.0,
            "center_x_mean": statistics.mean(centers_x) if centers_x else 0.0,
            "center_y_mean": statistics.mean(centers_y) if centers_y else 0.0,
            "boundary_close_ratio": boundary_close / len(widths) if widths else 0.0,
        },
        "jpeg_size_kb": {
            "total_gb": sum(path.stat().st_size for path in image_files) / 1024.0 / 1024.0 / 1024.0 if image_files else 0.0,
            "mean": statistics.mean(image_sizes_kb) if image_sizes_kb else 0.0,
            "median": statistics.median(image_sizes_kb) if image_sizes_kb else 0.0,
            "p90": percentile(image_sizes_kb, 90),
            "p95": percentile(image_sizes_kb, 95),
            "max": max(image_sizes_kb) if image_sizes_kb else 0.0,
        },
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    stats_path = dataset_root / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    result = {
        "dataset_root": str(dataset_root),
        "source_stats": stats,
        "analysis": {
            "train": analyze_split(dataset_root, "train", int(stats["train"]["processed_images"])),
            "val": analyze_split(dataset_root, "val", int(stats["val"]["processed_images"])),
        },
    }

    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
