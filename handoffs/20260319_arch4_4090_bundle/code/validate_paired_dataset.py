#!/usr/bin/env python3
"""
Validate image/label pairs and emit only valid pairs.

Use this before long-running eval/training on another machine to reduce the
chance of crashes caused by corrupt images, malformed labels, or missing pairs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paired image/label dataset and write valid pair manifests.")
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--allow_empty_labels", action="store_true", help="Treat empty label files as valid.")
    parser.add_argument("--max_items", type=int, default=0, help="0 means all.")
    return parser.parse_args()


def build_stem_map(root: Path, exts: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    allowed = {e.lower() for e in exts}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        key = str(path.relative_to(root).with_suffix(""))
        out[key] = path
    return out


def validate_image(path: Path) -> Tuple[bool, str]:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def validate_label(path: Path, allow_empty: bool) -> Tuple[bool, str]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if not raw:
        return (True, "") if allow_empty else (False, "empty label file")

    for lineno, line in enumerate(raw.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) < 5:
            return False, f"line {lineno}: expected >=5 columns, got {len(parts)}"
        try:
            nums = [float(x) for x in parts[:5]]
        except ValueError as exc:
            return False, f"line {lineno}: non-numeric token ({exc})"
        if not all(math.isfinite(x) for x in nums):
            return False, f"line {lineno}: non-finite value"
        cls, cx, cy, w, h = nums
        if cls < 0:
            return False, f"line {lineno}: negative class id"
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            return False, f"line {lineno}: center out of [0,1]"
        if not (0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            return False, f"line {lineno}: width/height out of [0,1]"
    return True, ""


def write_lines(path: Path, rows: List[str]) -> None:
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir).resolve()
    labels_dir = Path(args.labels_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_map = build_stem_map(images_dir, IMAGE_EXTS)
    label_map = build_stem_map(labels_dir, {".txt"})

    image_keys = set(image_map.keys())
    label_keys = set(label_map.keys())
    both_keys = sorted(image_keys & label_keys)
    only_images = sorted(image_keys - label_keys)
    only_labels = sorted(label_keys - image_keys)

    if args.max_items > 0:
        both_keys = both_keys[: args.max_items]

    valid_pairs: List[dict] = []
    invalid_images: List[str] = []
    invalid_labels: List[str] = []
    invalid_pair_reasons: List[str] = []

    for idx, key in enumerate(both_keys, start=1):
        img_path = image_map[key]
        lbl_path = label_map[key]

        ok_img, reason_img = validate_image(img_path)
        ok_lbl, reason_lbl = validate_label(lbl_path, allow_empty=args.allow_empty_labels)

        if ok_img and ok_lbl:
            valid_pairs.append(
                {
                    "stem": key,
                    "image": str(img_path),
                    "label": str(lbl_path),
                }
            )
        else:
            if not ok_img:
                invalid_images.append(f"{key}\t{img_path}\t{reason_img}")
            if not ok_lbl:
                invalid_labels.append(f"{key}\t{lbl_path}\t{reason_lbl}")
            invalid_pair_reasons.append(
                f"{key}\timage_ok={ok_img}\tlabel_ok={ok_lbl}\timg_reason={reason_img or '-'}\tlbl_reason={reason_lbl or '-'}"
            )

        if idx % 1000 == 0:
            print(f"[validate_paired_dataset] checked {idx}/{len(both_keys)} pairs")

    valid_pairs_path = out_dir / "valid_pairs.json"
    valid_images_txt = out_dir / "valid_images.txt"
    valid_labels_txt = out_dir / "valid_labels.txt"
    invalid_images_txt = out_dir / "invalid_images.txt"
    invalid_labels_txt = out_dir / "invalid_labels.txt"
    invalid_pairs_txt = out_dir / "invalid_pairs.txt"
    missing_images_txt = out_dir / "missing_image_for_label.txt"
    missing_labels_txt = out_dir / "missing_label_for_image.txt"

    valid_pairs_path.write_text(json.dumps(valid_pairs, indent=2), encoding="utf-8")
    write_lines(valid_images_txt, [row["image"] for row in valid_pairs])
    write_lines(valid_labels_txt, [row["label"] for row in valid_pairs])
    write_lines(invalid_images_txt, invalid_images)
    write_lines(invalid_labels_txt, invalid_labels)
    write_lines(invalid_pairs_txt, invalid_pair_reasons)
    write_lines(missing_images_txt, [f"{k}\t{label_map[k]}" for k in only_labels])
    write_lines(missing_labels_txt, [f"{k}\t{image_map[k]}" for k in only_images])

    summary = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "allow_empty_labels": bool(args.allow_empty_labels),
        "total_images": len(image_map),
        "total_labels": len(label_map),
        "paired_candidates": len(both_keys),
        "valid_pairs": len(valid_pairs),
        "invalid_pairs": len(both_keys) - len(valid_pairs),
        "missing_label_for_image": len(only_images),
        "missing_image_for_label": len(only_labels),
        "outputs": {
            "valid_pairs_json": str(valid_pairs_path),
            "valid_images_txt": str(valid_images_txt),
            "valid_labels_txt": str(valid_labels_txt),
            "invalid_images_txt": str(invalid_images_txt),
            "invalid_labels_txt": str(invalid_labels_txt),
            "invalid_pairs_txt": str(invalid_pairs_txt),
            "missing_images_txt": str(missing_images_txt),
            "missing_labels_txt": str(missing_labels_txt),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
