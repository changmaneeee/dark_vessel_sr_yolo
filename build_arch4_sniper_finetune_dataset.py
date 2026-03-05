#!/usr/bin/env python3
"""
Build a YOLO-format fine-tuning dataset for Arch4 Sniper.

Idea
----
Use the *current best Arch4 front half* (Scout + ROI grouping + SR) to generate
realistic SR crops, then project HR ground-truth boxes into each crop.
This makes the Sniper detector train on the same kind of inputs it will see at inference.

Outputs
-------
out_root/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  data.yaml
  metadata.csv

Assumptions
-----------
- HR and LR datasets share the same image stem (e.g. abc123.jpg in both splits).
- HR labels are YOLO-format txt files normalized to the HR image size.
- Arch4 config points to the desired Scout detector / SR model.
- Sniper detector weights in the config are irrelevant here; we are building *training data*.
"""

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = project_root_from_script()

# Make repo imports work when this file is copied into the project.
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pipelines.arch4_roi_awareNMS import Arch4RoiAwareNMS  # noqa: E402


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitPaths:
    lr_images_dir: Path
    hr_images_dir: Path
    hr_labels_dir: Path


# -----------------------------------------------------------------------------
# Generic YAML path resolution helpers
# -----------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_with_path_block(data: dict, split: str) -> Optional[Path]:
    # Ultralytics-style YAML: path:, train:, val:
    if split not in data:
        return None
    base = Path(data.get("path", ""))
    p = Path(data[split])
    if not p.is_absolute():
        p = base / p
    return p


def _resolve_custom_images_dir(data: dict, split: str, modality: str) -> Optional[Path]:
    # Custom YAML variants like hr_root/hr_val_images, lr_root/lr_val_images
    root_key = f"{modality}_root"
    split_key = f"{modality}_{split}_images"
    if root_key in data and split_key in data:
        p = Path(data[root_key]) / data[split_key]
        return p
    return None


def resolve_images_dir(data_yaml: Path, split: str, modality: str) -> Path:
    data = load_yaml(data_yaml)
    p = _resolve_with_path_block(data, split)
    if p is None:
        p = _resolve_custom_images_dir(data, split, modality)
    if p is None:
        raise FileNotFoundError(
            f"Could not resolve images dir for split='{split}', modality='{modality}' from {data_yaml}"
        )
    return p.expanduser().resolve()


def labels_dir_from_images_dir(images_dir: Path) -> Path:
    # Typical dataset layout: .../images/train -> .../labels/train
    parts = list(images_dir.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    except ValueError:
        # Fallback: sibling folder named labels/<split>
        parent = images_dir.parent.parent if len(images_dir.parents) >= 2 else images_dir.parent
        return parent / "labels" / images_dir.name


# -----------------------------------------------------------------------------
# Image / label helpers
# -----------------------------------------------------------------------------

def build_stem_index(images_dir: Path) -> Dict[str, Path]:
    out = {}
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p
    return out


def list_images(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])


def cv2_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float() / 255.0
    return t.unsqueeze(0)


def tensor_to_bgr_u8(img_tensor_chw: torch.Tensor) -> np.ndarray:
    x = img_tensor_chw.detach().float().cpu().clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def load_yolo_labels_abs_hr(label_path: Path, hr_w: int, hr_h: int) -> np.ndarray:
    """Return Nx5 array: cls, x1, y1, x2, y2 in HR absolute pixels."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    rows = []
    with open(label_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            cls, xc, yc, w, h = map(float, s.split())
            x1 = (xc - w / 2.0) * hr_w
            y1 = (yc - h / 2.0) * hr_h
            x2 = (xc + w / 2.0) * hr_w
            y2 = (yc + h / 2.0) * hr_h
            rows.append([cls, x1, y1, x2, y2])
    return np.asarray(rows, dtype=np.float32)


def hr_abs_to_lr_abs(hr_boxes: np.ndarray, hr_w: int, hr_h: int, lr_w: int, lr_h: int) -> np.ndarray:
    if hr_boxes.shape[0] == 0:
        return hr_boxes.copy()
    sx = hr_w / float(lr_w)
    sy = hr_h / float(lr_h)
    out = hr_boxes.copy()
    out[:, 1] /= sx
    out[:, 3] /= sx
    out[:, 2] /= sy
    out[:, 4] /= sy
    return out


def yolo_label_lines_for_sr_crop(
    lr_boxes_abs: np.ndarray,
    coord: Tuple[int, int, int, int],
    crop_size_lr: int,
    upscale: int,
    min_box_px: float = 2.0,
) -> List[str]:
    """
    Map LR absolute boxes into the final SR crop coordinate system.

    The Arch4 crop pipeline is:
      original LR ROI -> resize to crop_size_lr x crop_size_lr -> SR x upscale
    so final crop size is crop_size_lr * upscale.
    """
    ix1, iy1, ix2, iy2 = coord
    roi_w = max(1, ix2 - ix1)
    roi_h = max(1, iy2 - iy1)
    out_size = crop_size_lr * upscale

    lines = []
    for row in lr_boxes_abs:
        cls_id, x1, y1, x2, y2 = row.tolist()
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        # Keep boxes whose center lies inside this ROI.
        if not (ix1 <= cx <= ix2 and iy1 <= cy <= iy2):
            continue

        x1_rel = ((x1 - ix1) / roi_w) * out_size
        y1_rel = ((y1 - iy1) / roi_h) * out_size
        x2_rel = ((x2 - ix1) / roi_w) * out_size
        y2_rel = ((y2 - iy1) / roi_h) * out_size

        x1_rel = float(np.clip(x1_rel, 0, out_size - 1))
        y1_rel = float(np.clip(y1_rel, 0, out_size - 1))
        x2_rel = float(np.clip(x2_rel, 0, out_size - 1))
        y2_rel = float(np.clip(y2_rel, 0, out_size - 1))

        bw = x2_rel - x1_rel
        bh = y2_rel - y1_rel
        if bw < min_box_px or bh < min_box_px:
            continue

        xc = (x1_rel + x2_rel) * 0.5 / out_size
        yc = (y1_rel + y2_rel) * 0.5 / out_size
        wn = bw / out_size
        hn = bh / out_size
        lines.append(f"{int(cls_id)} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    return lines


# -----------------------------------------------------------------------------
# Main build loop
# -----------------------------------------------------------------------------

def make_split_paths(hr_data_yaml: Path, lr_data_yaml: Path, split: str) -> SplitPaths:
    lr_images_dir = resolve_images_dir(lr_data_yaml, split, modality="lr")
    hr_images_dir = resolve_images_dir(hr_data_yaml, split, modality="hr")
    hr_labels_dir = labels_dir_from_images_dir(hr_images_dir)
    return SplitPaths(
        lr_images_dir=lr_images_dir,
        hr_images_dir=hr_images_dir,
        hr_labels_dir=hr_labels_dir,
    )


def patch_for_build(cfg: dict, device: str) -> dict:
    """Force build-friendly config values without changing core A11 strategy."""
    cfg = json.loads(json.dumps(cfg))  # deep-copy via json-serializable structure
    model = cfg.setdefault("model", {})
    arch4 = model.setdefault("arch4", {})
    arch4.setdefault("drop_uncertain_if_sniper_hits", True)
    arch4.setdefault("scout_nms_iou", 0.55)
    arch4.setdefault("roi_merge_iou", 0.25)
    arch4.setdefault("final_nms_iou", 0.45)
    arch4.setdefault("sniper_score_bonus", 0.0)
    cfg["device"] = device
    return cfg


def build_dataset(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = patch_for_build(load_yaml(Path(args.arch4_config)), args.device)

    out_root = Path(args.out_root).expanduser().resolve()
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    metadata_path = out_root / "metadata.csv"

    print("[BUILD] instantiate Arch4RoiAwareNMS ...")
    model = Arch4RoiAwareNMS(cfg)
    model.eval()

    rows = []
    split_stats = {}

    for split in args.splits:
        sp = make_split_paths(Path(args.hr_data_yaml), Path(args.lr_data_yaml), split)
        lr_images = list_images(sp.lr_images_dir)
        hr_index = build_stem_index(sp.hr_images_dir)

        if args.max_images > 0:
            lr_images = lr_images[: args.max_images]

        num_saved = 0
        num_pos = 0
        num_neg = 0

        print(f"\n[BUILD] split={split} | images={len(lr_images)}")
        for idx, lr_path in enumerate(lr_images, 1):
            stem = lr_path.stem
            hr_path = hr_index.get(stem)
            if hr_path is None:
                continue

            lr_img = cv2.imread(str(lr_path))
            hr_img = cv2.imread(str(hr_path))
            if lr_img is None or hr_img is None:
                continue

            lr_h, lr_w = lr_img.shape[:2]
            hr_h, hr_w = hr_img.shape[:2]

            label_path = sp.hr_labels_dir / f"{stem}.txt"
            hr_boxes = load_yolo_labels_abs_hr(label_path, hr_w, hr_h)
            lr_boxes = hr_abs_to_lr_abs(hr_boxes, hr_w, hr_h, lr_w, lr_h)

            lr_tensor = cv2_to_tensor(lr_img).to(model.cfg.device)

            with torch.no_grad():
                scout = model.scout_detector.predict(
                    lr_tensor,
                    conf=model.cfg.pass1_conf,
                    iou=model.cfg.scout_nms_iou,
                )[0]
                scout = model._apply_batched_nms(scout, model.cfg.scout_nms_iou)

                boxes = scout["boxes"]
                scores = scout["scores"]
                classes = scout["classes"]

                confident_mask = scores >= model.cfg.pass2_conf
                uncertain_boxes = boxes[~confident_mask]
                uncertain_scores = scores[~confident_mask]
                uncertain_classes = classes[~confident_mask]

                roi_groups = model._build_roi_groups(uncertain_boxes, uncertain_scores, uncertain_classes)

                # Optionally keep only top-K ROI groups by max member score.
                if args.max_rois_per_image > 0 and len(roi_groups) > args.max_rois_per_image:
                    roi_groups = sorted(
                        roi_groups,
                        key=lambda g: float(g["member_scores"].max().item()) if len(g["member_scores"]) else 0.0,
                        reverse=True,
                    )[: args.max_rois_per_image]

                for gidx, group in enumerate(roi_groups):
                    merged_box = group["merged_box"].unsqueeze(0)
                    crops, coords = model._extract_crops(lr_tensor[0], merged_box)
                    if len(crops) == 0:
                        continue

                    batch_lr = torch.stack([crops[0]]).to(model.cfg.device)
                    batch_hr = model._run_batch_sr(batch_lr)
                    sr_crop = batch_hr[0]
                    coord = coords[0]

                    label_lines = yolo_label_lines_for_sr_crop(
                        lr_boxes_abs=lr_boxes,
                        coord=coord,
                        crop_size_lr=int(model.cfg.crop_size_lr),
                        upscale=int(model.cfg.upscale_factor),
                        min_box_px=args.min_box_px,
                    )

                    is_positive = len(label_lines) > 0
                    if (not is_positive) and (random.random() > args.neg_keep_prob):
                        continue

                    out_stem = f"{stem}_roi{gidx:03d}"
                    out_img = out_root / "images" / split / f"{out_stem}.jpg"
                    out_lbl = out_root / "labels" / split / f"{out_stem}.txt"

                    cv2.imwrite(str(out_img), tensor_to_bgr_u8(sr_crop))
                    with open(out_lbl, "w") as f:
                        if label_lines:
                            f.write("\n".join(label_lines) + "\n")

                    num_saved += 1
                    if is_positive:
                        num_pos += 1
                    else:
                        num_neg += 1

                    rows.append({
                        "split": split,
                        "source_stem": stem,
                        "crop_stem": out_stem,
                        "coord": json.dumps(coord),
                        "num_gt_boxes": len(label_lines),
                        "is_positive": int(is_positive),
                        "lr_path": str(lr_path),
                        "hr_path": str(hr_path),
                    })

            if idx % 200 == 0 or idx == len(lr_images):
                print(f"  processed {idx}/{len(lr_images)} | saved={num_saved} pos={num_pos} neg={num_neg}")

        split_stats[split] = {
            "saved": num_saved,
            "positive": num_pos,
            "negative": num_neg,
        }

    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "split", "source_stem", "crop_stem", "coord", "num_gt_boxes", "is_positive", "lr_path", "hr_path"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    data_yaml = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "nc": args.num_classes,
        "names": args.names,
    }
    with open(out_root / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    with open(out_root / "build_stats.json", "w") as f:
        json.dump({"split_stats": split_stats, "args": vars(args)}, f, indent=2)

    print("\n=== BUILD DONE ===")
    print(f"out_root : {out_root}")
    print(f"data.yaml: {out_root / 'data.yaml'}")
    print(f"metadata : {metadata_path}")
    print(json.dumps(split_stats, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch4_config", required=True)
    p.add_argument("--hr_data_yaml", required=True)
    p.add_argument("--lr_data_yaml", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_images", type=int, default=0, help="0 means all images per split")
    p.add_argument("--max_rois_per_image", type=int, default=0, help="0 means keep all ROI groups")
    p.add_argument("--neg_keep_prob", type=float, default=0.30, help="keep-prob for negative SR crops")
    p.add_argument("--min_box_px", type=float, default=2.0)
    p.add_argument("--num_classes", type=int, default=1)
    p.add_argument("--names", nargs="+", default=["ship"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(args)
