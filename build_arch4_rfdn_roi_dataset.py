#!/usr/bin/env python3
"""
Build an SR training dataset from Arch4 uncertain ROI crops.

Goal
----
Train the SR model on the same crop domain that Arch4 sees at inference:
  LR image -> Scout -> uncertain ROI groups -> crop -> SR -> Sniper

Instead of using full-image SR pairs, this builder extracts aligned
``(LR ROI crop, HR ROI crop)`` pairs from the train/val split so that RFDN can
be retrained on the ROI regime directly.

Outputs
-------
out_root/
  lr/train/*.png
  lr/val/*.png
  hr/train/*.png
  hr/val/*.png
  metadata.csv
  build_stats.json
  data.yaml
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parent


PROJECT_ROOT = project_root_from_script()

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_with_path_block(data: dict, split: str):
    if split not in data:
        return None
    base = Path(data.get("path", ""))
    p = Path(data[split])
    if not p.is_absolute():
        p = base / p
    return p


def _resolve_custom_images_dir(data: dict, split: str, modality: str):
    root_key = f"{modality}_root"
    split_key = f"{modality}_{split}_images"
    if root_key in data and split_key in data:
        return Path(data[root_key]) / data[split_key]
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
    parts = list(images_dir.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    except ValueError:
        parent = images_dir.parent.parent if len(images_dir.parents) >= 2 else images_dir.parent
        return parent / "labels" / images_dir.name


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


def make_split_paths(hr_data_yaml: Path, lr_data_yaml: Path, split: str):
    lr_images_dir = resolve_images_dir(lr_data_yaml, split, modality="lr")
    hr_images_dir = resolve_images_dir(hr_data_yaml, split, modality="hr")
    hr_labels_dir = labels_dir_from_images_dir(hr_images_dir)
    return lr_images_dir, hr_images_dir, hr_labels_dir


def patch_for_roi_sr_build(
    cfg: dict,
    device: str,
    roi_expansion_override: float,
    crop_size_lr_override: int,
) -> dict:
    cfg = json.loads(json.dumps(cfg))
    model = cfg.setdefault("model", {})
    arch4 = model.setdefault("arch4", {})
    arch4.setdefault("drop_uncertain_if_sniper_hits", True)
    arch4.setdefault("scout_nms_iou", 0.50)
    arch4.setdefault("roi_merge_iou", 0.30)
    arch4.setdefault("roi_center_ratio", 0.35)
    arch4.setdefault("sniper_nms_iou", 0.45)
    arch4.setdefault("final_nms_iou", 0.50)
    arch4.setdefault("sniper_score_bonus", 0.0)
    if roi_expansion_override > 0:
        arch4["roi_expansion"] = roi_expansion_override
    if crop_size_lr_override > 0:
        arch4["crop_size_lr"] = crop_size_lr_override
    cfg["device"] = device
    return cfg


def extract_hr_crop_from_coord(
    hr_tensor: torch.Tensor,
    coord_lr: Tuple[int, int, int, int],
    upscale_factor: int,
    target_size: int,
) -> torch.Tensor:
    ix1, iy1, ix2, iy2 = coord_lr
    _, hr_h, hr_w = hr_tensor.shape
    scale = float(upscale_factor)

    hx1 = max(0, min(hr_w, int(round(ix1 * scale))))
    hy1 = max(0, min(hr_h, int(round(iy1 * scale))))
    hx2 = max(0, min(hr_w, int(round(ix2 * scale))))
    hy2 = max(0, min(hr_h, int(round(iy2 * scale))))

    if hx2 <= hx1 or hy2 <= hy1:
        raise ValueError(f"Invalid HR coord after scaling: {(hx1, hy1, hx2, hy2)} from LR {coord_lr}")

    crop = hr_tensor[:, hy1:hy2, hx1:hx2].unsqueeze(0)
    if crop.numel() == 0:
        raise ValueError(f"Empty HR crop for coord_lr={coord_lr}")

    resized = F.interpolate(
        crop,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )[0]
    return resized


def count_gt_boxes_inside_roi(lr_boxes_abs: np.ndarray, coord: Tuple[int, int, int, int]) -> int:
    if lr_boxes_abs.shape[0] == 0:
        return 0
    ix1, iy1, ix2, iy2 = coord
    count = 0
    for _, x1, y1, x2, y2 in lr_boxes_abs.tolist():
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        if ix1 <= cx <= ix2 and iy1 <= cy <= iy2:
            count += 1
    return count


def select_uncertain_subset(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    min_conf: float,
    max_conf: float,
):
    mask = (scores >= float(min_conf)) & (scores < float(max_conf))
    return boxes[mask], scores[mask], classes[mask], mask


def save_dataset_yaml(out_root: Path):
    data_yaml = {
        "path": str(out_root),
        "train": {
            "lr": "lr/train",
            "hr": "hr/train",
        },
        "val": {
            "lr": "lr/val",
            "hr": "hr/val",
        },
        "metadata": "metadata.csv",
    }
    with open(out_root / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)


def build_dataset(args):
    from src.models.pipelines.arch4_roi_awareNMS import Arch4RoiAwareNMS

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = patch_for_roi_sr_build(
        load_yaml(Path(args.arch4_config)),
        device=args.device,
        roi_expansion_override=args.roi_expansion_override,
        crop_size_lr_override=args.crop_size_lr_override,
    )

    out_root = Path(args.out_root).expanduser().resolve()
    for split in args.splits:
        (out_root / "lr" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "hr" / split).mkdir(parents=True, exist_ok=True)

    print("[BUILD] instantiate Arch4RoiAwareNMS ...")
    model = Arch4RoiAwareNMS(cfg)
    model.eval()

    uncertain_min_conf = (
        float(args.uncertain_min_conf)
        if args.uncertain_min_conf is not None
        else float(model.cfg.pass1_conf)
    )
    uncertain_max_conf = (
        float(args.uncertain_max_conf)
        if args.uncertain_max_conf is not None
        else float(model.cfg.pass2_conf)
    )

    target_lr_size = int(model.cfg.crop_size_lr)
    target_hr_size = int(model.cfg.crop_size_lr * model.cfg.upscale_factor)

    rows: List[Dict[str, object]] = []
    split_stats: Dict[str, Dict[str, object]] = {}

    for split in args.splits:
        lr_images_dir, hr_images_dir, hr_labels_dir = make_split_paths(
            Path(args.hr_data_yaml),
            Path(args.lr_data_yaml),
            split,
        )
        lr_images = list_images(lr_images_dir)
        hr_index = build_stem_index(hr_images_dir)

        if args.max_images > 0:
            lr_images = lr_images[: args.max_images]

        num_saved = 0
        num_pos = 0
        num_neg = 0
        num_total_roi = 0

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
            label_path = hr_labels_dir / f"{stem}.txt"

            hr_boxes = load_yolo_labels_abs_hr(label_path, hr_w, hr_h)
            lr_boxes = hr_abs_to_lr_abs(hr_boxes, hr_w, hr_h, lr_w, lr_h)

            lr_tensor = cv2_to_tensor(lr_img).to(model.cfg.device)
            hr_tensor = cv2_to_tensor(hr_img).to(model.cfg.device)[0]

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

                uncertain_boxes, uncertain_scores, uncertain_classes, _ = select_uncertain_subset(
                    boxes,
                    scores,
                    classes,
                    min_conf=uncertain_min_conf,
                    max_conf=uncertain_max_conf,
                )

                roi_groups = model._build_roi_groups(
                    uncertain_boxes,
                    uncertain_scores,
                    uncertain_classes,
                )

                if args.max_rois_per_image > 0 and len(roi_groups) > args.max_rois_per_image:
                    roi_groups = sorted(
                        roi_groups,
                        key=lambda g: float(g["member_scores"].max().item()) if len(g["member_scores"]) else 0.0,
                        reverse=True,
                    )[: args.max_rois_per_image]

                num_total_roi += len(roi_groups)

                for gidx, group in enumerate(roi_groups):
                    merged_box = group["merged_box"].unsqueeze(0)
                    crops, coords = model._extract_crops(lr_tensor[0], merged_box)
                    if len(crops) == 0:
                        continue

                    lr_crop = crops[0]
                    coord = tuple(int(v) for v in coords[0])
                    hr_crop = extract_hr_crop_from_coord(
                        hr_tensor=hr_tensor,
                        coord_lr=coord,
                        upscale_factor=int(model.cfg.upscale_factor),
                        target_size=target_hr_size,
                    )

                    num_gt_boxes = count_gt_boxes_inside_roi(lr_boxes, coord)
                    is_positive = num_gt_boxes > 0
                    if (not is_positive) and (random.random() > args.neg_keep_prob):
                        continue

                    out_stem = f"{stem}_roi{gidx:03d}"
                    out_lr = out_root / "lr" / split / f"{out_stem}.png"
                    out_hr = out_root / "hr" / split / f"{out_stem}.png"

                    cv2.imwrite(str(out_lr), tensor_to_bgr_u8(lr_crop))
                    cv2.imwrite(str(out_hr), tensor_to_bgr_u8(hr_crop))

                    max_member_score = (
                        float(group["member_scores"].max().item())
                        if len(group["member_scores"])
                        else 0.0
                    )
                    rows.append({
                        "split": split,
                        "source_stem": stem,
                        "crop_stem": out_stem,
                        "coord_lr": json.dumps(coord),
                        "coord_hr": json.dumps([
                            int(round(coord[0] * model.cfg.upscale_factor)),
                            int(round(coord[1] * model.cfg.upscale_factor)),
                            int(round(coord[2] * model.cfg.upscale_factor)),
                            int(round(coord[3] * model.cfg.upscale_factor)),
                        ]),
                        "num_gt_boxes": num_gt_boxes,
                        "is_positive": int(is_positive),
                        "num_member_boxes": int(group["member_boxes"].shape[0]),
                        "max_member_score": max_member_score,
                        "uncertain_min_conf": uncertain_min_conf,
                        "uncertain_max_conf": uncertain_max_conf,
                        "roi_expansion": float(model.cfg.roi_expansion),
                        "crop_size_lr": int(model.cfg.crop_size_lr),
                        "upscale_factor": int(model.cfg.upscale_factor),
                        "lr_path": str(lr_path),
                        "hr_path": str(hr_path),
                        "label_path": str(label_path),
                    })

                    num_saved += 1
                    if is_positive:
                        num_pos += 1
                    else:
                        num_neg += 1

            if idx % 200 == 0 or idx == len(lr_images):
                print(
                    f"  processed {idx}/{len(lr_images)} | roi={num_total_roi} "
                    f"| saved={num_saved} pos={num_pos} neg={num_neg}"
                )

        split_stats[split] = {
            "images": len(lr_images),
            "roi_groups": num_total_roi,
            "saved": num_saved,
            "positive": num_pos,
            "negative": num_neg,
            "uncertain_min_conf": uncertain_min_conf,
            "uncertain_max_conf": uncertain_max_conf,
            "roi_expansion": float(model.cfg.roi_expansion),
            "crop_size_lr": int(model.cfg.crop_size_lr),
            "crop_size_hr": target_hr_size,
        }

    metadata_path = out_root / "metadata.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "split", "source_stem", "crop_stem", "coord_lr", "coord_hr",
        "num_gt_boxes", "is_positive", "num_member_boxes", "max_member_score",
        "uncertain_min_conf", "uncertain_max_conf", "roi_expansion",
        "crop_size_lr", "upscale_factor", "lr_path", "hr_path", "label_path",
    ]
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(out_root / "build_stats.json", "w") as f:
        json.dump({"split_stats": split_stats, "args": vars(args)}, f, indent=2)

    save_dataset_yaml(out_root)

    print("\n=== BUILD DONE ===")
    print(f"out_root : {out_root}")
    print(f"metadata : {metadata_path}")
    print(json.dumps(split_stats, indent=2))


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
    p.add_argument("--neg_keep_prob", type=float, default=0.30)
    p.add_argument("--roi_expansion_override", type=float, default=0.0)
    p.add_argument("--crop_size_lr_override", type=int, default=0)
    p.add_argument("--uncertain_min_conf", type=float, default=None)
    p.add_argument("--uncertain_max_conf", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
