#!/usr/bin/env python3
"""
Dump Arch4 ROI-SR crops and crop-local YOLO labels for Sniper fine-tuning.

This script reuses the current Arch4 Scout -> ROI -> crop -> SR path and saves:
- 256x256 SR crop images (JPEG)
- crop-local YOLO labels derived from HR GT labels
- split-level stats and resume metadata
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump ROI-SR crops for Arch4 Sniper fine-tuning.")
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument("--arch4_config", type=str, required=True)
    parser.add_argument("--arch4_py", type=str, required=True)
    parser.add_argument("--lr_images_dir", type=str, required=True)
    parser.add_argument("--hr_images_dir", type=str, required=True)
    parser.add_argument("--hr_labels_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--sr_weights", type=str, default=None)
    parser.add_argument("--yolo_weights_lr", type=str, default=None)
    parser.add_argument("--yolo_weights_hr", type=str, default=None)
    parser.add_argument("--sniper_imgsz_mode", type=str, default=None)
    parser.add_argument("--sniper_imgsz_fixed", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--print_every", type=int, default=500)
    parser.add_argument("--checkpoint_every", type=int, default=10000)
    return parser.parse_args()


def sanitize_key(key: str) -> str:
    return key.replace("/", "__").replace("\\", "__")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def tensor_to_jpeg(t: torch.Tensor, out_path: Path, quality: int) -> None:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype("uint8")
    Image.fromarray(arr).save(out_path, format="JPEG", quality=int(quality), subsampling=0)


def save_label_file(out_path: Path, rows: Sequence[str]) -> None:
    if rows:
        out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    else:
        out_path.write_text("", encoding="utf-8")


def map_gt_to_crop_labels(
    gt_boxes_hr: torch.Tensor,
    gt_classes: torch.Tensor,
    hr_w: int,
    hr_h: int,
    upscale_factor: float,
    crop_coord_lr: Tuple[int, int, int, int],
) -> List[str]:
    rows: List[str] = []
    if gt_boxes_hr.numel() == 0:
        return rows

    ix1, iy1, ix2, iy2 = [int(v) for v in crop_coord_lr]
    crop_w_lr = max(1.0, float(ix2 - ix1))
    crop_h_lr = max(1.0, float(iy2 - iy1))

    boxes_hr = gt_boxes_hr.detach().cpu().float()
    classes = gt_classes.detach().cpu().long()

    lr_scale = float(upscale_factor)
    boxes_lr = boxes_hr / lr_scale

    for box_lr, cls in zip(boxes_lr, classes):
        x1_lr, y1_lr, x2_lr, y2_lr = [float(v) for v in box_lr.tolist()]
        cx_lr = 0.5 * (x1_lr + x2_lr)
        cy_lr = 0.5 * (y1_lr + y2_lr)
        w_lr = max(1e-6, x2_lr - x1_lr)
        h_lr = max(1e-6, y2_lr - y1_lr)

        if not (ix1 <= cx_lr <= ix2 and iy1 <= cy_lr <= iy2):
            continue

        cx_crop = cx_lr - float(ix1)
        cy_crop = cy_lr - float(iy1)

        cx_norm = clamp(cx_crop / crop_w_lr, 0.0, 1.0)
        cy_norm = clamp(cy_crop / crop_h_lr, 0.0, 1.0)
        w_norm = clamp(w_lr / crop_w_lr, 0.001, 1.0)
        h_norm = clamp(h_lr / crop_h_lr, 0.001, 1.0)

        rows.append(f"{int(cls)} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    return rows


def write_data_yaml(out_dir: Path) -> None:
    text = "\n".join(
        [
            f"path: {out_dir}",
            "train: images/train",
            "val: images/val",
            "nc: 1",
            "names:",
            "  0: ship",
            "",
        ]
    )
    (out_dir / "data.yaml").write_text(text, encoding="utf-8")


def load_processed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_stats(path: Path, split: str, processed_count: int) -> Dict[str, Any]:
    if path.exists():
        stats = json.loads(path.read_text(encoding="utf-8"))
    else:
        stats = {
            "split": split,
            "processed_images": 0,
            "total_images": 0,
            "total_roi_groups": 0,
            "total_crops": 0,
            "positive_crops": 0,
            "negative_crops": 0,
            "total_gt_in_positive_crops": 0,
            "avg_gt_per_positive_crop": 0.0,
            "crop_per_image_mean": 0.0,
            "elapsed_sec": 0.0,
            "updated_at": None,
        }
    stats["processed_images"] = max(int(stats.get("processed_images", 0)), int(processed_count))
    return stats


def main() -> None:
    args = parse_args()

    sys.path.insert(0, args.project_root)
    from iac_jetson.arch4_wiring_check import (  # pylint: disable=import-error
        AutocastContext,
        ensure_project_root,
        load_arch4_class,
        load_image_tensor,
        load_yolo_labels,
        pair_dataset,
        patch_config_dict,
        read_yaml,
        sync_if_needed,
    )

    ensure_project_root(args)
    cfg = patch_config_dict(read_yaml(Path(args.arch4_config)), args)
    cfg.setdefault("model", {}).setdefault("arch4", {})
    cfg["model"]["arch4"]["crop_refine_mode"] = "sr"

    out_dir = Path(args.out_dir).resolve()
    img_out_dir = out_dir / "images" / args.split
    label_out_dir = out_dir / "labels" / args.split
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)
    write_data_yaml(out_dir)

    processed_keys_path = out_dir / f"processed_keys_{args.split}.txt"
    stats_ckpt_path = out_dir / f"stats_checkpoint_{args.split}.json"
    stats_final_path = out_dir / f"stats_{args.split}.json"

    processed_keys = load_processed_keys(processed_keys_path)
    stats = load_stats(stats_ckpt_path, args.split, len(processed_keys))

    pairs = pair_dataset(
        lr_images_dir=Path(args.lr_images_dir),
        hr_images_dir=Path(args.hr_images_dir),
        hr_labels_dir=Path(args.hr_labels_dir),
        max_images=args.max_images,
    )
    stats["total_images"] = len(pairs)

    Arch4Class = load_arch4_class(args)
    model = Arch4Class(cfg)
    model.eval()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    crop_mode = getattr(model.cfg, "crop_refine_mode", "sr")
    if str(crop_mode).lower() != "sr":
        model.cfg.crop_refine_mode = "sr"

    upscale = float(getattr(model.cfg, "upscale_factor", cfg.get("data", {}).get("upscale_factor", 4)))
    start_time = time.time()

    print(f"[arch4_dump_sniper_crops] split={args.split} total_pairs={len(pairs)} resume_done={len(processed_keys)}")

    for index, (key, lr_path, hr_path, label_path) in enumerate(pairs, start=1):
        safe_key = sanitize_key(key)
        if safe_key in processed_keys:
            continue

        hr_w, hr_h = Image.open(hr_path).size
        gt_boxes_hr, gt_classes = load_yolo_labels(label_path, img_w=hr_w, img_h=hr_h)

        lr_tensor = load_image_tensor(lr_path).to(device, non_blocking=True)
        with AutocastContext(args.half, device):
            pass1_preds = model.scout_detector.predict(
                lr_tensor,
                conf=model.cfg.pass1_conf,
                iou=model.cfg.scout_nms_iou,
            )
        det = model._apply_batched_nms(pass1_preds[0], model.cfg.scout_nms_iou)  # type: ignore[attr-defined]

        boxes = det["boxes"]
        scores = det["scores"]
        classes = det["classes"]
        uncertain_mask = scores < model.cfg.pass2_conf
        uncertain_boxes = boxes[uncertain_mask]
        uncertain_scores = scores[uncertain_mask]
        uncertain_classes = classes[uncertain_mask]

        roi_groups = model._build_roi_groups(  # type: ignore[attr-defined]
            uncertain_boxes,
            uncertain_scores,
            uncertain_classes,
        )
        stats["total_roi_groups"] += int(len(roi_groups))

        all_crops_lr: List[torch.Tensor] = []
        crop_meta: List[Dict[str, Any]] = []

        for roi_idx, group in enumerate(roi_groups):
            merged_box = group["merged_box"].unsqueeze(0)
            crops, coords = model._extract_crops(lr_tensor[0], merged_box)  # type: ignore[attr-defined]
            if len(crops) == 0:
                continue
            group["coord"] = coords[0]
            group["roi_index"] = int(roi_idx)
            all_crops_lr.append(crops[0])
            crop_meta.append({"img_idx": 0, "group": group})

        if all_crops_lr:
            batch_crops_lr = torch.stack(all_crops_lr).to(device, non_blocking=True)
            with AutocastContext(args.half, device):
                batch_crops_refined = model._prepare_sniper_inputs(  # type: ignore[attr-defined]
                    batch_crops_lr=batch_crops_lr,
                    crop_metadata=crop_meta,
                    hr_images=None,
                )
            sync_if_needed(device)
        else:
            batch_crops_refined = torch.empty((0, 3, int(model.cfg.crop_size_lr * upscale), int(model.cfg.crop_size_lr * upscale)), device=device)

        for crop_idx, meta in enumerate(crop_meta):
            roi_name = f"{safe_key}_roi{crop_idx:03d}"
            image_out = img_out_dir / f"{roi_name}.jpg"
            label_out = label_out_dir / f"{roi_name}.txt"

            if image_out.exists() and label_out.exists():
                continue

            coord = tuple(int(v) for v in meta["group"]["coord"])
            label_rows = map_gt_to_crop_labels(
                gt_boxes_hr=gt_boxes_hr,
                gt_classes=gt_classes,
                hr_w=hr_w,
                hr_h=hr_h,
                upscale_factor=upscale,
                crop_coord_lr=coord,
            )

            tensor_to_jpeg(batch_crops_refined[crop_idx], image_out, args.jpeg_quality)
            save_label_file(label_out, label_rows)

            stats["total_crops"] += 1
            if label_rows:
                stats["positive_crops"] += 1
                stats["total_gt_in_positive_crops"] += len(label_rows)
            else:
                stats["negative_crops"] += 1

        processed_keys.add(safe_key)
        with processed_keys_path.open("a", encoding="utf-8") as f:
            f.write(safe_key + "\n")
        stats["processed_images"] = len(processed_keys)

        if index % args.print_every == 0:
            elapsed = time.time() - start_time
            crop_mean = stats["total_crops"] / max(1, stats["processed_images"])
            pos_ratio = stats["positive_crops"] / max(1, stats["total_crops"])
            print(
                f"[{args.split}] {index}/{len(pairs)} images | "
                f"processed={stats['processed_images']} | crops={stats['total_crops']} | "
                f"pos={stats['positive_crops']} neg={stats['negative_crops']} | "
                f"pos_ratio={pos_ratio:.4f} | crop/img={crop_mean:.3f} | elapsed={elapsed/60:.1f}m"
            )

        if index % args.checkpoint_every == 0:
            stats["elapsed_sec"] = time.time() - start_time
            stats["avg_gt_per_positive_crop"] = (
                stats["total_gt_in_positive_crops"] / max(1, stats["positive_crops"])
            )
            stats["crop_per_image_mean"] = stats["total_crops"] / max(1, stats["processed_images"])
            stats["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            stats_ckpt_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        del lr_tensor, pass1_preds, det, boxes, scores, classes, uncertain_boxes, uncertain_scores, uncertain_classes
        if all_crops_lr:
            del batch_crops_lr, batch_crops_refined
        if index % 1000 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

    stats["elapsed_sec"] = time.time() - start_time
    stats["avg_gt_per_positive_crop"] = (
        stats["total_gt_in_positive_crops"] / max(1, stats["positive_crops"])
    )
    stats["crop_per_image_mean"] = stats["total_crops"] / max(1, stats["processed_images"])
    stats["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats_ckpt_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    stats_final_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    merged_stats_path = out_dir / "stats.json"
    merged: Dict[str, Any] = {}
    if merged_stats_path.exists():
        merged = json.loads(merged_stats_path.read_text(encoding="utf-8"))
    merged[args.split] = stats
    merged_stats_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    print(
        f"[done] split={args.split} processed={stats['processed_images']} "
        f"crops={stats['total_crops']} pos={stats['positive_crops']} neg={stats['negative_crops']} "
        f"avg_gt/pos={stats['avg_gt_per_positive_crop']:.3f}"
    )


if __name__ == "__main__":
    main()
