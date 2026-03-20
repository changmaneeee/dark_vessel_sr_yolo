#!/usr/bin/env python3
"""
Scout recall diagnostic for Arch4.

Measure whether the LR Scout detector covers each GT object before the
uncertain/ROI/Sniper stages. This separates "Scout never saw it" from
"Scout saw it but downstream lost it".
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
import torch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Scout recall against HR GT labels projected into LR space.")
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument("--scout_weights", type=str, required=True)
    parser.add_argument("--lr_images_dir", type=str, required=True)
    parser.add_argument("--hr_labels_dir", type=str, required=True)
    parser.add_argument("--upscale_factor", type=float, default=4.0)
    parser.add_argument("--scout_conf", type=float, default=0.1)
    parser.add_argument("--scout_iou", type=float, default=0.5)
    parser.add_argument("--match_iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--print_every", type=int, default=500)
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


def pair_dataset(lr_images_dir: Path, hr_labels_dir: Path, max_images: int) -> List[Tuple[str, Path, Path]]:
    lr_map = build_stem_map(lr_images_dir, IMAGE_EXTS)
    label_map = build_stem_map(hr_labels_dir, {".txt"})
    keys = sorted(set(lr_map.keys()) & set(label_map.keys()))
    if max_images > 0:
        keys = keys[:max_images]
    if not keys:
        raise FileNotFoundError(
            f"No paired samples found across\n  LR: {lr_images_dir}\n  Labels: {hr_labels_dir}"
        )
    return [(key, lr_map[key], label_map[key]) for key in keys]


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> Tuple[torch.Tensor, torch.Tensor]:
    boxes: List[List[float]] = []
    classes: List[int] = []
    if not label_path.exists():
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

    with label_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            x1 = (xc - w / 2.0) * img_w
            y1 = (yc - h / 2.0) * img_h
            x2 = (xc + w / 2.0) * img_w
            y2 = (yc + h / 2.0) * img_h
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)

    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.long)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(classes, dtype=torch.long)


def box_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    if box_a.numel() == 0 or box_b.numel() == 0:
        return torch.zeros((box_a.shape[0], box_b.shape[0]), dtype=torch.float32)
    a = box_a.float()
    b = box_b.float()
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))[None, :]
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-6)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.project_root)

    from ultralytics import YOLO

    scout = YOLO(args.scout_weights)

    pairs = pair_dataset(
        lr_images_dir=Path(args.lr_images_dir),
        hr_labels_dir=Path(args.hr_labels_dir),
        max_images=args.max_images,
    )
    print(f"[scout_diagnostic] paired images={len(pairs)}")

    total_gt = 0
    gt_found_by_scout = 0
    gt_missed_by_scout = 0
    total_scout_boxes = 0
    total_images = 0
    confident_matched = 0
    uncertain_matched = 0
    unmatched_images = 0
    all_matched_scores: List[float] = []
    per_image_details: List[Dict[str, object]] = []

    for idx, (stem, img_path, label_path) in enumerate(pairs, start=1):
        with Image.open(img_path) as img:
            lr_w, lr_h = img.size

        hr_w = int(round(lr_w * args.upscale_factor))
        hr_h = int(round(lr_h * args.upscale_factor))
        gt_boxes_hr, gt_classes = load_yolo_labels(label_path, img_w=hr_w, img_h=hr_h)
        if gt_boxes_hr.numel() == 0:
            continue

        gt_boxes_lr = gt_boxes_hr / float(args.upscale_factor)

        results = scout.predict(
            str(img_path),
            conf=args.scout_conf,
            iou=args.scout_iou,
            verbose=False,
            device=args.device,
        )
        boxes_obj = results[0].boxes
        if boxes_obj is not None and len(boxes_obj) > 0:
            scout_boxes = boxes_obj.xyxy.detach().cpu().float()
            scout_scores = boxes_obj.conf.detach().cpu().float()
        else:
            scout_boxes = torch.empty((0, 4), dtype=torch.float32)
            scout_scores = torch.empty((0,), dtype=torch.float32)

        total_images += 1
        total_gt += int(gt_boxes_lr.shape[0])
        total_scout_boxes += int(scout_boxes.shape[0])

        gt_matched = [False] * int(gt_boxes_lr.shape[0])
        matched_scores: List[float] = []
        matched_confident = 0
        matched_uncertain = 0
        best_ious: List[float] = []

        if scout_boxes.numel() > 0:
            iou_mat = box_iou(gt_boxes_lr, scout_boxes)
            for g in range(gt_boxes_lr.shape[0]):
                row = iou_mat[g]
                best_iou = float(row.max().item()) if row.numel() > 0 else 0.0
                best_ious.append(best_iou)
                if best_iou >= args.match_iou:
                    gt_matched[g] = True
                    best_j = int(row.argmax().item())
                    best_score = float(scout_scores[best_j].item())
                    matched_scores.append(best_score)
                    all_matched_scores.append(best_score)
                    if best_score >= 0.45:
                        confident_matched += 1
                        matched_confident += 1
                    else:
                        uncertain_matched += 1
                        matched_uncertain += 1
                else:
                    best_ious.append(best_iou)
        else:
            best_ious = [0.0] * int(gt_boxes_lr.shape[0])

        found = int(sum(gt_matched))
        missed = int(gt_boxes_lr.shape[0] - found)
        gt_found_by_scout += found
        gt_missed_by_scout += missed
        if missed > 0:
            unmatched_images += 1

        per_image_details.append(
            {
                "stem": stem,
                "n_gt": int(gt_boxes_lr.shape[0]),
                "n_scout_boxes": int(scout_boxes.shape[0]),
                "gt_found": found,
                "gt_missed": missed,
                "matched_scores": matched_scores,
                "matched_confident": matched_confident,
                "matched_uncertain": matched_uncertain,
                "best_iou_per_gt": best_ious[: int(gt_boxes_lr.shape[0])],
            }
        )

        if idx % args.print_every == 0:
            running_recall = gt_found_by_scout / max(1, total_gt)
            print(f"  [{idx}/{len(pairs)}] scout recall@{args.match_iou:.2f}={running_recall:.4f}")

    scout_recall = gt_found_by_scout / max(1, total_gt)
    score_bins = {
        "below_0.25": sum(1 for s in all_matched_scores if s < 0.25),
        "0.25_to_0.45": sum(1 for s in all_matched_scores if 0.25 <= s < 0.45),
        "0.45_to_0.60": sum(1 for s in all_matched_scores if 0.45 <= s < 0.60),
        "0.60_to_0.80": sum(1 for s in all_matched_scores if 0.60 <= s < 0.80),
        "above_0.80": sum(1 for s in all_matched_scores if s >= 0.80),
    }

    summary = {
        "total_images": total_images,
        "total_gt": total_gt,
        "total_scout_boxes": total_scout_boxes,
        "avg_scout_boxes_per_image": total_scout_boxes / max(1, total_images),
        "gt_found_by_scout": gt_found_by_scout,
        "gt_missed_by_scout": gt_missed_by_scout,
        "scout_recall_at_iou50": scout_recall,
        "scout_conf_used": args.scout_conf,
        "scout_iou_used": args.scout_iou,
        "match_iou_used": args.match_iou,
        "matched_score_distribution": score_bins,
        "matched_score_mean": float(np.mean(all_matched_scores)) if all_matched_scores else 0.0,
        "matched_score_median": float(np.median(all_matched_scores)) if all_matched_scores else 0.0,
        "matched_confident_count": confident_matched,
        "matched_uncertain_count": uncertain_matched,
        "matched_confident_ratio": confident_matched / max(1, gt_found_by_scout),
        "matched_uncertain_ratio": uncertain_matched / max(1, gt_found_by_scout),
        "images_with_any_gt_missed": unmatched_images,
        "per_image_details": per_image_details,
        "interpretation": {
            "if_recall_above_085": "Scout는 대부분 찾고 있음. FN은 downstream(crop/merge)에서 발생. Scout 재학습보다 downstream 개선이 우선.",
            "if_recall_070_085": "Scout가 중간 수준. Scout 개선과 downstream 개선 둘 다 필요.",
            "if_recall_below_070": "Scout 자체가 병목. Scout 재학습이 최우선.",
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[scout_diagnostic] === RESULTS ===")
    print(f"  Total GT objects: {total_gt}")
    print(f"  Scout found (IoU>={args.match_iou}): {gt_found_by_scout} ({scout_recall:.4f})")
    print(f"  Scout missed: {gt_missed_by_scout} ({1.0 - scout_recall:.4f})")
    print(f"  Avg scout boxes/img: {total_scout_boxes / max(1, total_images):.2f}")
    print(f"  Matched score mean/median: {summary['matched_score_mean']:.4f} / {summary['matched_score_median']:.4f}")
    print(f"  Matched confident/uncertain: {confident_matched} / {uncertain_matched}")
    print(f"  Score bins: {json.dumps(score_bins)}")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
