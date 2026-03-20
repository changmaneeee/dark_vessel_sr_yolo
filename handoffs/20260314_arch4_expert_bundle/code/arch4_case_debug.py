#!/usr/bin/env python3
"""
Focused case debugger for Arch4 ROI-aware ablation runtime.

Given a set of image keys, this script runs the same image through multiple
crop refinement modes and saves:
- input / GT overlays
- mode-specific final overlays
- a few LR / refined ROI crops
- per-case JSON with ROI-level debug info
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused Arch4 case debugger.")
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument("--arch4_config", type=str, required=True)
    parser.add_argument("--arch4_py", type=str, required=True)
    parser.add_argument("--lr_images_dir", type=str, required=True)
    parser.add_argument("--hr_images_dir", type=str, required=True)
    parser.add_argument("--hr_labels_dir", type=str, required=True)
    parser.add_argument("--keys", type=str, required=True, help="Comma-separated image keys.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--modes", type=str, default="sr,bilinear,hr_ref")
    parser.add_argument("--sniper_imgsz_mode", type=str, default="fixed")
    parser.add_argument("--sniper_imgsz_fixed", type=int, default=256)
    parser.add_argument("--sr_weights", type=str, default=None)
    parser.add_argument("--yolo_weights_lr", type=str, default=None)
    parser.add_argument("--yolo_weights_hr", type=str, default=None)
    parser.add_argument("--save_first_n_crops", type=int, default=3)
    return parser.parse_args()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def draw_boxes(base_tensor: torch.Tensor, boxes: List[List[float]], color: str) -> Image.Image:
    img = tensor_to_pil(base_tensor)
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
    return img


def save_overlay(base_tensor: torch.Tensor, gt_boxes: List[List[float]], pred_boxes: List[List[float]], out_path: Path) -> None:
    img = tensor_to_pil(base_tensor)
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in gt_boxes:
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=2)
    for x1, y1, x2, y2 in pred_boxes:
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
    img.save(out_path)


def main() -> None:
    args = parse_args()

    sys.path.insert(0, args.project_root)
    from iac_jetson.arch4_wiring_check import (
        AutocastContext,
        detections_to_eval_space,
        ensure_project_root,
        load_arch4_class,
        load_image_tensor,
        load_yolo_labels,
        match_predictions,
        maybe_pick_final_sig,
        pair_dataset,
        patch_config_dict,
        read_yaml,
        summarize_debug,
        sync_if_needed,
    )

    ensure_project_root(args)
    config_dict = patch_config_dict(read_yaml(Path(args.arch4_config)), args)
    Arch4Class = load_arch4_class(args)
    model = Arch4Class(config_dict)
    model.eval()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    upscale_factor = float(config_dict.get("data", {}).get("upscale_factor", 4))

    if hasattr(getattr(model, "cfg", None), "sniper_imgsz_mode"):
        model.cfg.sniper_imgsz_mode = args.sniper_imgsz_mode
    if hasattr(getattr(model, "cfg", None), "sniper_imgsz_fixed"):
        model.cfg.sniper_imgsz_fixed = args.sniper_imgsz_fixed

    forward_sig = inspect.signature(model.forward)
    supports_hr_images = "hr_images" in forward_sig.parameters
    supports_debug = "debug" in forward_sig.parameters
    supports_crop_mode = hasattr(getattr(model, "cfg", None), "crop_refine_mode")

    pairs = pair_dataset(
        lr_images_dir=Path(args.lr_images_dir),
        hr_images_dir=Path(args.hr_images_dir),
        hr_labels_dir=Path(args.hr_labels_dir),
        max_images=0,
    )
    pair_map = {key: (lr_path, hr_path, label_path) for key, lr_path, hr_path, label_path in pairs}

    keys = [key.strip() for key in args.keys.split(",") if key.strip()]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_all: Dict[str, Any] = {}

    for key in keys:
        if key not in pair_map:
            raise KeyError(f"Image key not found in paired dataset: {key}")

        lr_path, hr_path, label_path = pair_map[key]
        lr_tensor = load_image_tensor(lr_path)
        hr_tensor = load_image_tensor(hr_path)
        _, _, hr_h, hr_w = hr_tensor.shape
        gt_boxes_hr, gt_classes = load_yolo_labels(label_path, img_w=hr_w, img_h=hr_h)
        gt_boxes_list = gt_boxes_hr.detach().cpu().float().tolist()

        case_dir = out_dir / key
        case_dir.mkdir(parents=True, exist_ok=True)
        tensor_to_pil(lr_tensor[0]).save(case_dir / "lr_input.png")
        tensor_to_pil(hr_tensor[0]).save(case_dir / "hr_input.png")
        draw_boxes(hr_tensor[0], gt_boxes_list, "lime").save(case_dir / "hr_gt_overlay.png")

        lr_tensor = lr_tensor.to(device, non_blocking=True)
        hr_tensor = hr_tensor.to(device, non_blocking=True)

        case_summary: Dict[str, Any] = {
            "key": key,
            "gt_boxes_hr": gt_boxes_list,
            "modes": {},
        }

        for mode in modes:
            if supports_crop_mode:
                model.cfg.crop_refine_mode = mode

            kwargs: Dict[str, Any] = {}
            if supports_debug:
                kwargs["debug"] = True
            if mode == "hr_ref" and supports_hr_images:
                kwargs["hr_images"] = hr_tensor

            sync_if_needed(device)
            t0 = time.perf_counter()
            with AutocastContext(args.half, device):
                output = model.forward(lr_tensor, **kwargs)
            sync_if_needed(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            det_boxes_hr, det_scores, det_classes = detections_to_eval_space(
                output["detections"][0],
                upscale_factor=upscale_factor,
                eval_space="hr",
            )
            matched = match_predictions(
                pred_boxes=det_boxes_hr,
                pred_scores=det_scores,
                pred_classes=det_classes,
                gt_boxes=gt_boxes_hr,
                gt_classes=gt_classes,
                iou_thresh=0.5,
            )
            debug = output.get("debug_info", {}) or {}
            stage = summarize_debug(output)

            mode_dir = case_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            save_overlay(
                base_tensor=hr_tensor[0].cpu(),
                gt_boxes=gt_boxes_list,
                pred_boxes=det_boxes_hr.detach().cpu().float().tolist(),
                out_path=mode_dir / "final_plus_gt_overlay.png",
            )

            for idx, crop in enumerate((debug.get("crops_lr") or [])[: args.save_first_n_crops]):
                tensor_to_pil(crop).save(mode_dir / f"crop_lr_{idx:02d}.png")
            for idx, crop in enumerate((debug.get("crops_refined") or debug.get("crops_sr") or [])[: args.save_first_n_crops]):
                tensor_to_pil(crop).save(mode_dir / f"crop_refined_{idx:02d}.png")

            mode_summary = {
                "tp": matched["tp50"],
                "fp": matched["fp50"],
                "fn": matched["fn50"],
                "num_preds": matched["num_preds"],
                "elapsed_ms": round(elapsed_ms, 2),
                "final_boxes_hr": det_boxes_hr.detach().cpu().float().tolist(),
                "final_scores": det_scores.detach().cpu().float().tolist(),
                "final_classes": det_classes.detach().cpu().long().tolist(),
                "stage_summary": {
                    "roi_crops_total": stage.get("roi_crops_total", 0),
                    "pass2_raw_boxes_total": stage.get("pass2_raw_boxes_total", 0),
                    "pass2_after_nms_boxes_total": stage.get("pass2_after_nms_boxes_total", 0),
                    "sniper_hits": stage.get("stats.sniper_hit_groups_total"),
                    "fallback_groups": stage.get("stats.fallback_groups_total"),
                    "first_crop_refined_hash": stage.get("first_crop_refined_hash"),
                    "pass2_hash": stage.get("pass2_hash"),
                    "final_hash": maybe_pick_final_sig(output),
                },
                "roi_debug": debug.get("roi_debug", []),
                "final_records_pre_nms": debug.get("final_records_pre_nms", []),
                "final_records_post_nms": debug.get("final_records_post_nms", []),
            }
            case_summary["modes"][mode] = mode_summary

            with (mode_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(mode_summary, f, ensure_ascii=False, indent=2)

        with (case_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(case_summary, f, ensure_ascii=False, indent=2)
        summary_all[key] = case_summary

    with (out_dir / "summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(summary_all, f, ensure_ascii=False, indent=2)

    print("[arch4_case_debug] saved ->", out_dir)
    print(json.dumps({"out_dir": str(out_dir), "keys": keys, "modes": modes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
