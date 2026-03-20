#!/usr/bin/env python3
"""
Lightweight direct accuracy probe for Arch0 full-image SR inference.

Purpose
- Evaluate Arch0 on the same LR/HR subset using direct TP/FP/FN@0.5 matching.
- Keep the metric axis aligned with arch2_accuracy_probe.py and arch4_wiring_check.py.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for arch0_accuracy_probe.py") from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Arch0 accuracy on LR validation images.")
    parser.add_argument("--project_root", type=str, default=None, help="Repo root to prepend to sys.path.")
    parser.add_argument("--arch0_config", type=str, required=True, help="Path to arch0 YAML config.")
    parser.add_argument("--lr_images_dir", type=str, required=True, help="Directory of LR validation images.")
    parser.add_argument("--hr_labels_dir", type=str, required=True, help="Directory of HR YOLO labels.")
    parser.add_argument("--max_images", type=int, default=500, help="Maximum images to process.")
    parser.add_argument("--device", type=str, default="cuda", help="Device string, e.g. cuda or cpu.")
    parser.add_argument("--half", action="store_true", help="Use torch autocast(fp16) on CUDA.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="Detector IoU threshold.")
    parser.add_argument("--sr_weights", type=str, default=None, help="Optional override for model.weights.sr_model.")
    parser.add_argument("--yolo_weights", type=str, default=None, help="Optional override for model.weights.detector.")
    parser.add_argument("--out_json", type=str, required=True, help="Where to save the summary JSON.")
    parser.add_argument("--print_every", type=int, default=50, help="Progress logging frequency.")
    return parser.parse_args()


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def ensure_project_root(args: argparse.Namespace) -> Optional[Path]:
    candidates: List[Path] = []
    if args.project_root:
        candidates.append(Path(args.project_root).resolve())

    cfg_path = Path(args.arch0_config).resolve()
    parts = cfg_path.parts
    if "configs" in parts:
        idx = parts.index("configs")
        candidates.append(Path(*parts[:idx]).resolve())

    for cand in candidates:
        if cand.exists():
            sys.path.insert(0, str(cand))
            return cand
    return None


def patch_config_dict(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))
    cfg["device"] = args.device
    model = cfg.setdefault("model", {})
    weights_cfg = model.setdefault("weights", {})
    yolo_cfg = model.setdefault("yolo", {})
    if args.sr_weights:
        weights_cfg["sr_model"] = args.sr_weights
    if args.yolo_weights:
        weights_cfg["detector"] = args.yolo_weights
        yolo_cfg["weights_path"] = args.yolo_weights
    return cfg


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


def load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor.unsqueeze(0)


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


def match_predictions(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_classes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    iou_thresh: float = 0.5,
) -> Dict[str, int]:
    if pred_boxes.numel() == 0:
        return {"tp50": 0, "fp50": 0, "fn50": int(gt_boxes.shape[0]), "num_preds": 0, "num_gts": int(gt_boxes.shape[0])}
    if gt_boxes.numel() == 0:
        return {
            "tp50": 0,
            "fp50": int(pred_boxes.shape[0]),
            "fn50": 0,
            "num_preds": int(pred_boxes.shape[0]),
            "num_gts": 0,
        }

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    pred_classes = pred_classes[order]

    gt_used = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool)
    tp = 0
    fp = 0
    for i in range(pred_boxes.shape[0]):
        same_class = (gt_classes == pred_classes[i]) & (~gt_used)
        if same_class.sum().item() == 0:
            fp += 1
            continue
        candidate_idx = torch.nonzero(same_class, as_tuple=False).squeeze(1)
        ious = box_iou(pred_boxes[i:i + 1], gt_boxes[candidate_idx]).squeeze(0)
        best_iou, best_pos = torch.max(ious, dim=0)
        if best_iou.item() >= iou_thresh:
            gt_used[candidate_idx[best_pos]] = True
            tp += 1
        else:
            fp += 1
    fn = int((~gt_used).sum().item())
    return {
        "tp50": int(tp),
        "fp50": int(fp),
        "fn50": fn,
        "num_preds": int(pred_boxes.shape[0]),
        "num_gts": int(gt_boxes.shape[0]),
    }


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class AutocastContext:
    def __init__(self, enabled: bool, device: torch.device):
        self.enabled = bool(enabled and device.type == "cuda" and torch.cuda.is_available())
        self.device = device

    def __enter__(self):
        if self.enabled:
            self.ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
            return self.ctx.__enter__()
        self.ctx = nullcontext()
        return self.ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self.ctx.__exit__(exc_type, exc, tb)


def detections_to_cpu(det: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes = det["boxes"].detach().cpu().float()
    scores = det["scores"].detach().cpu().float()
    classes = det["classes"].detach().cpu().long()
    return boxes, scores, classes


def main() -> None:
    args = parse_args()
    project_root = ensure_project_root(args)
    config_dict = patch_config_dict(read_yaml(Path(args.arch0_config)), args)

    from src.models.pipelines.arch0_sequential import Arch0Sequential

    model = Arch0Sequential(config_dict)
    model.eval()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = model.to(device)

    upscale_factor = int(config_dict.get("data", {}).get("upscale_factor", 4))
    pairs = pair_dataset(Path(args.lr_images_dir), Path(args.hr_labels_dir), args.max_images)

    acc: Dict[str, Any] = {
        "num_images": 0,
        "tp50": 0,
        "fp50": 0,
        "fn50": 0,
        "num_preds": 0,
        "num_gts": 0,
        "elapsed_ms_total": 0.0,
    }

    print(f"[arch0_probe] project_root : {project_root}")
    print(f"[arch0_probe] config       : {Path(args.arch0_config).resolve()}")
    print(f"[arch0_probe] images       : {len(pairs)}")
    print(f"[arch0_probe] device       : {device}")

    for idx, (_, lr_path, label_path) in enumerate(pairs, start=1):
        lr_tensor = load_image_tensor(lr_path)
        _, _, lr_h, lr_w = lr_tensor.shape
        gt_boxes, gt_classes = load_yolo_labels(label_path, img_w=lr_w * upscale_factor, img_h=lr_h * upscale_factor)

        lr_tensor = lr_tensor.to(device, non_blocking=True)
        model.eval()
        sync_if_needed(device)
        t0 = time.perf_counter()
        with AutocastContext(args.half, device):
            output = model.inference(lr_tensor, conf_threshold=args.conf, iou_threshold=args.iou)
        sync_if_needed(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        det = output["detections"][0]
        pred_boxes, pred_scores, pred_classes = detections_to_cpu(det)
        matched = match_predictions(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_classes=pred_classes,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            iou_thresh=0.5,
        )

        acc["num_images"] += 1
        acc["tp50"] += matched["tp50"]
        acc["fp50"] += matched["fp50"]
        acc["fn50"] += matched["fn50"]
        acc["num_preds"] += matched["num_preds"]
        acc["num_gts"] += matched["num_gts"]
        acc["elapsed_ms_total"] += elapsed_ms

        if args.print_every > 0 and (idx % args.print_every == 0 or idx == len(pairs)):
            print(f"[arch0_probe] processed {idx}/{len(pairs)} images")

    tp = acc["tp50"]
    fp = acc["fp50"]
    fn = acc["fn50"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    avg_ms = acc["elapsed_ms_total"] / max(1, acc["num_images"])
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    out = {
        "meta": {
            "arch0_config": str(Path(args.arch0_config).resolve()),
            "lr_images_dir": str(Path(args.lr_images_dir).resolve()),
            "hr_labels_dir": str(Path(args.hr_labels_dir).resolve()),
            "num_images": len(pairs),
            "device": str(device),
            "half_amp": bool(args.half and device.type == "cuda" and torch.cuda.is_available()),
            "upscale_factor": upscale_factor,
            "conf": args.conf,
            "iou": args.iou,
            "sr_weights": args.sr_weights,
            "yolo_weights": args.yolo_weights,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "results": {
            "arch0": {
                "num_images": acc["num_images"],
                "tp50": tp,
                "fp50": fp,
                "fn50": fn,
                "num_preds": acc["num_preds"],
                "num_gts": acc["num_gts"],
                "precision50_direct": precision,
                "recall50_direct": recall,
                "f1_50_direct": f1,
                "avg_ms_per_image": avg_ms,
                "fps": fps,
            }
        },
    }

    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("[arch0_probe] saved ->", out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
