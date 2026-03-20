#!/usr/bin/env python3
"""
Lightweight direct accuracy probe for Arch2 selective-skip inference.

Purpose
- Check whether the Jetson selective-skip implementation preserves detection quality.
- Compare current selective mode against one or more alternative thresholds and/or
  full-blend mode on the same LR validation subset.
- Report direct TP/FP/FN@0.5 along with gate statistics and SR-applied ratio.

Notes
- This is intentionally lightweight and self-contained. It is not a full
  Ultralytics mAP reproduction.
- Ground-truth labels are expected in HR space (e.g. smart_airbus_data/labels/val)
  while LR inputs are read from the LR image directory.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import math
import platform
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for arch2_accuracy_probe.py") from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Arch2 accuracy on LR validation images.")
    parser.add_argument("--project_root", type=str, default=None, help="Repo root to prepend to sys.path.")
    parser.add_argument("--arch2_config", type=str, required=True, help="Path to arch2 YAML config.")
    parser.add_argument(
        "--arch2_py",
        type=str,
        default=None,
        help="Optional explicit path to arch2_softgate.py (patched selective-skip runtime).",
    )
    parser.add_argument("--lr_images_dir", type=str, required=True, help="Directory of LR validation images.")
    parser.add_argument(
        "--hr_labels_dir",
        type=str,
        required=True,
        help="Directory of HR YOLO labels corresponding to the LR validation images.",
    )
    parser.add_argument("--max_images", type=int, default=500, help="Maximum images to process.")
    parser.add_argument("--device", type=str, default="cuda", help="Device string, e.g. cuda or cpu.")
    parser.add_argument("--half", action="store_true", help="Use torch autocast(fp16) on CUDA.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="Detector IoU threshold.")
    parser.add_argument(
        "--modes",
        type=str,
        default="full_blend,thr=0.5",
        help=(
            "Comma-separated comparison modes. Supported tokens: "
            "full_blend, selective, thr=<float>. Example: full_blend,thr=0.3,thr=0.5,thr=0.7"
        ),
    )
    parser.add_argument(
        "--sr_weights",
        type=str,
        default=None,
        help="Optional override for model.rfdn.pretrain_path or model.mamba.pretrain_path.",
    )
    parser.add_argument("--gate_weights", type=str, default=None, help="Optional override for gate weights path.")
    parser.add_argument("--yolo_weights", type=str, default=None, help="Optional override for SR-domain YOLO weights path.")
    parser.add_argument(
        "--blend_selected",
        type=int,
        default=None,
        choices=[0, 1],
        help="Optional override for blend_selected_inference (0 or 1).",
    )
    parser.add_argument(
        "--save_examples",
        type=int,
        default=0,
        help="Save reduced debug JSON for the first N images of each mode when --debug_dir is set.",
    )
    parser.add_argument("--debug_dir", type=str, default=None, help="Optional directory for example JSON files.")
    parser.add_argument("--out_json", type=str, required=True, help="Where to save the summary JSON.")
    parser.add_argument("--print_every", type=int, default=50, help="Progress logging frequency.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config / import helpers
# ---------------------------------------------------------------------------


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}



def ensure_project_root(args: argparse.Namespace) -> Optional[Path]:
    candidates: List[Path] = []
    if args.project_root:
        candidates.append(Path(args.project_root).resolve())

    if args.arch2_py:
        p = Path(args.arch2_py).resolve()
        parts = p.parts
        if "src" in parts:
            idx = parts.index("src")
            candidates.append(Path(*parts[:idx]).resolve())

    cfg_path = Path(args.arch2_config).resolve()
    parts = cfg_path.parts
    if "configs" in parts:
        idx = parts.index("configs")
        candidates.append(Path(*parts[:idx]).resolve())

    for cand in candidates:
        if cand.exists():
            sys.path.insert(0, str(cand))
            return cand
    return None



def load_arch2_class(args: argparse.Namespace):
    if args.arch2_py:
        module_path = Path(args.arch2_py).resolve()
        spec = importlib.util.spec_from_file_location("arch2_softgate_runtime", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module spec from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module("src.models.pipelines.arch2_softgate")
    if not hasattr(module, "Arch2SoftGate"):
        raise AttributeError("Arch2 module does not expose Arch2SoftGate")
    return module.Arch2SoftGate



def patch_config_dict(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))
    cfg["device"] = args.device

    data_cfg = cfg.setdefault("data", {})
    data_cfg.setdefault("upscale_factor", 4)

    model = cfg.setdefault("model", {})
    yolo_cfg = model.setdefault("yolo", {})
    gate_cfg = model.setdefault("gate", {})
    sr_type = str(model.get("sr_type", "rfdn")).lower()
    sr_cfg = model.setdefault("mamba" if sr_type == "mamba" else "rfdn", {})

    if args.yolo_weights:
        yolo_cfg["weights_path"] = args.yolo_weights
    if args.gate_weights:
        gate_cfg["weights_path"] = args.gate_weights
    if args.sr_weights:
        sr_cfg["pretrain_path"] = args.sr_weights
    if args.blend_selected is not None:
        gate_cfg["blend_selected_inference"] = bool(args.blend_selected)

    return cfg


# ---------------------------------------------------------------------------
# Dataset / IO helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


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
        return {
            "tp50": 0,
            "fp50": 0,
            "fn50": int(gt_boxes.shape[0]),
            "num_preds": 0,
            "num_gts": int(gt_boxes.shape[0]),
        }
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


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


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



def parse_modes(raw: str, current_threshold: Optional[float]) -> List[Dict[str, Any]]:
    modes: List[Dict[str, Any]] = []
    for token in [t.strip() for t in raw.split(",") if t.strip()]:
        low = token.lower()
        if low == "full_blend":
            modes.append({"name": "full_blend", "kind": "full_blend", "threshold": None})
        elif low in {"selective", "current"}:
            thr = float(current_threshold) if current_threshold is not None else 0.5
            modes.append({"name": f"thr={thr:.3f}", "kind": "selective", "threshold": thr})
        elif low.startswith("thr="):
            thr = float(low.split("=", 1)[1])
            modes.append({"name": f"thr={thr:.3f}", "kind": "selective", "threshold": thr})
        else:
            try:
                thr = float(low)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported mode token {token!r}. Use full_blend, selective, or thr=<float>."
                ) from exc
            modes.append({"name": f"thr={thr:.3f}", "kind": "selective", "threshold": thr})
    if not modes:
        raise ValueError("At least one mode must be provided via --modes")
    return modes



def apply_arch2_mode(model: Any, mode: Dict[str, Any]) -> Dict[str, Any]:
    prev = {
        "use_selective_inference": getattr(model, "use_selective_inference", None),
        "inference_gate_threshold": getattr(model, "inference_gate_threshold", None),
    }

    if mode["kind"] == "full_blend":
        if hasattr(model, "use_selective_inference"):
            model.use_selective_inference = False
    else:
        if hasattr(model, "use_selective_inference"):
            model.use_selective_inference = True
        if mode["threshold"] is not None and hasattr(model, "inference_gate_threshold"):
            model.inference_gate_threshold = float(mode["threshold"])
    return prev



def restore_arch2_mode(model: Any, prev: Dict[str, Any]) -> None:
    for k, v in prev.items():
        if v is not None and hasattr(model, k):
            setattr(model, k, v)



def predict_arch2(
    model: Any,
    lr_tensor: torch.Tensor,
    conf: float,
    iou: float,
    use_half: bool,
    device: torch.device,
) -> Dict[str, Any]:
    sig = inspect.signature(model.forward)
    kwargs: Dict[str, Any] = {}
    if "return_intermediates" in sig.parameters:
        kwargs["return_intermediates"] = False
    if "det_conf" in sig.parameters:
        kwargs["det_conf"] = conf
    if "det_iou" in sig.parameters:
        kwargs["det_iou"] = iou

    lr_tensor = lr_tensor.to(device, non_blocking=True)
    model.eval()
    sync_if_needed(device)
    t0 = time.perf_counter()
    with AutocastContext(use_half, device):
        output = model.forward(lr_tensor, **kwargs)
    sync_if_needed(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    output["elapsed_ms"] = elapsed_ms
    return output



def detections_to_cpu(det: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes = det["boxes"].detach().cpu().float()
    scores = det["scores"].detach().cpu().float()
    classes = det["classes"].detach().cpu().long()
    return boxes, scores, classes



def maybe_save_example(
    debug_dir: Optional[Path],
    key: str,
    mode_name: str,
    save_examples: int,
    example_idx: int,
    output: Dict[str, Any],
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
) -> None:
    if debug_dir is None or example_idx >= save_examples:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / f"{example_idx:03d}_{mode_name}_{key.replace('/', '__')}.json"
    det = output["detections"][0]
    gate = output.get("gate")
    payload = {
        "key": key,
        "mode": mode_name,
        "elapsed_ms": float(output.get("elapsed_ms", math.nan)),
        "gate_mean": float(gate.detach().float().mean().item()) if gate is not None else None,
        "sr_selected_ratio": float(output["sr_selected_mask"].float().mean().item()) if "sr_selected_mask" in output else None,
        "detections": {
            "boxes": det["boxes"].detach().cpu().tolist(),
            "scores": det["scores"].detach().cpu().tolist(),
            "classes": det["classes"].detach().cpu().tolist(),
        },
        "gt": {
            "boxes": gt_boxes.tolist(),
            "classes": gt_classes.tolist(),
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    project_root = ensure_project_root(args)

    config_dict = patch_config_dict(read_yaml(Path(args.arch2_config)), args)
    Arch2Class = load_arch2_class(args)
    model = Arch2Class(config_dict)
    model.eval()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = model.to(device)

    upscale_factor = int(config_dict.get("data", {}).get("upscale_factor", getattr(model, "upscale_factor", 4)))
    modes = parse_modes(args.modes, getattr(model, "inference_gate_threshold", None))
    pairs = pair_dataset(
        lr_images_dir=Path(args.lr_images_dir),
        hr_labels_dir=Path(args.hr_labels_dir),
        max_images=args.max_images,
    )

    results: Dict[str, Dict[str, Any]] = {
        mode["name"]: {
            "num_images": 0,
            "tp50": 0,
            "fp50": 0,
            "fn50": 0,
            "num_preds": 0,
            "num_gts": 0,
            "elapsed_ms_total": 0.0,
            "gate_values": [],
            "gate_means": [],
            "sr_ratios": [],
            "effective_threshold": None,
            "selective_enabled": None,
            "blend_selected": None,
        }
        for mode in modes
    }

    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    print(f"[arch2_probe] project_root : {project_root}")
    print(f"[arch2_probe] config       : {Path(args.arch2_config).resolve()}")
    print(f"[arch2_probe] arch2_py     : {Path(args.arch2_py).resolve() if args.arch2_py else 'src.models.pipelines.arch2_softgate'}")
    print(f"[arch2_probe] images       : {len(pairs)}")
    print(f"[arch2_probe] device       : {device}")
    print(f"[arch2_probe] modes        : {', '.join(mode['name'] for mode in modes)}")

    for idx, (key, lr_path, label_path) in enumerate(pairs, start=1):
        lr_tensor = load_image_tensor(lr_path)
        _, _, lr_h, lr_w = lr_tensor.shape
        gt_boxes, gt_classes = load_yolo_labels(label_path, img_w=lr_w * upscale_factor, img_h=lr_h * upscale_factor)

        for mode in modes:
            prev = apply_arch2_mode(model, mode)
            try:
                output = predict_arch2(
                    model=model,
                    lr_tensor=lr_tensor,
                    conf=args.conf,
                    iou=args.iou,
                    use_half=args.half,
                    device=device,
                )
            finally:
                restore_arch2_mode(model, prev)

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

            gate = output.get("gate")
            gate_vals = gate.detach().float().cpu().view(-1) if gate is not None else torch.empty((0,), dtype=torch.float32)
            if "sr_selected_mask" in output:
                sr_ratio = float(output["sr_selected_mask"].detach().float().mean().item())
            elif mode["kind"] == "full_blend":
                sr_ratio = 1.0
            else:
                thr = float(mode["threshold"] if mode["threshold"] is not None else getattr(model, "inference_gate_threshold", 0.5))
                sr_ratio = float((gate_vals > thr).float().mean().item()) if gate_vals.numel() > 0 else 0.0

            acc = results[mode["name"]]
            acc["num_images"] += 1
            acc["tp50"] += matched["tp50"]
            acc["fp50"] += matched["fp50"]
            acc["fn50"] += matched["fn50"]
            acc["num_preds"] += matched["num_preds"]
            acc["num_gts"] += matched["num_gts"]
            acc["elapsed_ms_total"] += float(output["elapsed_ms"])
            acc["gate_values"].extend(gate_vals.tolist())
            if gate_vals.numel() > 0:
                acc["gate_means"].append(float(gate_vals.mean().item()))
            acc["sr_ratios"].append(sr_ratio)
            acc["effective_threshold"] = float(mode["threshold"]) if mode["threshold"] is not None else None
            acc["selective_enabled"] = bool(mode["kind"] == "selective")
            acc["blend_selected"] = bool(getattr(model, "blend_selected_inference", False))

            maybe_save_example(
                debug_dir=debug_dir,
                key=key,
                mode_name=mode["name"],
                save_examples=args.save_examples,
                example_idx=idx - 1,
                output=output,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
            )

        if args.print_every > 0 and (idx % args.print_every == 0 or idx == len(pairs)):
            print(f"[arch2_probe] processed {idx}/{len(pairs)} images")

    summary_modes: Dict[str, Dict[str, Any]] = {}
    for mode_name, acc in results.items():
        tp = acc["tp50"]
        fp = acc["fp50"]
        fn = acc["fn50"]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        avg_ms = acc["elapsed_ms_total"] / max(1, acc["num_images"])
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        gate_arr = np.asarray(acc["gate_values"], dtype=np.float32) if acc["gate_values"] else np.asarray([], dtype=np.float32)
        sr_arr = np.asarray(acc["sr_ratios"], dtype=np.float32) if acc["sr_ratios"] else np.asarray([], dtype=np.float32)
        threshold = acc["effective_threshold"]
        ratio_gt_threshold = (
            float((gate_arr > threshold).mean()) if gate_arr.size and threshold is not None else None
        )

        summary_modes[mode_name] = {
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
            "selective_enabled": acc["selective_enabled"],
            "effective_gate_threshold": threshold,
            "blend_selected_inference": acc["blend_selected"],
            "gate_mean": float(gate_arr.mean()) if gate_arr.size else None,
            "gate_std": float(gate_arr.std()) if gate_arr.size else None,
            "gate_min": float(gate_arr.min()) if gate_arr.size else None,
            "gate_max": float(gate_arr.max()) if gate_arr.size else None,
            "ratio_gt_threshold": ratio_gt_threshold,
            "sr_applied_ratio_mean": float(sr_arr.mean()) if sr_arr.size else None,
            "sr_applied_ratio_std": float(sr_arr.std()) if sr_arr.size else None,
        }

    out = {
        "meta": {
            "arch2_config": str(Path(args.arch2_config).resolve()),
            "arch2_py": str(Path(args.arch2_py).resolve()) if args.arch2_py else "src.models.pipelines.arch2_softgate",
            "lr_images_dir": str(Path(args.lr_images_dir).resolve()),
            "hr_labels_dir": str(Path(args.hr_labels_dir).resolve()),
            "num_images": len(pairs),
            "device": str(device),
            "half_amp": bool(args.half and device.type == "cuda" and torch.cuda.is_available()),
            "upscale_factor": upscale_factor,
            "conf": args.conf,
            "iou": args.iou,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "results": summary_modes,
    }

    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("[arch2_probe] saved ->", out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
