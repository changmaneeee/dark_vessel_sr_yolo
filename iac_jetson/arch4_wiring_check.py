#!/usr/bin/env python3
"""
Stage-by-stage wiring check for Arch4 ROI-aware pipeline.

Purpose
- Verify whether Scout -> ROI -> refine -> Sniper -> merge stages are all active.
- Compare multiple crop refinement modes on the same paired LR/HR subset.
- Detect whether the runtime actually supports crop_refine_mode / hr_images / stats.
- Report both direct TP/FP/FN@0.5 and intermediate-stage signatures.

Typical use
- For canonical repo runtime (to see what is actually happening now):
    --arch4_py /path/to/repo/src/models/pipelines/arch4_roi_awareNMS.py
- For patched ablation runtime (to verify mode wiring):
    --arch4_py /path/to/repo/src/models/pipelines/arch4_roi_awareNMS_ablation.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import platform
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for arch4_wiring_check.py") from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-by-stage wiring check for Arch4 ROI-aware pipeline.")
    parser.add_argument("--project_root", type=str, default=None, help="Repo root to prepend to sys.path.")
    parser.add_argument("--arch4_config", type=str, required=True, help="Path to arch4 YAML config.")
    parser.add_argument(
        "--arch4_py",
        type=str,
        default=None,
        help="Optional explicit path to the arch4 runtime python file.",
    )
    parser.add_argument("--lr_images_dir", type=str, required=True, help="Directory of LR validation images.")
    parser.add_argument("--hr_images_dir", type=str, default=None, help="Optional HR validation images dir (needed for hr_ref).")
    parser.add_argument("--hr_labels_dir", type=str, required=True, help="Directory of HR YOLO labels.")
    parser.add_argument("--max_images", type=int, default=100, help="Maximum paired images to inspect.")
    parser.add_argument("--device", type=str, default="cuda", help="Device string, e.g. cuda or cpu.")
    parser.add_argument("--half", action="store_true", help="Use torch autocast(fp16) on CUDA.")
    parser.add_argument(
        "--modes",
        type=str,
        default="sr,bilinear,hr_ref",
        help="Comma-separated crop refinement modes to compare.",
    )
    parser.add_argument(
        "--sniper_imgsz_mode",
        type=str,
        default=None,
        help="Optional override: dynamic or fixed.",
    )
    parser.add_argument(
        "--sniper_imgsz_fixed",
        type=int,
        default=None,
        help="Optional fixed sniper imgsz override.",
    )
    parser.add_argument("--sr_weights", type=str, default=None, help="Optional override for model.sr.weights.")
    parser.add_argument("--yolo_weights_lr", type=str, default=None, help="Optional override for model.yolo.weights_lr.")
    parser.add_argument("--yolo_weights_hr", type=str, default=None, help="Optional override for model.yolo.weights_hr.")
    parser.add_argument(
        "--eval_space",
        type=str,
        default="hr",
        choices=["hr", "lr"],
        help="Compare predictions in HR or LR coordinate space.",
    )
    parser.add_argument(
        "--save_examples",
        type=int,
        default=10,
        help="Keep reduced per-image records for the first N images in the output JSON.",
    )
    parser.add_argument("--out_json", type=str, required=True, help="Where to save the summary JSON.")
    parser.add_argument("--print_every", type=int, default=25, help="Progress logging frequency.")
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

    if args.arch4_py:
        p = Path(args.arch4_py).resolve()
        parts = p.parts
        if "src" in parts:
            idx = parts.index("src")
            candidates.append(Path(*parts[:idx]).resolve())

    cfg_path = Path(args.arch4_config).resolve()
    parts = cfg_path.parts
    if "configs" in parts:
        idx = parts.index("configs")
        candidates.append(Path(*parts[:idx]).resolve())

    for cand in candidates:
        if cand.exists():
            sys.path.insert(0, str(cand))
            return cand
    return None



def load_arch4_class(args: argparse.Namespace):
    if args.arch4_py:
        module_path = Path(args.arch4_py).resolve()
        spec = importlib.util.spec_from_file_location("arch4_roi_runtime", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module spec from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module("src.models.pipelines.arch4_roi_awareNMS")
    if not hasattr(module, "Arch4RoiAwareNMS"):
        raise AttributeError("Arch4 module does not expose Arch4RoiAwareNMS")
    return module.Arch4RoiAwareNMS



def patch_config_dict(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))
    cfg["device"] = args.device

    data_cfg = cfg.setdefault("data", {})
    data_cfg.setdefault("upscale_factor", 4)

    model = cfg.setdefault("model", {})
    sr_cfg = model.setdefault("sr", {})
    yolo_cfg = model.setdefault("yolo", {})
    arch4_cfg = model.setdefault("arch4", {})

    if args.sr_weights:
        sr_cfg["weights"] = args.sr_weights
    if args.yolo_weights_lr:
        yolo_cfg["weights_lr"] = args.yolo_weights_lr
    if args.yolo_weights_hr:
        yolo_cfg["weights_hr"] = args.yolo_weights_hr
    if args.sniper_imgsz_mode:
        arch4_cfg["sniper_imgsz_mode"] = args.sniper_imgsz_mode
    if args.sniper_imgsz_fixed is not None:
        arch4_cfg["sniper_imgsz_fixed"] = int(args.sniper_imgsz_fixed)

    return cfg


# ---------------------------------------------------------------------------
# Dataset helpers
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



def pair_dataset(
    lr_images_dir: Path,
    hr_images_dir: Optional[Path],
    hr_labels_dir: Path,
    max_images: int,
) -> List[Tuple[str, Path, Optional[Path], Path]]:
    lr_map = build_stem_map(lr_images_dir, IMAGE_EXTS)
    label_map = build_stem_map(hr_labels_dir, {".txt"})
    hr_map = build_stem_map(hr_images_dir, IMAGE_EXTS) if hr_images_dir is not None else {}

    keys = set(lr_map.keys()) & set(label_map.keys())
    if hr_images_dir is not None:
        keys &= set(hr_map.keys())
    keys = sorted(keys)
    if max_images > 0:
        keys = keys[:max_images]
    if not keys:
        raise FileNotFoundError(
            f"No paired samples found across\n  LR: {lr_images_dir}\n  HR: {hr_images_dir}\n  Labels: {hr_labels_dir}"
        )
    return [(key, lr_map[key], hr_map.get(key), label_map[key]) for key in keys]



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
# Metrics helpers
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



def detections_to_eval_space(
    det: Dict[str, torch.Tensor],
    upscale_factor: float,
    eval_space: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes = det["boxes"].detach().cpu().float()
    scores = det["scores"].detach().cpu().float()
    classes = det["classes"].detach().cpu().long()
    if eval_space == "hr" and boxes.numel() > 0:
        boxes = boxes * float(upscale_factor)
    return boxes, scores, classes



def tensor_hash(t: Optional[torch.Tensor]) -> Optional[str]:
    if t is None:
        return None
    arr = t.detach().cpu().float().contiguous().numpy()
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]



def detections_hash(det_list: Sequence[Dict[str, torch.Tensor]]) -> str:
    payload: List[np.ndarray] = []
    for det in det_list:
        boxes = det["boxes"].detach().cpu().float().numpy().round(3)
        scores = det["scores"].detach().cpu().float().numpy().round(4)
        classes = det["classes"].detach().cpu().long().numpy()
        payload.extend([boxes, scores, classes])
    if not payload:
        return "empty"
    packed = b"".join(arr.tobytes() for arr in payload)
    return hashlib.sha1(packed).hexdigest()[:16]



def summarize_debug(output: Dict[str, Any]) -> Dict[str, Any]:
    debug = output.get("debug_info", {}) or {}
    stats = output.get("stats", {}) or {}

    pass1_raw = debug.get("pass1_raw", []) or []
    pass1_after_nms = debug.get("pass1_after_nms", []) or []
    roi_groups = debug.get("roi_groups", []) or []
    crops_lr = debug.get("crops_lr", []) or []
    crops_sr = debug.get("crops_sr", []) or []
    pass2_raw = debug.get("pass2_raw", []) or []
    pass2_after_nms = debug.get("pass2_after_nms", []) or []

    out: Dict[str, Any] = {
        "has_stats": bool(stats),
        "has_debug_info": bool(debug),
        "scout_raw_boxes_total": int(sum(len(det.get("boxes", [])) for det in pass1_raw if isinstance(det, dict))),
        "pass1_after_nms_boxes_total": int(sum(len(det.get("boxes", [])) for det in pass1_after_nms if isinstance(det, dict))),
        "roi_groups_total": int(sum(len(groups) for groups in roi_groups if isinstance(groups, list))),
        "roi_crops_total": int(len(crops_lr)),
        "refined_crops_total": int(len(crops_sr)),
        "pass2_raw_boxes_total": int(sum(len(det.get("boxes", [])) for det in pass2_raw if isinstance(det, dict))),
        "pass2_after_nms_boxes_total": int(sum(len(det.get("boxes", [])) for det in pass2_after_nms if isinstance(det, dict))),
        "first_crop_lr_hash": tensor_hash(crops_lr[0]) if len(crops_lr) > 0 else None,
        "first_crop_refined_hash": tensor_hash(crops_sr[0]) if len(crops_sr) > 0 else None,
        "pass2_hash": detections_hash(pass2_raw) if len(pass2_raw) > 0 else "empty",
    }

    if stats:
        for key, value in stats.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                out[f"stats.{key}"] = value
    return out



def maybe_pick_final_sig(output: Dict[str, Any]) -> str:
    dets = output.get("detections", []) or []
    if len(dets) == 0:
        return "empty"
    return detections_hash(dets)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    project_root = ensure_project_root(args)

    hr_images_dir = Path(args.hr_images_dir) if args.hr_images_dir else None
    config_dict = patch_config_dict(read_yaml(Path(args.arch4_config)), args)
    Arch4Class = load_arch4_class(args)
    model = Arch4Class(config_dict)
    model.eval()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    upscale_factor = float(config_dict.get("data", {}).get("upscale_factor", 4))
    pairs = pair_dataset(
        lr_images_dir=Path(args.lr_images_dir),
        hr_images_dir=hr_images_dir,
        hr_labels_dir=Path(args.hr_labels_dir),
        max_images=args.max_images,
    )
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        raise ValueError("At least one mode must be provided via --modes")

    forward_sig = inspect.signature(model.forward)
    supports_hr_images = "hr_images" in forward_sig.parameters
    supports_debug = "debug" in forward_sig.parameters
    supports_crop_mode = hasattr(getattr(model, "cfg", None), "crop_refine_mode")
    supports_imgsz_mode = hasattr(getattr(model, "cfg", None), "sniper_imgsz_mode")

    if args.sniper_imgsz_mode and supports_imgsz_mode:
        model.cfg.sniper_imgsz_mode = args.sniper_imgsz_mode
    if args.sniper_imgsz_fixed is not None and hasattr(getattr(model, "cfg", None), "sniper_imgsz_fixed"):
        model.cfg.sniper_imgsz_fixed = int(args.sniper_imgsz_fixed)

    print(f"[arch4_wiring] project_root         : {project_root}")
    print(f"[arch4_wiring] config               : {Path(args.arch4_config).resolve()}")
    print(f"[arch4_wiring] arch4_py             : {Path(args.arch4_py).resolve() if args.arch4_py else 'src.models.pipelines.arch4_roi_awareNMS'}")
    print(f"[arch4_wiring] images               : {len(pairs)}")
    print(f"[arch4_wiring] device               : {device}")
    print(f"[arch4_wiring] modes                : {', '.join(modes)}")
    print(f"[arch4_wiring] supports_hr_images   : {supports_hr_images}")
    print(f"[arch4_wiring] supports_debug       : {supports_debug}")
    print(f"[arch4_wiring] supports_crop_mode   : {supports_crop_mode}")
    print(f"[arch4_wiring] supports_imgsz_mode  : {supports_imgsz_mode}")

    results: Dict[str, Dict[str, Any]] = {
        mode: {
            "num_images": 0,
            "tp50": 0,
            "fp50": 0,
            "fn50": 0,
            "num_preds": 0,
            "num_gts": 0,
            "elapsed_ms_total": 0.0,
            "stage_sum": {},
            "errors": [],
        }
        for mode in modes
    }
    per_image_kept: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in modes}
    mode_signatures: Dict[str, Dict[str, Dict[str, Any]]] = {mode: {} for mode in modes}

    for idx, (key, lr_path, hr_path, label_path) in enumerate(pairs, start=1):
        lr_tensor = load_image_tensor(lr_path)
        hr_tensor = load_image_tensor(hr_path) if hr_path is not None else None
        _, _, hr_h, hr_w = (hr_tensor.shape if hr_tensor is not None else (1, 3, int(lr_tensor.shape[-2] * upscale_factor), int(lr_tensor.shape[-1] * upscale_factor)))
        gt_boxes_hr, gt_classes = load_yolo_labels(label_path, img_w=hr_w, img_h=hr_h)
        gt_boxes = gt_boxes_hr / upscale_factor if args.eval_space == "lr" and gt_boxes_hr.numel() > 0 else gt_boxes_hr

        lr_tensor = lr_tensor.to(device, non_blocking=True)
        if hr_tensor is not None:
            hr_tensor = hr_tensor.to(device, non_blocking=True)

        for mode in modes:
            acc = results[mode]
            requested_hr = str(mode).lower() in {"hr_ref", "hr", "oracle", "gt"}

            if supports_crop_mode:
                model.cfg.crop_refine_mode = mode

            kwargs: Dict[str, Any] = {}
            if supports_debug:
                kwargs["debug"] = True
            if requested_hr and supports_hr_images:
                kwargs["hr_images"] = hr_tensor
            elif requested_hr and not supports_hr_images:
                acc["errors"].append(f"{key}: hr_ref requested but forward() has no hr_images parameter")
                continue

            sync_if_needed(device)
            t0 = time.perf_counter()
            try:
                with AutocastContext(args.half, device):
                    output = model.forward(lr_tensor, **kwargs)
            except Exception as exc:
                acc["errors"].append(f"{key}: {type(exc).__name__}: {exc}")
                continue
            sync_if_needed(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            det_boxes, det_scores, det_classes = detections_to_eval_space(
                output["detections"][0],
                upscale_factor=upscale_factor,
                eval_space=args.eval_space,
            )
            matched = match_predictions(
                pred_boxes=det_boxes,
                pred_scores=det_scores,
                pred_classes=det_classes,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                iou_thresh=0.5,
            )

            stage = summarize_debug(output)
            stage["final_detections_total"] = int(len(output["detections"][0]["boxes"]))
            stage["final_detections_hash"] = maybe_pick_final_sig(output)
            stage["elapsed_ms"] = float(elapsed_ms)

            acc["num_images"] += 1
            acc["tp50"] += matched["tp50"]
            acc["fp50"] += matched["fp50"]
            acc["fn50"] += matched["fn50"]
            acc["num_preds"] += matched["num_preds"]
            acc["num_gts"] += matched["num_gts"]
            acc["elapsed_ms_total"] += elapsed_ms
            for k, v in stage.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    acc["stage_sum"][k] = acc["stage_sum"].get(k, 0.0) + float(v)

            mode_signatures[mode][key] = {
                "final_hash": stage["final_detections_hash"],
                "pass2_hash": stage.get("pass2_hash"),
                "first_crop_refined_hash": stage.get("first_crop_refined_hash"),
                "first_crop_lr_hash": stage.get("first_crop_lr_hash"),
                "roi_crops_total": stage.get("roi_crops_total", 0),
            }

            if idx <= args.save_examples:
                reduced = {
                    "key": key,
                    "mode": mode,
                    "tp50": matched["tp50"],
                    "fp50": matched["fp50"],
                    "fn50": matched["fn50"],
                    "stage": stage,
                    "gt_boxes_count": int(gt_boxes.shape[0]),
                }
                per_image_kept[mode].append(reduced)

        if args.print_every > 0 and (idx % args.print_every == 0 or idx == len(pairs)):
            print(f"[arch4_wiring] processed {idx}/{len(pairs)} images")

    summary_modes: Dict[str, Dict[str, Any]] = {}
    for mode, acc in results.items():
        tp = acc["tp50"]
        fp = acc["fp50"]
        fn = acc["fn50"]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        avg_ms = acc["elapsed_ms_total"] / max(1, acc["num_images"])
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        avg_stage = {
            k: (v / max(1, acc["num_images"]))
            for k, v in acc["stage_sum"].items()
        }
        summary_modes[mode] = {
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
            "avg_stage": avg_stage,
            "num_errors": len(acc["errors"]),
            "errors_head": acc["errors"][:10],
        }

    reference_mode = modes[0]
    cross_mode: Dict[str, Any] = {"reference_mode": reference_mode, "comparisons": {}}
    ref_map = mode_signatures.get(reference_mode, {})
    for mode in modes[1:]:
        cmp_map = mode_signatures.get(mode, {})
        shared_keys = sorted(set(ref_map.keys()) & set(cmp_map.keys()))
        if not shared_keys:
            cross_mode["comparisons"][mode] = {
                "shared_images": 0,
                "final_hash_same": 0,
                "pass2_hash_same": 0,
                "refined_crop_hash_same": 0,
                "images_with_any_roi": 0,
            }
            continue
        final_same = 0
        pass2_same = 0
        refined_same = 0
        any_roi = 0
        for key in shared_keys:
            ref_sig = ref_map[key]
            cmp_sig = cmp_map[key]
            if ref_sig.get("roi_crops_total", 0) > 0 or cmp_sig.get("roi_crops_total", 0) > 0:
                any_roi += 1
            if ref_sig.get("final_hash") == cmp_sig.get("final_hash"):
                final_same += 1
            if ref_sig.get("pass2_hash") == cmp_sig.get("pass2_hash"):
                pass2_same += 1
            if ref_sig.get("first_crop_refined_hash") == cmp_sig.get("first_crop_refined_hash"):
                refined_same += 1
        cross_mode["comparisons"][mode] = {
            "shared_images": len(shared_keys),
            "images_with_any_roi": any_roi,
            "final_hash_same": final_same,
            "pass2_hash_same": pass2_same,
            "refined_crop_hash_same": refined_same,
            "final_hash_same_ratio": final_same / max(1, len(shared_keys)),
            "pass2_hash_same_ratio": pass2_same / max(1, len(shared_keys)),
            "refined_crop_hash_same_ratio": refined_same / max(1, len(shared_keys)),
        }

    out = {
        "meta": {
            "arch4_config": str(Path(args.arch4_config).resolve()),
            "arch4_py": str(Path(args.arch4_py).resolve()) if args.arch4_py else "src.models.pipelines.arch4_roi_awareNMS",
            "lr_images_dir": str(Path(args.lr_images_dir).resolve()),
            "hr_images_dir": str(hr_images_dir.resolve()) if hr_images_dir is not None else None,
            "hr_labels_dir": str(Path(args.hr_labels_dir).resolve()),
            "num_images": len(pairs),
            "device": str(device),
            "half_amp": bool(args.half and device.type == "cuda" and torch.cuda.is_available()),
            "eval_space": args.eval_space,
            "upscale_factor": upscale_factor,
            "supports_hr_images": supports_hr_images,
            "supports_debug": supports_debug,
            "supports_crop_mode": supports_crop_mode,
            "supports_imgsz_mode": supports_imgsz_mode,
            "sniper_imgsz_mode": getattr(getattr(model, "cfg", None), "sniper_imgsz_mode", None),
            "sniper_imgsz_fixed": getattr(getattr(model, "cfg", None), "sniper_imgsz_fixed", None),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "results": summary_modes,
        "cross_mode": cross_mode,
        "per_image_examples": per_image_kept,
    }

    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("[arch4_wiring] saved ->", out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
