#!/usr/bin/env python3
"""
Direct probe for Arch4 ROI crop refinement modes.

Purpose
- Run the same Arch4 ROI-aware pipeline with multiple crop refinement modes
  on the same paired LR/HR validation images.
- Produce quick direct TP/FP/FN@0.5 comparisons for:
    * sr       : original SR model (RFDN/Mamba)
    * bilinear : interpolation only
    * hr_ref   : paired HR crop oracle reference
- Summarize ROI burden (uncertain boxes, merged ROI groups, crop count).

This is intentionally lightweight and self-contained. It does not try to
reproduce full Ultralytics mAP parity; it is for diagnosis / relative ranking.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
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
except Exception as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("PyYAML is required for arch4_crop_mode_probe.py") from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Arch4 ROI crop modes on paired LR/HR images.")
    parser.add_argument("--project_root", type=str, default=None, help="Repo root to prepend to sys.path.")
    parser.add_argument("--arch4_config", type=str, required=True, help="Path to arch4 YAML config.")
    parser.add_argument(
        "--arch4_py",
        type=str,
        default=None,
        help="Optional explicit path to the patched arch4_roi_awareNMS.py file.",
    )
    parser.add_argument("--lr_images_dir", type=str, required=True, help="Directory of LR validation images.")
    parser.add_argument("--hr_images_dir", type=str, required=True, help="Directory of HR validation images.")
    parser.add_argument("--hr_labels_dir", type=str, required=True, help="Directory of HR YOLO labels.")
    parser.add_argument("--max_images", type=int, default=200, help="Maximum paired images to process.")
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
    parser.add_argument(
        "--sr_weights",
        type=str,
        default=None,
        help="Optional override for model.sr.weights in the YAML config.",
    )
    parser.add_argument(
        "--yolo_weights_lr",
        type=str,
        default=None,
        help="Optional override for model.yolo.weights_lr in the YAML config.",
    )
    parser.add_argument(
        "--yolo_weights_hr",
        type=str,
        default=None,
        help="Optional override for model.yolo.weights_hr in the YAML config.",
    )
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
        default=0,
        help="Save debug JSON for the first N images of each mode when --debug_dir is set.",
    )
    parser.add_argument(
        "--debug_dir",
        type=str,
        default=None,
        help="Optional directory for debug JSON outputs.",
    )
    parser.add_argument("--out_json", type=str, required=True, help="Where to save the probe summary JSON.")
    parser.add_argument(
        "--print_every",
        type=int,
        default=25,
        help="Progress logging frequency.",
    )
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
# Dataset pairing / loading helpers
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



def pair_dataset(lr_images_dir: Path, hr_images_dir: Path, hr_labels_dir: Path, max_images: int) -> List[Tuple[str, Path, Path, Path]]:
    lr_map = build_stem_map(lr_images_dir, IMAGE_EXTS)
    hr_map = build_stem_map(hr_images_dir, IMAGE_EXTS)
    label_map = build_stem_map(hr_labels_dir, {".txt"})

    keys = sorted(set(label_map.keys()) & set(lr_map.keys()) & set(hr_map.keys()))
    if max_images > 0:
        keys = keys[:max_images]
    if not keys:
        raise FileNotFoundError(
            f"No paired samples found across\n"
            f"  LR: {lr_images_dir}\n"
            f"  HR: {hr_images_dir}\n"
            f"  Labels: {hr_labels_dir}"
        )
    return [(key, lr_map[key], hr_map[key], label_map[key]) for key in keys]



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
            'tp50': 0,
            'fp50': 0,
            'fn50': int(gt_boxes.shape[0]),
            'num_preds': 0,
            'num_gts': int(gt_boxes.shape[0]),
        }
    if gt_boxes.numel() == 0:
        return {
            'tp50': 0,
            'fp50': int(pred_boxes.shape[0]),
            'fn50': 0,
            'num_preds': int(pred_boxes.shape[0]),
            'num_gts': 0,
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
        'tp50': int(tp),
        'fp50': int(fp),
        'fn50': fn,
        'num_preds': int(pred_boxes.shape[0]),
        'num_gts': int(gt_boxes.shape[0]),
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
    boxes = det['boxes'].detach().cpu().float()
    scores = det['scores'].detach().cpu().float()
    classes = det['classes'].detach().cpu().long()
    if eval_space == 'hr' and boxes.numel() > 0:
        boxes = boxes * float(upscale_factor)
    return boxes, scores, classes



def maybe_save_example(
    debug_dir: Optional[Path],
    key: str,
    mode: str,
    save_examples: int,
    example_idx: int,
    output: Dict[str, Any],
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
) -> None:
    if debug_dir is None or example_idx >= save_examples:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / f"{example_idx:03d}_{mode}_{key.replace('/', '__')}.json"
    det = output['detections'][0]
    payload = {
        'key': key,
        'mode': mode,
        'stats': output.get('stats', {}),
        'detections': {
            'boxes': det['boxes'].detach().cpu().tolist(),
            'scores': det['scores'].detach().cpu().tolist(),
            'classes': det['classes'].detach().cpu().tolist(),
        },
        'gt': {
            'boxes': gt_boxes.tolist(),
            'classes': gt_classes.tolist(),
        },
    }
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    ensure_project_root(args)

    config_dict = patch_config_dict(read_yaml(Path(args.arch4_config)), args)
    Arch4Class = load_arch4_class(args)
    model = Arch4Class(config_dict)
    model.eval()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    upscale_factor = float(config_dict.get('data', {}).get('upscale_factor', 4))

    pairs = pair_dataset(
        lr_images_dir=Path(args.lr_images_dir),
        hr_images_dir=Path(args.hr_images_dir),
        hr_labels_dir=Path(args.hr_labels_dir),
        max_images=args.max_images,
    )

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    if not modes:
        raise ValueError("At least one mode must be provided via --modes")

    results: Dict[str, Dict[str, Any]] = {
        mode: {
            'num_images': 0,
            'tp50': 0,
            'fp50': 0,
            'fn50': 0,
            'num_preds': 0,
            'num_gts': 0,
            'elapsed_ms_total': 0.0,
            'stats_sum': {},
        }
        for mode in modes
    }

    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    for idx, (key, lr_path, hr_path, label_path) in enumerate(pairs, start=1):
        lr_tensor = load_image_tensor(lr_path)
        hr_tensor = load_image_tensor(hr_path)
        _, _, hr_h, hr_w = hr_tensor.shape
        gt_boxes_hr, gt_classes = load_yolo_labels(label_path, img_w=hr_w, img_h=hr_h)

        if args.eval_space == 'lr' and gt_boxes_hr.numel() > 0:
            gt_boxes = gt_boxes_hr / upscale_factor
        else:
            gt_boxes = gt_boxes_hr

        lr_tensor = lr_tensor.to(device, non_blocking=True)
        hr_tensor = hr_tensor.to(device, non_blocking=True)

        for mode in modes:
            model.cfg.crop_refine_mode = mode

            sync_if_needed(device)
            t0 = time.perf_counter()
            with AutocastContext(args.half, device):
                output = model.forward(
                    lr_tensor,
                    debug=False,
                )
            sync_if_needed(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            det_boxes, det_scores, det_classes = detections_to_eval_space(
                output['detections'][0],
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

            acc = results[mode]
            acc['num_images'] += 1
            acc['tp50'] += matched['tp50']
            acc['fp50'] += matched['fp50']
            acc['fn50'] += matched['fn50']
            acc['num_preds'] += matched['num_preds']
            acc['num_gts'] += matched['num_gts']
            acc['elapsed_ms_total'] += elapsed_ms

            stats = output.get('stats', {})
            stats_sum = acc['stats_sum']
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    stats_sum[k] = stats_sum.get(k, 0.0) + float(v)

            maybe_save_example(
                debug_dir=debug_dir,
                key=key,
                mode=mode,
                save_examples=args.save_examples,
                example_idx=idx - 1,
                output=output,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
            )

        if args.print_every > 0 and (idx % args.print_every == 0 or idx == len(pairs)):
            print(f"[arch4_probe] processed {idx}/{len(pairs)} paired images")

    summary_modes: Dict[str, Dict[str, Any]] = {}
    for mode, acc in results.items():
        tp = acc['tp50']
        fp = acc['fp50']
        fn = acc['fn50']
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        avg_ms = acc['elapsed_ms_total'] / max(1, acc['num_images'])
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        stats_avg = {
            k: (v / max(1, acc['num_images']))
            for k, v in acc['stats_sum'].items()
        }
        summary_modes[mode] = {
            'num_images': acc['num_images'],
            'tp50': tp,
            'fp50': fp,
            'fn50': fn,
            'num_preds': acc['num_preds'],
            'num_gts': acc['num_gts'],
            'precision50_direct': precision,
            'recall50_direct': recall,
            'f1_50_direct': f1,
            'avg_ms_per_image': avg_ms,
            'fps': fps,
            'avg_stats': stats_avg,
        }

    out = {
        'meta': {
            'arch4_config': str(Path(args.arch4_config).resolve()),
            'arch4_py': str(Path(args.arch4_py).resolve()) if args.arch4_py else 'src.models.pipelines.arch4_roi_awareNMS',
            'lr_images_dir': str(Path(args.lr_images_dir).resolve()),
            'hr_images_dir': str(Path(args.hr_images_dir).resolve()),
            'hr_labels_dir': str(Path(args.hr_labels_dir).resolve()),
            'num_images': len(pairs),
            'device': str(device),
            'eval_space': args.eval_space,
            'upscale_factor': upscale_factor,
            'modes': modes,
            'sniper_imgsz_mode_override': args.sniper_imgsz_mode,
            'sniper_imgsz_fixed_override': args.sniper_imgsz_fixed,
        },
        'results': summary_modes,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(f"[arch4_probe] wrote summary -> {out_path}")


if __name__ == "__main__":
    main()
