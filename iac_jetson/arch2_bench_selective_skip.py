#!/usr/bin/env python3
"""
Standalone benchmark runner for Arch2 selective-skip inference.

Goal
- Run the updated Arch2 pipeline on a directory of LR images.
- Report latency/FPS/gate stats in a JSON shape that the existing
  jetson_job_summary.py can parse.
- Optionally load the pipeline from an external file path (e.g. the
  patched arch2_softgate.py before copying it into the repo).

Typical usage:
  python arch2_bench_selective_skip.py \
    --project_root /path/to/repo \
    --arch2_config /path/to/repo/configs/experiment/arch2_softgate.yaml \
    --arch2_py /path/to/repo/src/models/pipelines/arch2_softgate.py \
    --sr_weights /path/to/rfdn.pt \
    --gate_weights /path/to/gate.pt \
    --yolo_weights /path/to/yolo_sr.pt \
    --images_dir /path/to/lr/images/val \
    --device cuda --half --max_images 200 --warmup 20 \
    --gate_threshold 0.5 \
    --out_json /tmp/arch2_bench.json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

try:
    import yaml
except Exception as exc:  # pragma: no cover - environment dependent
    raise RuntimeError(
        "PyYAML is required for arch2_bench_selective_skip.py. Please install pyyaml."
    ) from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark updated Arch2 selective-skip inference.")
    parser.add_argument("--project_root", type=str, default=None, help="Repo root to prepend to sys.path.")
    parser.add_argument("--arch2_config", type=str, required=True, help="Path to arch2 YAML config.")
    parser.add_argument(
        "--arch2_py",
        type=str,
        default=None,
        help="Optional explicit path to the patched arch2_softgate.py file.",
    )
    parser.add_argument("--sr_weights", type=str, required=True, help="RFDN/Mamba SR weights path.")
    parser.add_argument("--gate_weights", type=str, required=True, help="Gate weights path.")
    parser.add_argument("--yolo_weights", type=str, required=True, help="SR-domain YOLO weights path.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory of LR images for timing.")
    parser.add_argument("--max_images", type=int, default=200, help="Maximum number of images to benchmark.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup images before timing.")
    parser.add_argument("--device", type=str, default="cuda", help="Device string, e.g. cuda or cpu.")
    parser.add_argument("--half", action="store_true", help="Use torch autocast(fp16) on CUDA.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO conf threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold.")
    parser.add_argument(
        "--gate_threshold",
        type=float,
        default=None,
        help="Override inference gate threshold. Uses config value when omitted.",
    )
    parser.add_argument(
        "--disable_selective",
        action="store_true",
        help="Disable selective inference and fall back to full-image soft blending.",
    )
    parser.add_argument(
        "--blend_selected",
        action="store_true",
        help="When selective inference is enabled, blend selected SR outputs with bypass path.",
    )
    parser.add_argument(
        "--no_preload",
        action="store_true",
        help="Read images lazily instead of preloading tensors into memory.",
    )
    parser.add_argument("--out_json", type=str, required=True, help="Where to save benchmark JSON.")
    parser.add_argument(
        "--print_every",
        type=int,
        default=50,
        help="Progress logging frequency during timed run.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers: config, import, image loading
# ---------------------------------------------------------------------------


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def dict_to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_namespace(v) for v in obj]
    return obj


def ensure_project_root(args: argparse.Namespace) -> Path | None:
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
    cfg = json.loads(json.dumps(cfg))  # deep copy using stdlib types only
    cfg["device"] = args.device

    model = cfg.setdefault("model", {})
    yolo = model.setdefault("yolo", {})
    gate = model.setdefault("gate", {})

    # SR config can live in model.rfdn or model.mamba depending on sr_type.
    sr_type = str(model.get("sr_type", "rfdn")).lower()
    if sr_type == "mamba":
        sr_cfg = model.setdefault("mamba", {})
    else:
        sr_cfg = model.setdefault("rfdn", {})

    data_cfg = cfg.setdefault("data", {})
    data_cfg.setdefault("upscale_factor", 4)

    yolo["weights_path"] = args.yolo_weights
    gate["weights_path"] = args.gate_weights
    gate["use_selective_inference"] = not args.disable_selective
    if args.gate_threshold is not None:
        gate["inference_threshold"] = float(args.gate_threshold)
    gate["blend_selected_inference"] = bool(args.blend_selected)

    if sr_type == "mamba":
        sr_cfg["pretrain_path"] = args.sr_weights
    else:
        sr_cfg["pretrain_path"] = args.sr_weights

    return cfg


def list_images(images_dir: Path, max_images: int) -> List[Path]:
    files = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    files = sorted(files)
    if max_images > 0:
        files = files[:max_images]
    if not files:
        raise FileNotFoundError(f"No images found under {images_dir}")
    return files


def load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor.unsqueeze(0)


# ---------------------------------------------------------------------------
# Timing helpers
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


# ---------------------------------------------------------------------------
# Inference runner (manual so we can time stages cleanly)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_arch2_once(
    model: Any,
    lr_image: torch.Tensor,
    device: torch.device,
    use_half: bool,
    conf: float,
    iou: float,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    lr_image = lr_image.to(device, non_blocking=True)

    sync_if_needed(device)
    t0 = time.perf_counter()

    with AutocastContext(enabled=use_half, device=device):
        batch = lr_image.shape[0]
        gate = model.gate_network(lr_image)
        gate_flat = gate.view(batch)
        upsampled = model._run_bypass(lr_image)

        sr_selected_mask = torch.ones(batch, dtype=torch.bool, device=lr_image.device)
        sr_image = None

        if model.use_selective_inference:
            sr_selected_mask = gate_flat > model.inference_gate_threshold
            hr_image = upsampled.clone()

            if sr_selected_mask.any():
                selected_lr = lr_image[sr_selected_mask]
                selected_sr = model._run_sr_model(selected_lr)
                hr_dtype = hr_image.dtype
                if selected_sr.dtype != hr_dtype:
                    selected_sr = selected_sr.to(dtype=hr_dtype)
                if model.blend_selected_inference:
                    selected_gate = gate_flat[sr_selected_mask].view(-1, 1, 1, 1)
                    if selected_gate.dtype != upsampled.dtype:
                        selected_gate = selected_gate.to(dtype=upsampled.dtype)
                    selected_hr = selected_gate * selected_sr + (1.0 - selected_gate) * upsampled[sr_selected_mask]
                else:
                    selected_hr = selected_sr
                if selected_hr.dtype != hr_dtype:
                    selected_hr = selected_hr.to(dtype=hr_dtype)
                hr_image[sr_selected_mask] = selected_hr
                sr_image = selected_sr
        else:
            sr_image = model._run_sr_model(lr_image)
            if sr_image.dtype != upsampled.dtype:
                sr_image = sr_image.to(dtype=upsampled.dtype)
            gate_expanded = gate_flat.view(batch, 1, 1, 1)
            if gate_expanded.dtype != upsampled.dtype:
                gate_expanded = gate_expanded.to(dtype=upsampled.dtype)
            hr_image = gate_expanded * sr_image + (1.0 - gate_expanded) * upsampled

    sync_if_needed(device)
    t1 = time.perf_counter()

    with AutocastContext(enabled=use_half, device=device):
        model.detector.eval()
        detections = model.detector.predict(hr_image, conf=conf, iou=iou)

    sync_if_needed(device)
    t2 = time.perf_counter()

    times = {
        "gate_sr_blend_ms": (t1 - t0) * 1000.0,
        "yolo_ms": (t2 - t1) * 1000.0,
        "total_ms": (t2 - t0) * 1000.0,
    }
    result = {
        "gate": gate,
        "hr_image": hr_image,
        "sr_image": sr_image,
        "sr_selected_mask": sr_selected_mask,
        "detections": detections,
    }
    return result, times


# ---------------------------------------------------------------------------
# Statistics / JSON
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def build_output(
    args: argparse.Namespace,
    project_root: Path | None,
    model: Any,
    num_images: int,
    total_times: List[float],
    stage1_times: List[float],
    stage2_times: List[float],
    gate_values: List[float],
    sr_ratios: List[float],
) -> Dict[str, Any]:
    total_avg = float(np.mean(total_times)) if total_times else None
    total_median = float(np.median(total_times)) if total_times else None
    total_p95 = percentile(total_times, 95.0) if total_times else None
    stage1_avg = float(np.mean(stage1_times)) if stage1_times else None
    stage2_avg = float(np.mean(stage2_times)) if stage2_times else None
    fps = (1000.0 / total_avg) if total_avg and total_avg > 0 else None

    gate_arr = np.asarray(gate_values, dtype=np.float32) if gate_values else np.asarray([], dtype=np.float32)
    gate_thr = float(getattr(model, 'inference_gate_threshold', 0.5))
    ratio_gt = float((gate_arr > gate_thr).mean()) if gate_arr.size else None

    out = {
        "meta": {
            "arch": "arch2_softgate_selective_skip",
            "project_root": str(project_root) if project_root else None,
            "arch2_config": str(Path(args.arch2_config).resolve()),
            "arch2_py": str(Path(args.arch2_py).resolve()) if args.arch2_py else "src.models.pipelines.arch2_softgate",
            "sr_weights": args.sr_weights,
            "gate_weights": args.gate_weights,
            "yolo_weights": args.yolo_weights,
            "images_dir": str(Path(args.images_dir).resolve()),
            "num_images": num_images,
            "device": args.device,
            "half_amp": bool(args.half and args.device.startswith("cuda") and torch.cuda.is_available()),
            "upscale_factor": int(getattr(getattr(model, "config", SimpleNamespace()), "data", SimpleNamespace()).upscale_factor)
            if hasattr(getattr(model, "config", None), "data") and hasattr(model.config.data, "upscale_factor")
            else int(getattr(model, "upscale_factor", 4)),
            "conf": args.conf,
            "iou": args.iou,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "preload": not args.no_preload,
            "selective_inference": bool(getattr(model, "use_selective_inference", False)),
            "gate_threshold": float(getattr(model, "inference_gate_threshold", math.nan)),
            "blend_selected_inference": bool(getattr(model, "blend_selected_inference", False)),
            "avg_ms_per_image": total_avg,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "avg_ms_per_image": total_avg,
        "fps": fps,
        "fps_detail": {"avg": fps},
        "latency_ms": {
            "total_avg": total_avg,
            "total_median": total_median,
            "total_p95": total_p95,
            "stage_avg_gate_sr_blend": stage1_avg,
            "stage_avg_yolo": stage2_avg,
        },
        "gate_stats": {
            "mean": float(gate_arr.mean()) if gate_arr.size else None,
            "std": float(gate_arr.std()) if gate_arr.size else None,
            "min": float(gate_arr.min()) if gate_arr.size else None,
            "max": float(gate_arr.max()) if gate_arr.size else None,
            "ratio_gt_0_5": ratio_gt,
            "sr_applied_ratio_mean": float(np.mean(sr_ratios)) if sr_ratios else None,
            "sr_applied_ratio_std": float(np.std(sr_ratios)) if sr_ratios else None,
        },
    }
    return out


def main() -> None:
    args = parse_args()
    project_root = ensure_project_root(args)

    cfg_path = Path(args.arch2_config).resolve()
    images_dir = Path(args.images_dir).resolve()
    out_json = Path(args.out_json).resolve()

    cfg_dict = read_yaml(cfg_path)
    cfg_dict = patch_config_dict(cfg_dict, args)
    cfg = dict_to_namespace(cfg_dict)

    Arch2SoftGate = load_arch2_class(args)
    model = Arch2SoftGate(cfg)

    requested_device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = model.to(requested_device)
    model.eval()

    image_paths = list_images(images_dir, args.max_images)
    if args.no_preload:
        image_tensors = image_paths
    else:
        image_tensors = [load_image_tensor(p) for p in image_paths]

    warmup_n = max(0, min(args.warmup, len(image_paths)))
    timed_items = image_tensors[warmup_n:]
    if not timed_items:
        raise ValueError("No images left after warmup. Reduce --warmup or increase --max_images.")

    print(f"[Arch2 bench] project_root      : {project_root}")
    print(f"[Arch2 bench] config            : {cfg_path}")
    print(f"[Arch2 bench] arch2_py          : {args.arch2_py or 'src.models.pipelines.arch2_softgate'}")
    print(f"[Arch2 bench] images           : {len(image_paths)} (warmup={warmup_n}, timed={len(timed_items)})")
    print(f"[Arch2 bench] device            : {requested_device}")
    print(f"[Arch2 bench] selective         : {getattr(model, 'use_selective_inference', None)}")
    print(f"[Arch2 bench] gate_threshold    : {getattr(model, 'inference_gate_threshold', None)}")
    print(f"[Arch2 bench] blend_selected    : {getattr(model, 'blend_selected_inference', None)}")

    # Warmup
    for idx in range(warmup_n):
        img = image_tensors[idx] if not args.no_preload else load_image_tensor(image_paths[idx])
        _result, _times = run_arch2_once(model, img, requested_device, args.half, args.conf, args.iou)

    total_times: List[float] = []
    stage1_times: List[float] = []
    stage2_times: List[float] = []
    gate_values: List[float] = []
    sr_ratios: List[float] = []

    for i, item in enumerate(timed_items, start=1):
        img = item if isinstance(item, torch.Tensor) else load_image_tensor(item)
        result, times = run_arch2_once(model, img, requested_device, args.half, args.conf, args.iou)

        total_times.append(times["total_ms"])
        stage1_times.append(times["gate_sr_blend_ms"])
        stage2_times.append(times["yolo_ms"])

        gate = result["gate"].detach().float().cpu().view(-1)
        gate_values.extend(gate.tolist())
        sr_ratio = float(result["sr_selected_mask"].float().mean().item())
        sr_ratios.append(sr_ratio)

        if args.print_every > 0 and (i == 1 or i % args.print_every == 0 or i == len(timed_items)):
            print(
                f"[Arch2 bench] {i:4d}/{len(timed_items)} | "
                f"total={times['total_ms']:.2f} ms | "
                f"stage1={times['gate_sr_blend_ms']:.2f} ms | "
                f"stage2={times['yolo_ms']:.2f} ms | "
                f"sr_ratio={sr_ratio:.3f}"
            )

    out = build_output(
        args=args,
        project_root=project_root,
        model=model,
        num_images=len(timed_items),
        total_times=total_times,
        stage1_times=stage1_times,
        stage2_times=stage2_times,
        gate_values=gate_values,
        sr_ratios=sr_ratios,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("[Arch2 bench] saved ->", out_json)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
