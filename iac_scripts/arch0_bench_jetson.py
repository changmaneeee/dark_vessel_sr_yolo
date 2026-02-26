#!/usr/bin/env python3
"""
Arch0 Jetson benchmark (latency-focused)

What we measure:
- End-to-end latency per image:  LR -> SR -> YOLO -> detections
- (Optional) stage timing: SR time / YOLO time

Why this script exists:
- yolo.val() is great for ACCURACY metrics (mAP, P/R/F1),
  but it is not designed for measuring real deployment latency on Jetson.
- On Jetson, we care about ms/image, FPS, and later power efficiency.

How to use:
- Provide arch0 config yaml + sr weights + yolo weights
- Provide a folder of LR images (e.g. LR val images)
"""

import sys
import time
import json
import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml
import cv2
import numpy as np
import torch

# --- Make `src/` importable even when running from iac_scripts/ ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pipelines.arch0_sequential import Arch0Sequential


def dict_to_namespace(d):
    """Recursively convert dict -> SimpleNamespace (so code can use getattr)."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    return d


def load_config_yaml(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return dict_to_namespace(cfg)


def list_images(images_dir: Path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for e in exts:
        paths += list(images_dir.glob(e))
    return sorted(paths)


def preprocess_cv2_to_tensor(img_bgr):
    """
    Convert OpenCV BGR image -> torch tensor (1,3,H,W) float in [0,1]
    This matches your inference_arch0.py style.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def percentile(arr, p):
    """Simple percentile helper (p in [0,100])."""
    if len(arr) == 0:
        return None
    return float(np.percentile(np.array(arr, dtype=np.float32), p))


def main():
    p = argparse.ArgumentParser()

    # --- Required ---
    p.add_argument("--arch0_config", required=True, help="configs/experiment/arch0_sequential.yaml")
    p.add_argument("--sr_weights", required=True, help="RFDN weights path")
    p.add_argument("--yolo_weights", required=True, help="YOLO weights path")
    p.add_argument("--images_dir", required=True, help="LR images folder (e.g. LR val images)")

    # --- Runtime / measurement options ---
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--half", action="store_true", help="Use FP16 (recommended on Jetson)")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations (not measured)")
    p.add_argument("--max_images", type=int, default=0, help="0 = all images")
    p.add_argument("--conf", type=float, default=0.25, help="Deployment conf threshold")
    p.add_argument("--iou", type=float, default=0.45, help="Deployment NMS IoU threshold")

    # --- Output ---
    p.add_argument("--out_json", default="iac_runs/arch0_bench_jetson.json")
    args = p.parse_args()

    images_dir = Path(args.images_dir).expanduser()
    img_paths = list_images(images_dir)
    if args.max_images and args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in: {images_dir}")

    device = args.device
    assert device in ["cuda", "cpu"]

    # ---- Load config and patch weights paths (so you don't need to edit YAML every time) ----
    cfg = load_config_yaml(args.arch0_config)

    # Many configs keep weights under cfg.model.weights.*
    if not hasattr(cfg, "model"):
        cfg.model = SimpleNamespace()

    if not hasattr(cfg.model, "weights"):
        cfg.model.weights = SimpleNamespace()

    cfg.model.weights.sr_model = args.sr_weights
    cfg.model.weights.detector = args.yolo_weights

    # Also set YOLO weights_path (some code reads from model.yolo.weights_path)
    if not hasattr(cfg.model, "yolo"):
        cfg.model.yolo = SimpleNamespace()
    cfg.model.yolo.weights_path = args.yolo_weights

    # Device into pipeline
    cfg.device = device

    # ---- Build pipeline ----
    print("[1/4] Building Arch0 pipeline...")
    model = Arch0Sequential(cfg).to(device)
    model.eval()

    # FP16 option
    # We keep this simple: if --half, convert model to half and input to half.
    # (YOLO wrapper may internally handle fp16 too, but at least SR part benefits.)
    if args.half and device == "cuda":
        model = model.half()

    # ---- Preload tensors (optional for stable timing) ----
    # On Jetson, disk I/O can cause jitter.
    # For pure "model latency", it helps to preload images into RAM.
    print("[2/4] Preloading images (to reduce disk I/O jitter)...")
    tensors = []
    for pth in img_paths:
        img = cv2.imread(str(pth))
        if img is None:
            continue
        t = preprocess_cv2_to_tensor(img)
        tensors.append(t)

    if len(tensors) == 0:
        raise RuntimeError("All images failed to load.")

    # ---- Warmup ----
    print("[3/4] Warmup...")
    with torch.no_grad():
        for i in range(min(args.warmup, len(tensors))):
            lr = tensors[i].to(device, non_blocking=True)
            if args.half and device == "cuda":
                lr = lr.half()

            _ = model.inference(lr, conf_threshold=args.conf, iou_threshold=args.iou)
            if device == "cuda":
                torch.cuda.synchronize()

    # ---- Timed loop ----
    print("[4/4] Benchmarking...")
    times_ms = []

    with torch.no_grad():
        for i, lr_cpu in enumerate(tensors):
            lr = lr_cpu.to(device, non_blocking=True)
            if args.half and device == "cuda":
                lr = lr.half()

            # IMPORTANT:
            # For GPU timing with CPU clock, synchronize before/after.
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            out = model.inference(lr, conf_threshold=args.conf, iou_threshold=args.iou)

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            dt_ms = (t1 - t0) * 1000.0
            times_ms.append(dt_ms)

            if (i + 1) % 50 == 0 or (i + 1) == len(tensors):
                print(f"  {i+1}/{len(tensors)} | last: {dt_ms:.2f} ms")

    # ---- Summary ----
    avg = float(np.mean(times_ms))
    med = float(np.median(times_ms))
    p95 = percentile(times_ms, 95)

    fps_avg = 1000.0 / avg if avg > 0 else None

    summary = {
        "meta": {
            "arch0_config": args.arch0_config,
            "sr_weights": args.sr_weights,
            "yolo_weights": args.yolo_weights,
            "images_dir": str(images_dir),
            "num_images": len(times_ms),
            "device": device,
            "half": args.half,
            "conf": args.conf,
            "iou": args.iou,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "latency_ms": {
            "avg": avg,
            "median": med,
            "p95": p95,
        },
        "fps": {
            "avg": fps_avg
        }
    }

    out_json = Path(args.out_json).expanduser()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BENCH DONE ===")
    print(f"Saved: {out_json}")
    print(f"Avg   : {avg:.2f} ms  | FPS ~ {fps_avg:.2f}")
    print(f"Median: {med:.2f} ms  | P95 : {p95:.2f} ms")


if __name__ == "__main__":
    main()