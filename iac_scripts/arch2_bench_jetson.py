#!/usr/bin/env python3
"""
Arch2 Jetson benchmark (latency-focused)

We measure:
- end-to-end latency: LR -> (Gate + SR + Blend) -> YOLO
- stage latency: (Gate+SR+Blend) time vs YOLO time
- gate statistics: mean/std, ratio(gate>0.5)

IMPORTANT (Jetson 안정성):
- Do NOT call model.half() for the whole pipeline (Ultralytics fuse dtype issues)
- Use AMP autocast only.
- Trigger YOLO fuse once in FP32 (autocast disabled), then benchmark with autocast.

Usage example:
python iac_scripts/arch2_bench_jetson.py \
  --arch2_config configs/experiment/arch2_softgate.yaml \
  --sr_weights /path/to/rfdn.pt \
  --gate_weights /path/to/gate_best.pt \
  --yolo_weights /path/to/yolo.pt \
  --images_dir /path/to/LR/images/val \
  --max_images 2000 --warmup 20 --device cuda --half \
  --conf 0.25 --iou 0.45 \
  --out_json iac_runs/arch2_bench_jetson_amp.json
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
import torch.nn.functional as F

# --- Make `src/` importable ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pipelines.arch2_softgate import Arch2SoftGate


# ----------------------------
# Utils
# ----------------------------

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    return d


def load_yaml_ns(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return dict_to_namespace(cfg)


def ensure_config_minimum(cfg: SimpleNamespace,
                          yolo_weights: str,
                          upscale_factor_default: int = 4):
    """
    Arch2SoftGate가 필요한 최소 필드를 보장.
    YAML이 다 갖고 있어도 괜찮고, 없으면 기본값을 채움.
    """
    if not hasattr(cfg, "model"):
        cfg.model = SimpleNamespace()

    # rfdn config
    if not hasattr(cfg.model, "rfdn"):
        cfg.model.rfdn = SimpleNamespace(nf=50, num_modules=4)
    if not hasattr(cfg.model.rfdn, "nf"):
        cfg.model.rfdn.nf = 50
    if not hasattr(cfg.model.rfdn, "num_modules"):
        cfg.model.rfdn.num_modules = 4

    # gate config
    if not hasattr(cfg.model, "gate"):
        cfg.model.gate = SimpleNamespace(base_channels=32, num_layers=4)
    if not hasattr(cfg.model.gate, "base_channels"):
        cfg.model.gate.base_channels = 32
    if not hasattr(cfg.model.gate, "num_layers"):
        cfg.model.gate.num_layers = 4

    # yolo config
    if not hasattr(cfg.model, "yolo"):
        cfg.model.yolo = SimpleNamespace(weights_path=yolo_weights, num_classes=1)
    if not hasattr(cfg.model.yolo, "weights_path"):
        cfg.model.yolo.weights_path = yolo_weights
    else:
        cfg.model.yolo.weights_path = yolo_weights
    if not hasattr(cfg.model.yolo, "num_classes"):
        cfg.model.yolo.num_classes = 1

    # data config
    if not hasattr(cfg, "data"):
        cfg.data = SimpleNamespace(upscale_factor=upscale_factor_default)
    if not hasattr(cfg.data, "upscale_factor") and not hasattr(cfg.data, "scale_factor"):
        cfg.data.upscale_factor = upscale_factor_default

    # training config (있어도 되고 없어도 됨)
    if not hasattr(cfg, "training"):
        cfg.training = SimpleNamespace(sr_weight=0.0, det_weight=1.0)
    if not hasattr(cfg.training, "sr_weight"):
        cfg.training.sr_weight = 0.0
    if not hasattr(cfg.training, "det_weight"):
        cfg.training.det_weight = 1.0

    return cfg


def list_images(images_dir: Path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for e in exts:
        paths += list(images_dir.glob(e))
    return sorted(paths)


def preprocess_cv2_to_tensor(img_bgr):
    """
    OpenCV BGR -> torch tensor (1,3,H,W), float32 in [0,1]
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float() / 255.0
    return t.unsqueeze(0)


def percentile(arr, p):
    if len(arr) == 0:
        return None
    return float(np.percentile(np.array(arr, dtype=np.float32), p))


def load_state_dict_flexible(ckpt):
    """
    checkpoint dict일 때 자주 쓰는 키들을 자동으로 찾아 state_dict 반환
    """
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict", "params_ema", "params", "net", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def strip_prefix(sd: dict):
    out = {}
    for k, v in sd.items():
        nk = k
        for p in ["module.", "model.", "net.", "net_g."]:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser()

    # Required
    p.add_argument("--arch2_config", required=True, help="configs/experiment/arch2_softgate.yaml")
    p.add_argument("--sr_weights", required=True, help="RFDN weights path")
    p.add_argument("--gate_weights", required=True, help="Gate weights path")
    p.add_argument("--yolo_weights", required=True, help="YOLO weights path")
    p.add_argument("--images_dir", required=True, help="LR images folder (e.g., LR/images/val)")

    # Runtime
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--half", action="store_true", help="Use AMP autocast (recommended on Jetson)")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations (not measured)")
    p.add_argument("--max_images", type=int, default=0, help="0=all images")
    p.add_argument("--conf", type=float, default=0.25, help="Deployment conf threshold")
    p.add_argument("--iou", type=float, default=0.45, help="Deployment NMS IoU threshold")

    # Preload (to reduce disk I/O jitter)
    p.add_argument("--no_preload", action="store_true", help="Disable image preload to save RAM")

    # Output
    p.add_argument("--out_json", default="iac_runs/arch2_bench_jetson.json")

    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    images_dir = Path(args.images_dir).expanduser()
    img_paths = list_images(images_dir)
    if args.max_images and args.max_images > 0:
        img_paths = img_paths[: args.max_images]
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in: {images_dir}")

    # Speed-related settings
    torch.set_grad_enabled(False)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- Load config + build model ----
    cfg = load_yaml_ns(args.arch2_config)
    cfg = ensure_config_minimum(cfg, yolo_weights=args.yolo_weights)

    cfg.device = device  # some pipelines read this

    # upscale factor
    upscale = getattr(cfg.data, "upscale_factor", None) or getattr(cfg.data, "scale_factor", 4)

    print("\n" + "=" * 70)
    print("Arch2 Jetson Bench")
    print("=" * 70)
    print(f"device    : {device}")
    print(f"half(AMP) : {args.half}")
    print(f"upscale   : {upscale}")
    print(f"conf/iou  : {args.conf} / {args.iou}")
    print(f"images    : {len(img_paths)} from {images_dir}")

    print("\n[1/5] Building Arch2 pipeline...")
    model = Arch2SoftGate(cfg).to(device)
    model.eval()

    # ---- Load SR weights ----
    print("[2/5] Loading SR (RFDN) weights...")
    sr_ckpt = torch.load(args.sr_weights, map_location="cpu", weights_only=False)
    sr_sd = strip_prefix(load_state_dict_flexible(sr_ckpt))
    try:
        model.sr_model.load_state_dict(sr_sd, strict=True)
        print(f"  ✓ SR loaded (strict=True): {args.sr_weights}")
    except Exception:
        model.sr_model.load_state_dict(sr_sd, strict=False)
        print(f"  ✓ SR loaded (strict=False): {args.sr_weights}")

    # ---- Load Gate weights ----
    print("[3/5] Loading Gate weights...")
    gate_ckpt = torch.load(args.gate_weights, map_location="cpu", weights_only=False)
    gate_sd = strip_prefix(load_state_dict_flexible(gate_ckpt))
    # gate checkpoint는 gate_state_dict 같은 키가 있을 수 있어 한 번 더 체크
    if isinstance(gate_ckpt, dict) and "gate_state_dict" in gate_ckpt and isinstance(gate_ckpt["gate_state_dict"], dict):
        gate_sd = strip_prefix(gate_ckpt["gate_state_dict"])
    try:
        model.gate_network.load_state_dict(gate_sd, strict=True)
        print(f"  ✓ Gate loaded (strict=True): {args.gate_weights}")
    except Exception:
        model.gate_network.load_state_dict(gate_sd, strict=False)
        print(f"  ✓ Gate loaded (strict=False): {args.gate_weights}")

    # ---- Prepare inputs (preload or stream) ----
    print("[4/5] Preparing inputs...")
    tensors = None
    if not args.no_preload:
        tensors = []
        for pth in img_paths:
            img = cv2.imread(str(pth))
            if img is None:
                continue
            tensors.append(preprocess_cv2_to_tensor(img))
        if len(tensors) == 0:
            raise RuntimeError("All images failed to load.")
        print(f"  ✓ Preloaded {len(tensors)} images to RAM")
    else:
        print("  ✓ Preload disabled (will read images from disk during benchmark)")

    # ---- YOLO fuse warmup in FP32 (autocast disabled) ----
    print("[5/5] YOLO fuse warmup (FP32, autocast disabled)...")
    # make dummy HR size based on first LR image size
    if tensors is not None:
        lr0 = tensors[0]
    else:
        img0 = cv2.imread(str(img_paths[0]))
        lr0 = preprocess_cv2_to_tensor(img0)

    H_lr, W_lr = int(lr0.shape[2]), int(lr0.shape[3])
    H_hr, W_hr = H_lr * int(upscale), W_lr * int(upscale)
    dummy = torch.zeros(1, 3, H_hr, W_hr, device=device, dtype=torch.float32)

    if device == "cuda":
        torch.cuda.synchronize()
    with torch.cuda.amp.autocast(enabled=False):
        _ = model.detector.predict(dummy, conf=args.conf, iou=args.iou)
    if device == "cuda":
        torch.cuda.synchronize()
    print("  ✓ fuse done")

    # ---- Warmup loop (not measured) ----
    print("\nWarmup...")
    warm_n = min(args.warmup, len(img_paths))
    for i in range(warm_n):
        if tensors is not None:
            lr = tensors[i].to(device, non_blocking=True)
        else:
            img = cv2.imread(str(img_paths[i]))
            if img is None:
                continue
            lr = preprocess_cv2_to_tensor(img).to(device)

        # end-to-end warmup with AMP (if enabled)
        with torch.cuda.amp.autocast(enabled=(args.half and device == "cuda")):
            # Gate
            gate = model.gate_network(lr)  # shape may be [B,1] or others
            g = gate.view(gate.shape[0], -1).mean(dim=1).clamp(0, 1).view(-1, 1, 1, 1)

            # SR
            sr_255 = model.sr_model(lr * 255.0)
            sr = torch.clamp(sr_255 / 255.0, 0.0, 1.0)

            # Upsample + Blend
            up = F.interpolate(lr, scale_factor=upscale, mode="bilinear", align_corners=False)
            hr = torch.clamp(g * sr + (1.0 - g) * up, 0.0, 1.0)

            # YOLO
            _ = model.detector.predict(hr, conf=args.conf, iou=args.iou)

        if device == "cuda":
            torch.cuda.synchronize()

    # ---- Timed benchmark ----
    print("\nBenchmarking...")
    t_total_ms = []
    t_stage_ms = []  # Gate+SR+Blend
    t_yolo_ms = []
    gate_vals = []

    n = len(img_paths) if tensors is None else len(tensors)

    for i in range(n):
        if tensors is not None:
            lr = tensors[i].to(device, non_blocking=True)
        else:
            img = cv2.imread(str(img_paths[i]))
            if img is None:
                continue
            lr = preprocess_cv2_to_tensor(img).to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.cuda.amp.autocast(enabled=(args.half and device == "cuda")):
            # -------------------------
            # Stage: Gate + SR + Blend
            # -------------------------
            s0 = time.perf_counter()

            gate = model.gate_network(lr)
            g_scalar = gate.view(gate.shape[0], -1).mean(dim=1).clamp(0, 1)  # [B]
            g = g_scalar.view(-1, 1, 1, 1)

            sr_255 = model.sr_model(lr * 255.0)
            sr = torch.clamp(sr_255 / 255.0, 0.0, 1.0)

            up = F.interpolate(lr, scale_factor=upscale, mode="bilinear", align_corners=False)
            hr = torch.clamp(g * sr + (1.0 - g) * up, 0.0, 1.0)

            s1 = time.perf_counter()

            # -------------------------
            # Stage: YOLO
            # -------------------------
            y0 = time.perf_counter()
            det = model.detector.predict(hr, conf=args.conf, iou=args.iou)
            y1 = time.perf_counter()

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        t_total_ms.append((t1 - t0) * 1000.0)
        t_stage_ms.append((s1 - s0) * 1000.0)
        t_yolo_ms.append((y1 - y0) * 1000.0)
        gate_vals.append(float(g_scalar[0].detach().float().cpu().item()))

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n} | last total {t_total_ms[-1]:.2f} ms | gate {gate_vals[-1]:.3f}")

    # ---- Summary ----
    avg_total = float(np.mean(t_total_ms))
    med_total = float(np.median(t_total_ms))
    p95_total = percentile(t_total_ms, 95)
    fps = 1000.0 / avg_total if avg_total > 0 else None

    avg_stage = float(np.mean(t_stage_ms))
    avg_yolo = float(np.mean(t_yolo_ms))

    gv = np.array(gate_vals, dtype=np.float32) if gate_vals else np.array([0.0], dtype=np.float32)

    summary = {
        "meta": {
            "arch": "arch2_softgate",
            "arch2_config": args.arch2_config,
            "sr_weights": args.sr_weights,
            "gate_weights": args.gate_weights,
            "yolo_weights": args.yolo_weights,
            "images_dir": str(images_dir),
            "num_images": int(len(t_total_ms)),
            "device": device,
            "half_amp": bool(args.half and device == "cuda"),
            "upscale_factor": int(upscale),
            "conf": float(args.conf),
            "iou": float(args.iou),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "preload": (not args.no_preload),
        },
        "latency_ms": {
            "total_avg": avg_total,
            "total_median": med_total,
            "total_p95": p95_total,
            "stage_avg_gate_sr_blend": avg_stage,
            "stage_avg_yolo": avg_yolo,
        },
        "fps": {
            "avg": fps
        },
        "gate_stats": {
            "mean": float(gv.mean()),
            "std": float(gv.std()),
            "min": float(gv.min()),
            "max": float(gv.max()),
            "ratio_gt_0_5": float((gv > 0.5).mean()),
        }
    }

    out_json = Path(args.out_json).expanduser()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BENCH DONE ===")
    print(f"Saved: {out_json}")
    print(f"Avg   : {avg_total:.2f} ms  | FPS ~ {fps:.2f}")
    print(f"Median: {med_total:.2f} ms  | P95 : {p95_total:.2f} ms")
    print(f"Stage(avg) Gate+SR+Blend: {avg_stage:.2f} ms | YOLO: {avg_yolo:.2f} ms")
    print(f"Gate mean/std: {summary['gate_stats']['mean']:.4f} / {summary['gate_stats']['std']:.4f} "
          f"(>0.5: {summary['gate_stats']['ratio_gt_0_5']*100:.1f}%)")


if __name__ == "__main__":
    main()