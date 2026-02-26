#!/usr/bin/env python3
"""
IAC Arch0 - SR cache generator

- Input: LR dataset (YOLO format) + HR labels/images(optional, for metrics/size match)
- Output: SR images folder + labels + sr_data.yaml

Design goals:
1) DO NOT run YOLO here (SR only). Evaluation is done separately with yolo.val().
2) Ensure SR image size matches HR image size (label correctness).
3) Save SR as PNG (lossless) to avoid extra JPEG artifacts.


python iac_scripts/arch0_make_sr_cache.py \
  --arch0_config configs/experiment/arch0_sequential.yaml \
  --sr_weights /home/changmin/dark_vessel_sr_yolo/weights/rfdn/model_best.pt \
  --hr_data_yaml /home/changmin/smart_airbus_data/data.yaml \
  --lr_data_yaml /home/changmin/smart_airbus_data_lr/data.yaml \
  --out_dir /home/$USER/tmp_iac/arch0_sr_cache \
  --max_images 20 \
  --clean
"""

import sys
import time
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import argparse
import yaml
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

# --- make `src` importable ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.sr_models.rfdn import RFDN  # requires sr_models/__init__.py to be safe


def dict_to_namespace(d):
    """nested dict -> SimpleNamespace (for convenience)"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    return d


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_val_dirs(data_yaml: str):
    """
    Resolve images/val and labels/val dirs from YOLO data.yaml.
    Assumes common layout: <root>/images/val , <root>/labels/val
    """
    cfg = load_yaml(data_yaml)
    root = Path(cfg.get("path", Path(data_yaml).parent)).expanduser()

    val_rel = cfg.get("val", "images/val")
    # val can be string or list; we take the first if list
    if isinstance(val_rel, list):
        val_rel = val_rel[0]

    val_images_dir = Path(val_rel)
    if not val_images_dir.is_absolute():
        val_images_dir = root / val_images_dir

    # infer labels dir
    # if val is ".../images/val" -> labels should be ".../labels/val"
    if "images" in val_images_dir.parts:
        parts = list(val_images_dir.parts)
        parts[parts.index("images")] = "labels"
        val_labels_dir = Path(*parts)
    else:
        # fallback
        val_labels_dir = root / "labels" / val_images_dir.name

    names = cfg.get("names", {0: "ship"})
    nc = cfg.get("nc", len(names) if isinstance(names, dict) else 1)
    return val_images_dir, val_labels_dir, names, nc


def load_rfdn_from_arch0_config(arch0_config_yaml: str, sr_weights_override: str, device: str):
    """
    Construct RFDN same way as Arch0 pipeline does:
    - LR(0~1) -> *255 -> RFDN -> /255 -> clamp
    """
    cfg = dict_to_namespace(load_yaml(arch0_config_yaml))
    model_cfg = getattr(cfg, "model", cfg)
    data_cfg = getattr(cfg, "data", SimpleNamespace())

    upscale = getattr(data_cfg, "upscale_factor", None) or getattr(data_cfg, "scale_factor", 4)

    rfdn_cfg = getattr(model_cfg, "rfdn", SimpleNamespace())
    sr_cfg = getattr(model_cfg, "sr_config", SimpleNamespace())

    nf = getattr(rfdn_cfg, "nf", 50)
    num_modules = getattr(rfdn_cfg, "num_modules", 4)

    # sr_config can override nf
    if hasattr(sr_cfg, "nf"):
        nf = getattr(sr_cfg, "nf", nf)

    # weights path
    weights_cfg = getattr(model_cfg, "weights", SimpleNamespace())
    sr_w = sr_weights_override or getattr(weights_cfg, "sr_model", None)

    # build model (match Arch0: input_range='0-255')
    model = RFDN(
        in_channels=3,
        out_channels=3,
        nf=nf,
        num_modules=num_modules,
        upscale=upscale,
        input_range="0-255",
    ).to(device)

    if sr_w is None:
        raise FileNotFoundError("SR weights path is None. Provide --sr_weights or set model.weights.sr_model in YAML.")

    sr_w = str(Path(sr_w).expanduser())
    ckpt = torch.load(sr_w, map_location="cpu")

    # robust state_dict extraction (matches your pipeline logic)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "params_ema" in ckpt:
            state = ckpt["params_ema"]
        elif "params" in ckpt:
            state = ckpt["params"]
        else:
            state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state, strict=False)
    model.eval()

    info = {
        "upscale_factor": upscale,
        "nf": nf,
        "num_modules": num_modules,
        "sr_weights": sr_w,
    }
    return model, info


@torch.no_grad()
def sr_forward(rfdn: torch.nn.Module, lr_01: torch.Tensor, half: bool):
    """
    lr_01: (1,3,H,W) float in [0,1]
    """
    if half:
        # SR model in fp16 (speed/mem). Input also half.
        rfdn = rfdn.half()
        lr_01 = lr_01.half()

    lr_255 = lr_01 * 255.0
    sr_255 = rfdn(lr_255)
    sr_01 = torch.clamp(sr_255 / 255.0, 0.0, 1.0)
    return sr_01.float()  # save as float32 for consistent image conversion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch0_config", required=True, help="configs/experiment/arch0_sequential.yaml")
    p.add_argument("--sr_weights", default=None, help="override SR weights path")
    p.add_argument("--hr_data_yaml", required=True, help="HR data.yaml (for HR size + labels)")
    p.add_argument("--lr_data_yaml", required=True, help="LR data.yaml (for LR images)")
    p.add_argument("--out_dir", required=True, help="output root, e.g. /home/user/tmp_eval/arch0_sr")
    p.add_argument("--max_images", type=int, default=0, help="0 means all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--half", action="store_true", help="use fp16 for SR stage")
    p.add_argument("--clean", action="store_true", help="delete out_dir if exists")
    p.add_argument("--compute_sr_metrics", action="store_true", help="compute PSNR/SSIM (slower)")
    args = p.parse_args()

    device = args.device
    out_root = Path(args.out_dir).expanduser()

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # resolve dataset dirs
    hr_images_dir, hr_labels_dir, names, nc = resolve_val_dirs(args.hr_data_yaml)
    lr_images_dir, _, _, _ = resolve_val_dirs(args.lr_data_yaml)

    # load SR model
    rfdn, sr_info = load_rfdn_from_arch0_config(args.arch0_config, args.sr_weights, device)

    # list LR images
    lr_imgs = sorted(list(lr_images_dir.glob("*.jpg")) + list(lr_images_dir.glob("*.png")) + list(lr_images_dir.glob("*.jpeg")))
    if args.max_images and args.max_images > 0:
        lr_imgs = lr_imgs[: args.max_images]

    if len(lr_imgs) == 0:
        raise RuntimeError(f"No images found in {lr_images_dir}")

    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    psnr_vals, ssim_vals = [], []

    t0 = time.time()
    for idx, lr_path in enumerate(lr_imgs, 1):
        # LR load
        lr_pil = Image.open(lr_path).convert("RGB")
        lr = to_tensor(lr_pil).unsqueeze(0).to(device)  # (1,3,H,W), 0~1

        # SR forward
        sr = sr_forward(rfdn, lr, half=args.half)[0].cpu()  # (3,Hs,Ws)

        # determine target size (match HR if available)
        hr_img_path = hr_images_dir / lr_path.name
        if hr_img_path.exists():
            hr_pil = Image.open(hr_img_path).convert("RGB")
            target_w, target_h = hr_pil.size
        else:
            # fallback: upscale LR size
            target_w = lr_pil.size[0] * sr_info["upscale_factor"]
            target_h = lr_pil.size[1] * sr_info["upscale_factor"]
            hr_pil = None

        # resize SR if needed (to keep labels valid!)
        if (sr.shape[1] != target_h) or (sr.shape[2] != target_w):
            sr = F.interpolate(sr.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False)[0]

        # save SR as PNG (lossless)
        out_img_path = out_root / "images" / "val" / f"{lr_path.stem}.png"
        to_pil(sr).save(out_img_path)

        # labels: copy or symlink from HR labels
        src_lbl = hr_labels_dir / f"{lr_path.stem}.txt"
        dst_lbl = out_root / "labels" / "val" / f"{lr_path.stem}.txt"
        if src_lbl.exists() and not dst_lbl.exists():
            try:
                dst_lbl.symlink_to(src_lbl)
            except Exception:
                shutil.copy(src_lbl, dst_lbl)

        # optional SR metrics (PSNR/SSIM)
        if args.compute_sr_metrics and hr_pil is not None:
            hr = to_tensor(hr_pil).unsqueeze(0)  # cpu, 0~1
            sr_for_m = sr.unsqueeze(0)  # cpu
            mse = torch.mean((sr_for_m - hr) ** 2).item()
            psnr = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
            psnr_vals.append(psnr)

            # simple SSIM fallback (fast-ish, not identical to standard but ok for trend)
            # (If you want exact SSIM, later we can plug torchmetrics)
            mu_sr = sr_for_m.mean()
            mu_hr = hr.mean()
            var_sr = sr_for_m.var()
            var_hr = hr.var()
            cov = ((sr_for_m - mu_sr) * (hr - mu_hr)).mean()
            c1, c2 = 0.01**2, 0.03**2
            ssim = ((2 * mu_sr * mu_hr + c1) * (2 * cov + c2)) / ((mu_sr**2 + mu_hr**2 + c1) * (var_sr + var_hr + c2))
            ssim_vals.append(ssim.item())

        if idx % 50 == 0 or idx == len(lr_imgs):
            print(f"[SR] {idx}/{len(lr_imgs)} done")

    total_s = time.time() - t0

    # write sr_data.yaml
    sr_data = {
        "path": str(out_root),
        "train": "images/val",
        "val": "images/val",
        "names": names,
        "nc": nc,
    }
    sr_yaml_path = out_root / "sr_data.yaml"
    with open(sr_yaml_path, "w") as f:
        yaml.safe_dump(sr_data, f, sort_keys=False)

    summary = {
        "sr_yaml": str(sr_yaml_path),
        "out_dir": str(out_root),
        "num_images": len(lr_imgs),
        "time_total_sec": total_s,
        "time_per_image_ms": (total_s / max(len(lr_imgs), 1)) * 1000.0,
        "sr_metrics": {
            "psnr_mean": float(np.mean(psnr_vals)) if psnr_vals else None,
            "ssim_mean": float(np.mean(ssim_vals)) if ssim_vals else None,
        },
        "sr_model_info": sr_info,
        "env": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": device,
        },
    }
    with open(out_root / "sr_cache_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SR cache completed ===")
    print(f"SR yaml: {sr_yaml_path}")
    print(f"Images : {out_root / 'images' / 'val'}")
    print(f"Labels : {out_root / 'labels' / 'val'}")
    print(f"Avg time/image: {summary['time_per_image_ms']:.2f} ms")


if __name__ == "__main__":
    main()

    