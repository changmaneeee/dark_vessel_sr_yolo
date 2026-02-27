#!/usr/bin/env python3
"""
IAC Arch2 - output cache generator (NO YOLO here)

Output image = gate * SR(LR) + (1-gate) * bilinear_upsample(LR)

- Input: LR data.yaml + HR data.yaml(라벨 복사/HR 크기 맞춤)
- Output: out_dir/images/val/*.png + out_dir/labels/val/*.txt + out_dir/sr_data.yaml

python iac_scripts/arch2_make_sr_cache.py \
  --arch2_config configs/experiment/arch2_softgate.yaml \
  --hr_data_yaml /home/changmin/smart_airbus_data/data.yaml \
  --lr_data_yaml /home/changmin/smart_airbus_data_lr/data.yaml \
  --out_dir /home/changmin/tmp_iac/arch2_sr_cache \
  --max_images 0 \
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

from src.models.sr_models.rfdn import RFDN
from src.models.gates.soft_gate import LightweightGateV1


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    return d


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_val_dirs(data_yaml: str):
    cfg = load_yaml(data_yaml)
    root = Path(cfg.get("path", Path(data_yaml).parent)).expanduser()

    val_rel = cfg.get("val", "images/val")
    if isinstance(val_rel, list):
        val_rel = val_rel[0]

    val_images_dir = Path(val_rel)
    if not val_images_dir.is_absolute():
        val_images_dir = root / val_images_dir

    if "images" in val_images_dir.parts:
        parts = list(val_images_dir.parts)
        parts[parts.index("images")] = "labels"
        val_labels_dir = Path(*parts)
    else:
        val_labels_dir = root / "labels" / val_images_dir.name

    names = cfg.get("names", {0: "ship"})
    nc = cfg.get("nc", len(names) if isinstance(names, dict) else 1)
    return val_images_dir, val_labels_dir, names, nc


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict", "model", "net"]:
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


@torch.no_grad()
def save_tensor_png(img01: torch.Tensor, path: Path):
    img01 = img01.detach().clamp(0, 1).cpu()
    arr = (img01.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--arch2_config", required=True, help="configs/experiment/arch2_softgate.yaml")
    p.add_argument("--hr_data_yaml", required=True)
    p.add_argument("--lr_data_yaml", required=True)

    p.add_argument("--sr_weights", default=None, help="override RFDN weights")
    p.add_argument("--gate_weights", default=None, help="override Gate weights")

    p.add_argument("--out_dir", required=True)
    p.add_argument("--clean", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--half", action="store_true", help="FP16 for SR+Gate (PC cache speedup)")
    p.add_argument("--max_images", type=int, default=0, help="0=all")

    args = p.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    out_root = Path(args.out_dir).expanduser()

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # dataset dirs
    hr_images_dir, hr_labels_dir, names, nc = resolve_val_dirs(args.hr_data_yaml)
    lr_images_dir, _, _, _ = resolve_val_dirs(args.lr_data_yaml)

    lr_imgs = sorted(list(lr_images_dir.glob("*.jpg")) + list(lr_images_dir.glob("*.png")) + list(lr_images_dir.glob("*.jpeg")))
    if args.max_images and args.max_images > 0:
        lr_imgs = lr_imgs[: args.max_images]
    if len(lr_imgs) == 0:
        raise RuntimeError(f"No images found in {lr_images_dir}")

    # load arch2 yaml
    cfg = dict_to_namespace(load_yaml(args.arch2_config))
    model_cfg = getattr(cfg, "model", cfg)
    data_cfg = getattr(cfg, "data", SimpleNamespace())

    upscale = getattr(data_cfg, "upscale_factor", None) or getattr(data_cfg, "scale_factor", 4)

    rfdn_cfg = getattr(model_cfg, "rfdn", SimpleNamespace())
    gate_cfg = getattr(model_cfg, "gate", SimpleNamespace())

    nf = getattr(rfdn_cfg, "nf", 50)
    num_modules = getattr(rfdn_cfg, "num_modules", 4)

    sr_w = args.sr_weights or getattr(rfdn_cfg, "pretrain_path", None)
    gate_w = args.gate_weights or getattr(gate_cfg, "weights_path", None)

    if sr_w is None:
        raise FileNotFoundError("SR weights not set. Use --sr_weights or set model.rfdn.pretrain_path.")
    sr_w = str(Path(sr_w).expanduser())

    # models
    gate_net = LightweightGateV1(
        in_channels=getattr(gate_cfg, "in_channels", 3),
        base_channels=getattr(gate_cfg, "base_channels", 32),
        num_layers=getattr(gate_cfg, "num_layers", 4),
    ).to(device).eval()

    sr_model = RFDN(
        in_channels=3, out_channels=3,
        nf=nf, num_modules=num_modules,
        upscale=upscale,
        input_range="0-255",
    ).to(device).eval()

    # load weights
    sr_model.load_pretrained(sr_w)

    if gate_w and Path(gate_w).exists():
        ckpt = torch.load(gate_w, map_location="cpu", weights_only=False)
        sd = strip_prefix(extract_state_dict(ckpt))
        try:
            gate_net.load_state_dict(sd, strict=True)
        except Exception:
            gate_net.load_state_dict(sd, strict=False)
        print(f"[Arch2 cache] ✓ Gate weights loaded: {gate_w}")
    else:
        print("[Arch2 cache] ⚠️ Gate weights not found -> random init (accuracy will be meaningless)")

    if args.half and device == "cuda":
        gate_net = gate_net.half()
        sr_model = sr_model.half()

    to_tensor = T.ToTensor()

    gate_vals = []
    times_ms = []

    t0 = time.time()
    for idx, lr_path in enumerate(lr_imgs, 1):
        lr_pil = Image.open(lr_path).convert("RGB")
        lr = to_tensor(lr_pil).unsqueeze(0).to(device)  # [1,3,H,W] 0~1
        if args.half and device == "cuda":
            lr = lr.half()

        # forward
        if device == "cuda":
            torch.cuda.synchronize()
        s0 = time.perf_counter()

        gate = gate_net(lr)                 # [1,1]
        g = gate.view(1, 1, 1, 1)

        lr_255 = lr * 255.0
        sr_255 = sr_model(lr_255)
        sr = torch.clamp(sr_255 / 255.0, 0.0, 1.0)

        up = F.interpolate(lr, scale_factor=upscale, mode="bilinear", align_corners=False)
        out = g * sr + (1.0 - g) * up       # [1,3,Hs,Ws]

        if device == "cuda":
            torch.cuda.synchronize()
        s1 = time.perf_counter()
        times_ms.append((s1 - s0) * 1000.0)

        gate_vals.append(float(gate.squeeze().float().cpu().item()))

        # match HR size (label validity)
        hr_img_path = hr_images_dir / lr_path.name
        if hr_img_path.exists():
            hr_w, hr_h = Image.open(hr_img_path).convert("RGB").size
            if (out.shape[2] != hr_h) or (out.shape[3] != hr_w):
                out = F.interpolate(out, size=(hr_h, hr_w), mode="bilinear", align_corners=False)

        out_img_path = out_root / "images" / "val" / f"{lr_path.stem}.png"
        save_tensor_png(out[0].float(), out_img_path)

        # copy labels from HR
        src_lbl = hr_labels_dir / f"{lr_path.stem}.txt"
        dst_lbl = out_root / "labels" / "val" / f"{lr_path.stem}.txt"
        if src_lbl.exists() and not dst_lbl.exists():
            shutil.copy(src_lbl, dst_lbl)

        if idx % 2000 == 0 or idx == len(lr_imgs):
            print(f"[Arch2 cache] {idx}/{len(lr_imgs)} done | gate {gate_vals[-1]:.3f}")

    total_s = time.time() - t0

    sr_data = {
        "path": str(out_root),
        "train": "images/val",
        "val": "images/val",
        "names": names,
        "nc": nc,
    }
    sr_yaml = out_root / "sr_data.yaml"
    with open(sr_yaml, "w") as f:
        yaml.safe_dump(sr_data, f, sort_keys=False)

    gv = np.array(gate_vals, dtype=np.float32) if gate_vals else np.array([0.0], dtype=np.float32)
    meta = {
        "arch": "arch2_softgate",
        "out_dir": str(out_root),
        "sr_yaml": str(sr_yaml),
        "num_images": len(lr_imgs),
        "device": device,
        "half": bool(args.half and device == "cuda"),
        "upscale_factor": upscale,
        "timing": {
            "avg_ms": float(np.mean(times_ms)),
            "median_ms": float(np.median(times_ms)),
            "p95_ms": float(np.percentile(times_ms, 95)),
            "total_sec": float(total_s),
        },
        "gate": {
            "mean": float(gv.mean()),
            "std": float(gv.std()),
            "min": float(gv.min()),
            "max": float(gv.max()),
            "ratio_gt_0_5": float((gv > 0.5).mean()),
        },
        "weights": {
            "sr": sr_w,
            "gate": gate_w,
        }
    }
    with open(out_root / "arch2_cache_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Arch2 cache completed ===")
    print(f"SR yaml: {sr_yaml}")
    print(f"Avg time/image: {meta['timing']['avg_ms']:.2f} ms")
    print(f"Gate mean/std: {meta['gate']['mean']:.4f} / {meta['gate']['std']:.4f}")
    print(f"Gate>0.5 ratio: {meta['gate']['ratio_gt_0_5']*100:.1f}%")

if __name__ == "__main__":
    main()