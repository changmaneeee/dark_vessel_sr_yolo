#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interpolate two Ultralytics YOLO checkpoints via EMA weights.")
    p.add_argument("--cropft", required=True, help="Path to crop-ft checkpoint")
    p.add_argument("--hardneg", required=True, help="Path to hard-negative checkpoint")
    p.add_argument("--alpha", required=True, type=float, help="w = alpha * cropft + (1-alpha) * hardneg")
    p.add_argument("--out", required=True, help="Output checkpoint path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    crop_path = Path(args.cropft).resolve()
    hard_path = Path(args.hardneg).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    c1 = torch.load(crop_path, map_location="cpu")
    c2 = torch.load(hard_path, map_location="cpu")

    if c1.get("ema") is None or c2.get("ema") is None:
        raise ValueError("Both checkpoints must contain EMA models")

    m1 = c1["ema"]
    m2 = c2["ema"]
    s1 = m1.state_dict()
    s2 = m2.state_dict()

    if set(s1.keys()) != set(s2.keys()):
        raise ValueError("Checkpoint state_dict keys do not match")

    alpha = float(args.alpha)
    blended = copy.deepcopy(m1)
    new_sd = {}
    for k in s1.keys():
        v1 = s1[k]
        v2 = s2[k]
        if v1.shape != v2.shape or v1.dtype != v2.dtype:
            raise ValueError(f"Incompatible tensor at key={k}")
        if torch.is_floating_point(v1):
            new_sd[k] = alpha * v1 + (1.0 - alpha) * v2
        else:
            # Running counters / integer buffers keep crop-ft side.
            new_sd[k] = v1.clone()

    blended.load_state_dict(new_sd, strict=True)

    out_ckpt = copy.deepcopy(c1)
    out_ckpt["epoch"] = max(int(c1.get("epoch", 0)), int(c2.get("epoch", 0)))
    out_ckpt["best_fitness"] = None
    out_ckpt["model"] = None
    out_ckpt["ema"] = blended
    out_ckpt["optimizer"] = None
    out_ckpt["scaler"] = None
    out_ckpt["updates"] = None
    out_ckpt["train_metrics"] = {
        "interpolated": True,
        "alpha": alpha,
        "cropft": str(crop_path),
        "hardneg": str(hard_path),
    }
    torch.save(out_ckpt, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
