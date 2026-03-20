#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import glob
import json
from pathlib import Path
from typing import Any

import torch
import yaml


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def set_nested(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def load_ema_state(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ema = ckpt.get("ema")
    if ema is None or not hasattr(ema, "state_dict"):
        raise ValueError(f"{path} does not contain EMA model state")
    return ckpt, ema.state_dict()


def cmd_interpolate(args: argparse.Namespace) -> None:
    a_path = Path(args.ckpt_a).resolve()
    b_path = Path(args.ckpt_b).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_a, sd_a = load_ema_state(a_path)
    _, sd_b = load_ema_state(b_path)

    if set(sd_a.keys()) != set(sd_b.keys()):
        raise ValueError("State dict keys do not match")

    alpha = float(args.alpha)
    blended_model = copy.deepcopy(ckpt_a["ema"])
    new_sd: dict[str, torch.Tensor] = {}
    for k in sd_a.keys():
        va = sd_a[k]
        vb = sd_b[k]
        if va.shape != vb.shape or va.dtype != vb.dtype:
            raise ValueError(f"Incompatible tensor at key={k}")
        if torch.is_floating_point(va):
            new_sd[k] = alpha * va + (1.0 - alpha) * vb
        else:
            new_sd[k] = va.clone()
    blended_model.load_state_dict(new_sd, strict=True)

    out_ckpt = copy.deepcopy(ckpt_a)
    out_ckpt["model"] = None
    out_ckpt["ema"] = blended_model
    out_ckpt["optimizer"] = None
    out_ckpt["scaler"] = None
    out_ckpt["updates"] = None
    out_ckpt["train_metrics"] = {
        "interpolated": True,
        "alpha": alpha,
        "ckpt_a": str(a_path),
        "ckpt_b": str(b_path),
    }
    torch.save(out_ckpt, out_path)
    print(str(out_path))


def cmd_patch_config(args: argparse.Namespace) -> None:
    in_path = Path(args.base).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for item in args.set or []:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item}")
        k, v = item.split("=", 1)
        set_nested(cfg, k, parse_scalar(v))
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(str(out_path))


def load_direct_result(path: Path) -> tuple[float, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    r = d.get("results", {}).get("sr", {})
    return float(r.get("f1_50_direct", 0.0)), r


def cmd_extract(args: argparse.Namespace) -> None:
    path = Path(args.json).resolve()
    f1, r = load_direct_result(path)
    if args.mode == "f1":
        print(f"{f1:.4f}")
        return
    print(
        "P={p:.4f} R={r:.4f} F1={f1:.4f} TP={tp} FP={fp} FN={fn} ms={ms:.2f}".format(
            p=float(r.get("precision50_direct", 0.0)),
            r=float(r.get("recall50_direct", 0.0)),
            f1=f1,
            tp=int(r.get("tp50", 0)),
            fp=int(r.get("fp50", 0)),
            fn=int(r.get("fn50", 0)),
            ms=float(r.get("avg_ms_per_image", 0.0)),
        )
    )


def cmd_choose_best(args: argparse.Namespace) -> None:
    candidates: list[Path] = []
    for pattern in args.glob:
        for p in sorted(glob.glob(pattern)):
            candidates.append(Path(p))
    if not candidates:
        raise FileNotFoundError("No result json matched")
    best_path = None
    best_f1 = -1.0
    for p in candidates:
        try:
            f1, _ = load_direct_result(p)
        except Exception:
            continue
        if f1 > best_f1:
            best_f1 = f1
            best_path = p
    if best_path is None:
        raise RuntimeError("No valid direct result JSON found")
    print(str(best_path.resolve()))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Arch4 overnight helper utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("interpolate")
    s.add_argument("--ckpt-a", required=True)
    s.add_argument("--ckpt-b", required=True)
    s.add_argument("--alpha", required=True, type=float)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_interpolate)

    s = sub.add_parser("patch-config")
    s.add_argument("--base", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--set", action="append", default=[])
    s.set_defaults(func=cmd_patch_config)

    s = sub.add_parser("extract")
    s.add_argument("--json", required=True)
    s.add_argument("--mode", choices=["f1", "prf"], default="prf")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("choose-best")
    s.add_argument("--glob", action="append", required=True)
    s.set_defaults(func=cmd_choose_best)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
