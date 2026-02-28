#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import yaml
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.pipelines.arch4_adaptive import Arch4Adaptive


def load_yaml(p):
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def bgr_to_tensor(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float() / 255.0
    return t.unsqueeze(0)


def draw_boxes(img_bgr, det, color=(0, 255, 0), name="ship"):
    """det: dict with boxes(xyxy), scores, classes"""
    out = img_bgr.copy()
    boxes = det["boxes"].detach().cpu().numpy() if torch.is_tensor(det["boxes"]) else det["boxes"]
    scores = det["scores"].detach().cpu().numpy() if torch.is_tensor(det["scores"]) else det["scores"]
    for b, s in zip(boxes, scores):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{name} {s:.2f}", (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def flatten_pred(x):
    """debug_info['pass1_raw'] / pass2_raw 형태가 list/dict 섞일 수 있어서 flatten"""
    out = []
    if isinstance(x, dict) and "boxes" in x:
        return [x]
    if isinstance(x, list):
        for a in x:
            out += flatten_pred(a)
    return out


def summarize(raw, tag):
    dets = flatten_pred(raw)
    total = 0
    mx = 0.0
    for d in dets:
        if "boxes" in d and torch.is_tensor(d["boxes"]):
            n = int(d["boxes"].shape[0])
            total += n
            if n > 0 and "scores" in d:
                mx = max(mx, float(d["scores"].max().detach().cpu().item()))
    print(f"[{tag}] total_boxes={total}, max_conf={mx:.4f}")


def save_tensor_img(t, save_path):
    """t: (3,H,W) in [0,1]"""
    c = t.detach().float().cpu()
    img = (c.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch4_config", required=True)
    ap.add_argument("--lr_image", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_dir", default="iac_runs/arch4_one")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(args.arch4_config)
    model = Arch4Adaptive(cfg)
    model.eval()

    img = cv2.imread(args.lr_image)
    assert img is not None, f"failed to read: {args.lr_image}"

    lr = bgr_to_tensor(img).to(device)

    with torch.no_grad():
        out = model.forward(lr, debug=True)

    det = out["detections"][0]
    dbg = out["debug_info"]

    print("\n==== SUMMARY ====")
    summarize(dbg.get("pass1_raw", None), "PASS1(SCOUT)")
    summarize(dbg.get("pass2_raw", None), "PASS2(SNIPER)")
    n_final = int(det["boxes"].shape[0])
    mx_final = float(det["scores"].max().item()) if n_final else 0.0
    print(f"[FINAL] det={n_final}, max_conf={mx_final:.4f}")

    # save final overlay
    vis = draw_boxes(img, det, color=(0, 255, 0))
    out_img = save_dir / "final_overlay.jpg"
    cv2.imwrite(str(out_img), vis)
    print("saved:", out_img)

    # save crops
    crops_lr = dbg.get("crops_lr", [])
    crops_sr = dbg.get("crops_sr", [])
    for i, (cl, cs) in enumerate(zip(crops_lr, crops_sr)):
        save_tensor_img(cl, save_dir / f"crop_lr_{i:03d}.png")
        save_tensor_img(cs, save_dir / f"crop_sr_{i:03d}.png")
    if len(crops_sr):
        print(f"saved {len(crops_sr)} crops to {save_dir}")


if __name__ == "__main__":
    main()