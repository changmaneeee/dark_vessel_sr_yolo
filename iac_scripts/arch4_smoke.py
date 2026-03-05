#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import yaml
import cv2
import torch
import numpy as np
"""
python iac_scripts/arch4_smoke.py \
  --arch4_config configs/experiment/arch4_adaptive.yaml \
  --lr_images_dir /home/changmin/smart_airbus_data_lr/images/val \
  --max_images 5 \
  --device cuda

"""
# 프로젝트 루트 추가 (src import용)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

#from src.models.pipelines.arch4_adaptive import Arch4Adaptive
from src.models.pipelines.arch4_roi_awareNMS import Arch4RoiAwareNMS

def load_yaml_dict(p: str) -> dict:
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def patch_arch4_config(cfg: dict) -> dict:
    """
    현재 repo의 arch4_adaptive.py는 다음 키를 기대:
      - model.yolo.weights_hr / weights_lr
      - model.yolo.num_classes
      - model.arch4.pass2_conf
    그런데 YAML은 보통:
      - model.yolo.classes
      - model.arch4.high_conf
    라서 alias를 맞춰줌.
    """
    cfg = cfg.copy()
    cfg.setdefault("model", {})
    cfg["model"].setdefault("yolo", {})
    cfg["model"].setdefault("arch4", {})
    cfg["model"].setdefault("sr", {})

    y = cfg["model"]["yolo"]
    a = cfg["model"]["arch4"]

    # classes -> num_classes
    if "num_classes" not in y and "classes" in y:
        y["num_classes"] = y["classes"]

    # high_conf -> pass2_conf
    if "pass2_conf" not in a and "high_conf" in a:
        a["pass2_conf"] = a["high_conf"]

    return cfg


def list_images(images_dir: Path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    out = []
    for e in exts:
        out += list(images_dir.glob(e))
    return sorted(out)


def cv2_to_tensor(img_bgr):
    # BGR -> RGB, [0,1], (1,3,H,W)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float() / 255.0
    return t.unsqueeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch4_config", required=True)
    p.add_argument("--lr_images_dir", required=True)
    p.add_argument("--max_images", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cfg = load_yaml_dict(args.arch4_config)
    cfg = patch_arch4_config(cfg)

    print("[SMOKE] Build Arch4...")
    model = Arch4RoiAwareNMS(cfg)  # 내부에서 device=cfg.device 기본을 씀(대부분 cuda) / Arch4Adaptive(cfg)
    model.eval()

    imgs = list_images(Path(args.lr_images_dir))
    imgs = imgs[: args.max_images]

    print(f"[SMOKE] images: {len(imgs)}")
    for i, ip in enumerate(imgs):
        img = cv2.imread(str(ip))
        if img is None:
            print(f"  - skip (read fail): {ip}")
            continue
        lr = cv2_to_tensor(img)
        lr = lr.to(device)

        with torch.no_grad():
            out = model.forward(lr, debug=args.debug)
            def _flatten_det(x):
                out = []
                if isinstance(x, dict) and "boxes" in x:
                    return [x]
                if isinstance(x, list):
                    for a in x:
                        out += _flatten_det(a)
                return out

            def _summarize(raw, name):
                dets = _flatten_det(raw)
                n = 0
                mx = 0.0
                for d in dets:
                    if "boxes" in d and hasattr(d["boxes"], "shape"):
                        n += int(d["boxes"].shape[0])
                        if int(d["boxes"].shape[0]) > 0 and "scores" in d:
                            mx = max(mx, float(d["scores"].max().detach().cpu().item()))
                print(f"[DEBUG] {name}: total_boxes={n}, max_conf={mx:.4f}")

            if args.debug:
                dbg = out.get("debug_info", None)
                dbg0 = dbg[0] if isinstance(dbg, list) else dbg
                _summarize(dbg0.get("pass1_raw", None), "pass1_raw")
                _summarize(dbg0.get("pass2_raw", None), "pass2_raw")
                print("[DEBUG] num_crops_lr:", len(dbg0.get("crops_lr", [])))
                print("[DEBUG] num_crops_sr:", len(dbg0.get("crops_sr", [])))
                dbg = out["debug_info"][0] if isinstance(out["debug_info"], list) else out["debug_info"]
                crops_sr = dbg.get("crops_sr", [])
                if len(crops_sr) > 0 and torch.is_tensor(crops_sr[0]):
                    crop = crops_sr[0].detach().float().cpu()  # (3,H,W) expected
                    if crop.ndim == 3:
                        img = (crop.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                        cv2.imwrite("/tmp/arch4_crop_sr0.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        print("[DEBUG] saved /tmp/arch4_crop_sr0.png")
        dets = out["detections"]
        det0 = dets[0]
        n = int(det0["boxes"].shape[0])
        mx = float(det0["scores"].max().item()) if n > 0 else 0.0
        print(f"  [{i}] {ip.name} | det={n} | max_conf={mx:.4f}")

    print("\n[SMOKE] DONE")


if __name__ == "__main__":
    main()