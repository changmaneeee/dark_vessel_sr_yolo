#!/usr/bin/env python3
"""
Arch4 evaluation (Ultralytics-style metrics)

핵심 아이디어:
- Arch4가 만들어낸 최종 detections(박스/score/class)를 이용해서
- Ultralytics가 yolo.val에서 쓰는 mAP 계산 방식(ap_per_class + process_batch 매칭)을 동일하게 적용한다.

결과는 yolo.val과 동일한 정의의:
- metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
를 뽑아준다.
"""

import sys
import time
import json
from pathlib import Path
import argparse
import yaml
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pipelines.arch4_adaptive import Arch4Adaptive


# -------------------------
# YAML / Config helpers
# -------------------------
def load_yaml_dict(p: str) -> dict:
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def patch_arch4_config(cfg: dict) -> dict:
    cfg = cfg.copy()
    cfg.setdefault("model", {})
    cfg["model"].setdefault("yolo", {})
    cfg["model"].setdefault("arch4", {})
    cfg["model"].setdefault("sr", {})

    y = cfg["model"]["yolo"]
    a = cfg["model"]["arch4"]

    if "num_classes" not in y and "classes" in y:
        y["num_classes"] = y["classes"]
    if "pass2_conf" not in a and "high_conf" in a:
        a["pass2_conf"] = a["high_conf"]

    return cfg


def parse_ultralytics_data_yaml(data_yaml: str):
    """
    Ultralytics data.yaml 예:
      path: /home/xxx/dataset
      val: images/val
      names: {0: 'ship'} or ['ship']
    """
    d = load_yaml_dict(data_yaml)
    root = Path(d.get("path", Path(data_yaml).parent)).expanduser()
    val_rel = d.get("val", "images/val")
    images_dir = (root / val_rel).resolve()

    # labels_dir 유추: images/val -> labels/val
    val_rel_p = Path(val_rel)
    if len(val_rel_p.parts) >= 2 and val_rel_p.parts[0] == "images":
        labels_rel = Path("labels") / Path(*val_rel_p.parts[1:])
    else:
        labels_rel = Path("labels") / val_rel_p.name
    labels_dir = (root / labels_rel).resolve()

    names = d.get("names", None)
    return images_dir, labels_dir, names


# -------------------------
# IO helpers
# -------------------------
def list_images(images_dir: Path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    out = []
    for e in exts:
        out += list(images_dir.glob(e))
    return sorted(out)


def cv2_to_tensor(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float() / 255.0
    return t.unsqueeze(0)


def load_yolo_label_file(label_path: Path):
    """
    YOLO label: cls x y w h (normalized)
    Return: (cls, x, y, w, h) float32 array
    """
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    lines = label_path.read_text().strip().splitlines()
    if len(lines) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for ln in lines:
        p = ln.strip().split()
        if len(p) != 5:
            continue
        rows.append([float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def xywhn_to_xyxy_pixels(labels_xywhn: np.ndarray, W: int, H: int):
    """
    labels_xywhn: Nx5 (cls, x, y, w, h) normalized
    return: Nx5 (cls, x1, y1, x2, y2) pixel
    """
    if labels_xywhn.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float32)

    cls = labels_xywhn[:, 0:1]
    x = labels_xywhn[:, 1] * W
    y = labels_xywhn[:, 2] * H
    w = labels_xywhn[:, 3] * W
    h = labels_xywhn[:, 4] * H

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    out = np.concatenate([cls, x1[:, None], y1[:, None], x2[:, None], y2[:, None]], axis=1)
    return out.astype(np.float32)


# -------------------------
# Ultralytics-style metric core
# -------------------------
def box_iou_torch(box1, box2):
    """
    box1: (M,4), box2: (N,4) in xyxy
    return iou: (M,N)
    """
    # intersection
    (x1, y1, x2, y2) = box1[:, 0:1], box1[:, 1:2], box1[:, 2:3], box1[:, 3:4]
    (X1, Y1, X2, Y2) = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = torch.maximum(x1, X1)
    inter_y1 = torch.maximum(y1, Y1)
    inter_x2 = torch.minimum(x2, X2)
    inter_y2 = torch.minimum(y2, Y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter = inter_w * inter_h

    area1 = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area2 = torch.clamp(X2 - X1, min=0) * torch.clamp(Y2 - Y1, min=0)
    union = area1 + area2 - inter + 1e-9
    return inter / union


def process_batch(detections, labels, iouv):
    """
    detections: (N,6) xyxy, conf, cls
    labels: (M,5) cls, xyxy
    return correct: (N, len(iouv)) boolean
    """
    correct = torch.zeros((detections.shape[0], iouv.numel()), dtype=torch.bool, device=detections.device)
    if labels.shape[0] == 0 or detections.shape[0] == 0:
        return correct

    iou = box_iou_torch(labels[:, 1:5], detections[:, 0:4])
    correct_class = labels[:, 0:1] == detections[:, 5]  # (M,N)

    for i, thr in enumerate(iouv):
        x = torch.where((iou >= thr) & correct_class)
        if x[0].numel() == 0:
            continue

        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]].unsqueeze(1)), 1).detach().cpu().numpy()
        if matches.shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]  # sort by IoU desc
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # unique det
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # unique label

        correct[matches[:, 1].astype(int), i] = True

    return correct


def main():
    p = argparse.ArgumentParser()

    # model config
    p.add_argument("--arch4_config", required=True)

    # dataset yamls
    p.add_argument("--hr_data_yaml", required=True, help="HR data.yaml (for GT size/labels)")
    p.add_argument("--lr_data_yaml", required=True, help="LR data.yaml (for input images list)")

    # eval options
    p.add_argument("--eval_space", choices=["hr", "lr"], default="hr",
                   help="hr: pred boxes are scaled by upscale_factor and evaluated in HR pixel space")
    p.add_argument("--max_images", type=int, default=0, help="0=all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--out_json", default="iac_runs/arch4_eval/arch4_eval_full_summary.json")

    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ultralytics AP function (yolo.val과 동일 계열)
    try:
        from ultralytics.utils.metrics import ap_per_class
    except Exception as e:
        raise ImportError("Need ultralytics installed (pip install ultralytics).") from e

    # dataset paths
    hr_images_dir, hr_labels_dir, names = parse_ultralytics_data_yaml(args.hr_data_yaml)
    lr_images_dir, lr_labels_dir, _ = parse_ultralytics_data_yaml(args.lr_data_yaml)

    img_paths = list_images(lr_images_dir)
    if args.max_images and args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    # load arch4
    cfg = patch_arch4_config(load_yaml_dict(args.arch4_config))
    upscale = int(cfg.get("data", {}).get("upscale_factor", 4))

    print("\n[Arch4 Eval]")
    print(f"  device     : {device}")
    print(f"  eval_space : {args.eval_space} (upscale={upscale})")
    print(f"  images     : {len(img_paths)}")
    print(f"  HR labels  : {hr_labels_dir}")
    print(f"  LR images  : {lr_images_dir}")

    model = Arch4Adaptive(cfg)
    model.eval()

    # IoU thresholds (COCO style): 0.50:0.95 step 0.05
    iouv = torch.linspace(0.5, 0.95, 10, device=device)

    stats = []  # list of (correct, conf, pred_cls, target_cls)
    t0 = time.perf_counter()

    bs = max(1, int(args.batch))
    for start in range(0, len(img_paths), bs):
        batch_paths = img_paths[start:start+bs]
        lr_batch = []
        shapes_hr = []
        label_batches = []

        # ---- load batch ----
        for ip in batch_paths:
            img = cv2.imread(str(ip))
            if img is None:
                continue
            lr_t = cv2_to_tensor(img)
            lr_batch.append(lr_t)

            # matching label file (same stem)
            lb = hr_labels_dir / f"{ip.stem}.txt"
            labels_xywhn = load_yolo_label_file(lb)

            if args.eval_space == "hr":
                # HR image size 기준으로 GT를 픽셀로 변환
                hr_img_path = hr_images_dir / ip.name
                hr_img = cv2.imread(str(hr_img_path))
                if hr_img is None:
                    # fallback: HR 이미지가 없으면 LR*upscale로 가정
                    H_lr, W_lr = img.shape[0], img.shape[1]
                    H_hr, W_hr = H_lr * upscale, W_lr * upscale
                else:
                    H_hr, W_hr = hr_img.shape[0], hr_img.shape[1]
                gt_xyxy = xywhn_to_xyxy_pixels(labels_xywhn, W_hr, H_hr)
                shapes_hr.append((H_hr, W_hr))
            else:
                # LR image size 기준
                H_lr, W_lr = img.shape[0], img.shape[1]
                gt_xyxy = xywhn_to_xyxy_pixels(labels_xywhn, W_lr, H_lr)
                shapes_hr.append((H_lr, W_lr))

            label_batches.append(gt_xyxy)

        if len(lr_batch) == 0:
            continue

        lr_tensor = torch.cat(lr_batch, dim=0).to(device)

        # ---- inference ----
        with torch.no_grad():
            out = model.forward(lr_tensor, debug=False)

        dets = out["detections"]

        # ---- per-image metric update ----
        for bi in range(len(dets)):
            det = dets[bi]
            boxes = det["boxes"].detach().to(device).float()
            scores = det["scores"].detach().to(device).float()
            cls = det["classes"].detach().to(device).float()

            if boxes.numel() == 0:
                pred = torch.zeros((0, 6), device=device)
            else:
                pred = torch.cat([boxes, scores.unsqueeze(1), cls.unsqueeze(1)], dim=1)  # (N,6)

            # Arch4 output boxes는 기본적으로 LR 좌표계라서 HR 평가 시 upscale 해줌
            H, W = shapes_hr[bi]
            if args.eval_space == "hr" and pred.shape[0] > 0:
                pred[:, 0:4] *= float(upscale)

            # clamp to image bounds
            if pred.shape[0] > 0:
                pred[:, 0].clamp_(0, W - 1)
                pred[:, 2].clamp_(0, W - 1)
                pred[:, 1].clamp_(0, H - 1)
                pred[:, 3].clamp_(0, H - 1)

            gt = torch.from_numpy(label_batches[bi]).to(device).float()  # (M,5) cls,xyxy
            correct = process_batch(pred, gt, iouv)

            stats.append((
                correct.detach().cpu(),
                pred[:, 4].detach().cpu(),
                pred[:, 5].detach().cpu(),
                gt[:, 0].detach().cpu(),
            ))

        if (start // bs + 1) % 50 == 0:
            print(f"  processed {min(start+bs, len(img_paths))}/{len(img_paths)}")

    # ---- aggregate ----
    if len(stats) == 0:
        raise RuntimeError("No stats collected. Check data paths or model outputs.")

    correct, conf, pred_cls, target_cls = [torch.cat(x, 0).numpy() for x in zip(*stats)]

    ap_results = ap_per_class(
    correct,
    conf,
    pred_cls,
    target_cls,
    plot=False,
    save_dir=None,
    names=names,
    )

    # Ultralytics 8.3.x:
    # (tp, fp, p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, x, prec_values)
    # 예전 버전 호환까지 생각해서 분기
    if len(ap_results) >= 7:
        _, _, p_, r_, f1, ap, ap_class, *_ = ap_results
    else:
        p_, r_, ap, f1, ap_class = ap_results

    p_ = np.atleast_1d(np.asarray(p_, dtype=np.float32))
    r_ = np.atleast_1d(np.asarray(r_, dtype=np.float32))
    ap = np.asarray(ap, dtype=np.float32)

    mp = float(p_.mean()) if p_.size else 0.0
    mr = float(r_.mean()) if r_.size else 0.0
    map50 = float(ap[:, 0].mean()) if ap.size else 0.0
    map5095 = float(ap.mean()) if ap.size else 0.0

    wall = time.perf_counter() - t0
    avg_ms = (wall / len(img_paths)) * 1000.0

    tp50 = int(correct[:, 0].sum()) if correct.size else 0
    num_pred = int(len(conf))
    num_gt = int(len(target_cls))

    fp50 = max(num_pred - tp50, 0)
    fn50 = max(num_gt - tp50, 0)

    precision50_direct = tp50 / num_pred if num_pred > 0 else 0.0
    recall50_direct = tp50 / num_gt if num_gt > 0 else 0.0


    summary = {
        "meta": {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "arch": "arch4_adaptive",
            "arch4_config": args.arch4_config,
            "hr_data_yaml": args.hr_data_yaml,
            "lr_data_yaml": args.lr_data_yaml,
            "eval_space": args.eval_space,
            "upscale_factor": upscale,
            "device": device,
            "batch": bs,
            "num_images": len(img_paths),
            "avg_ms_per_image": avg_ms,
        },
        "runs": [
            {
                "tag": "ARCH4",
                "results_dict": {
                    "metrics/precision(B)": mp,
                    "metrics/recall(B)": mr,
                    "metrics/mAP50(B)": map50,
                    "metrics/mAP50-95(B)": map5095,
                    "direct/tp50": tp50,
                    "direct/fp50": fp50,
                    "direct/fn50": fn50,
                    "direct/precision50": precision50_direct,
                    "direct/recall50": recall50_direct,
                }
                }
            
        ]
    }

    out_json = Path(args.out_json).expanduser()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== ARCH4 EVAL DONE ===")
    print(f"Saved: {out_json}")
    print(f"Precision: {mp:.4f} | Recall: {mr:.4f} | mAP50: {map50:.4f} | mAP50-95: {map5095:.4f}")
    print(f"Direct @IoU0.5 -> TP: {tp50} | FP: {fp50} | FN: {fn50} | "
        f"P50: {precision50_direct:.4f} | R50: {recall50_direct:.4f}")
    print(f"Avg time/image: {avg_ms:.2f} ms")


if __name__ == "__main__":
    main()