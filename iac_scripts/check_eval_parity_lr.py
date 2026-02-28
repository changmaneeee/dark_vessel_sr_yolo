#!/usr/bin/env python3
"""
LR-only parity check

목표:
1) 같은 LR 이미지 200장(혹은 지정한 수)의 subset을 고정으로 만든다.
2) 같은 YOLO weights / 같은 imgsz / 같은 conf / 같은 iou로
   - 공식 yolo.val()
   - custom evaluator
   를 각각 실행한다.
3) 두 결과 차이(delta)를 출력/저장한다.

이게 거의 같아야:
"우리가 Arch4에서 쓰는 evaluator logic도 공식 val과 같은 잣대다"
라고 말할 수 있다.
"""

import json
import time
import argparse
from pathlib import Path

import cv2
import yaml
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class


# =========================================================
# 1. Basic helpers
# =========================================================

def safe_float(x):
    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return None


def list_images(images_dir: Path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for e in exts:
        files += list(images_dir.glob(e))
    return sorted(files)


def parse_ultralytics_data_yaml(data_yaml: str):
    """
    예:
      path: /home/changmin/smart_airbus_data_lr
      val: images/val
      names: {0: ship}
    """
    ypath = Path(data_yaml).resolve()
    with open(ypath, "r") as f:
        d = yaml.safe_load(f) or {}

    root = Path(d.get("path", ypath.parent))
    if not root.is_absolute():
        root = (ypath.parent / root).resolve()

    val_rel = d.get("val", "images/val")
    if isinstance(val_rel, list):
        val_rel = val_rel[0]

    val_path = Path(val_rel)
    images_dir = val_path if val_path.is_absolute() else (root / val_path).resolve()

    # labels/val 추론
    if len(val_path.parts) >= 2 and val_path.parts[0] == "images":
        labels_rel = Path("labels") / Path(*val_path.parts[1:])
    else:
        labels_rel = Path("labels") / val_path.name
    labels_dir = labels_rel if labels_rel.is_absolute() else (root / labels_rel).resolve()

    names = d.get("names", {0: "ship"})
    return images_dir, labels_dir, names


def write_subset_files(img_paths, names, out_dir: Path):
    """
    official yolo.val()도 같은 200장을 보게 하기 위해
    txt 리스트 + yaml을 자동 생성
    """
    subset_txt = out_dir / "subset_lr_images.txt"
    with open(subset_txt, "w") as f:
        for p in img_paths:
            f.write(str(p.resolve()) + "\n")

    subset_yaml = out_dir / "subset_lr_data.yaml"
    subset_cfg = {
        "path": "/",
        "train": str(subset_txt.resolve()),
        "val": str(subset_txt.resolve()),
        "names": names,
    }
    with open(subset_yaml, "w") as f:
        yaml.safe_dump(subset_cfg, f, sort_keys=False)

    return subset_txt, subset_yaml


# =========================================================
# 2. GT label helpers
# =========================================================

def load_yolo_label_file(label_path: Path):
    """
    YOLO txt:
      cls x y w h  (normalized)
    return: Nx5 float32
    """
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)

    text = label_path.read_text().strip()
    if not text:
        return np.zeros((0, 5), dtype=np.float32)

    rows = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        rows.append([float(v) for v in parts])

    if not rows:
        return np.zeros((0, 5), dtype=np.float32)

    return np.asarray(rows, dtype=np.float32)


def xywhn_to_xyxy_pixels(labels_xywhn: np.ndarray, w: int, h: int):
    """
    labels_xywhn: Nx5 (cls, x, y, w, h), normalized
    return: Nx5 (cls, x1, y1, x2, y2), pixels
    """
    if labels_xywhn.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float32)

    cls = labels_xywhn[:, 0:1]
    x = labels_xywhn[:, 1] * w
    y = labels_xywhn[:, 2] * h
    bw = labels_xywhn[:, 3] * w
    bh = labels_xywhn[:, 4] * h

    x1 = x - bw / 2
    y1 = y - bh / 2
    x2 = x + bw / 2
    y2 = y + bh / 2

    out = np.concatenate([cls, x1[:, None], y1[:, None], x2[:, None], y2[:, None]], axis=1)
    return out.astype(np.float32)


# =========================================================
# 3. Metric core (custom evaluator)
# =========================================================

def box_iou_torch(box1, box2):
    """
    box1: (M,4), box2: (N,4), xyxy
    return: (M,N)
    """
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.shape[0], box2.shape[0]), device=box1.device)

    x1, y1, x2, y2 = box1[:, 0:1], box1[:, 1:2], box1[:, 2:3], box1[:, 3:4]
    X1, Y1, X2, Y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

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
    detections: (N,6) = xyxy, conf, cls
    labels:     (M,5) = cls, xyxy
    return: correct (N, len(iouv))
    """
    correct = torch.zeros((detections.shape[0], iouv.numel()), dtype=torch.bool, device=detections.device)

    if labels.shape[0] == 0 or detections.shape[0] == 0:
        return correct

    iou = box_iou_torch(labels[:, 1:5], detections[:, 0:4])  # (M,N)
    correct_class = labels[:, 0:1] == detections[:, 5]       # (M,N)

    for i, thr in enumerate(iouv):
        x = torch.where((iou >= thr) & correct_class)
        if x[0].numel() == 0:
            continue

        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]].unsqueeze(1)), 1).detach().cpu().numpy()

        if matches.shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]                    # IoU 큰 순
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]   # pred unique
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]   # gt unique

        correct[matches[:, 1].astype(int), i] = True

    return correct


def metrics_from_stats(stats, names, save_dir: Path):
    if len(stats) == 0:
        return {
            "metrics/precision(B)": 0.0,
            "metrics/recall(B)": 0.0,
            "metrics/mAP50(B)": 0.0,
            "metrics/mAP50-95(B)": 0.0,
            "direct/tp50": 0,
            "direct/fp50": 0,
            "direct/fn50": 0,
            "direct/precision50": 0.0,
            "direct/recall50": 0.0,
        }

    correct, conf, pred_cls, target_cls = [torch.cat(x, 0).numpy() for x in zip(*stats)]

    ap_results = ap_per_class(
        correct,
        conf,
        pred_cls,
        target_cls,
        plot=False,
        save_dir=save_dir,
        names=names,
    )

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

    # operating-point direct metrics at IoU=0.5
    tp50 = int(correct[:, 0].sum()) if correct.size else 0
    num_pred = int(len(conf))
    num_gt = int(len(target_cls))
    fp50 = max(num_pred - tp50, 0)
    fn50 = max(num_gt - tp50, 0)

    precision50_direct = tp50 / num_pred if num_pred > 0 else 0.0
    recall50_direct = tp50 / num_gt if num_gt > 0 else 0.0

    return {
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


# =========================================================
# 4. Official val
# =========================================================

def run_official_val(yolo_weights, subset_yaml, args):
    model = YOLO(yolo_weights)
    t0 = time.perf_counter()

    metrics = model.val(
        data=str(subset_yaml),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        plots=False,
        save_json=False,
        verbose=False,
    )

    wall = time.perf_counter() - t0
    rd = metrics.results_dict

    out = {
        "metrics/precision(B)": safe_float(rd.get("metrics/precision(B)", 0.0)),
        "metrics/recall(B)": safe_float(rd.get("metrics/recall(B)", 0.0)),
        "metrics/mAP50(B)": safe_float(rd.get("metrics/mAP50(B)", 0.0)),
        "metrics/mAP50-95(B)": safe_float(rd.get("metrics/mAP50-95(B)", 0.0)),
        "wall_time_sec": wall,
    }
    return out


# =========================================================
# 5. Custom eval on same subset
# =========================================================

def run_custom_eval(yolo_weights, img_paths, labels_dir, names, args, out_dir: Path):
    model = YOLO(yolo_weights)

    score_device = torch.device("cuda" if (torch.cuda.is_available() and str(args.device) != "cpu") else "cpu")
    iouv = torch.linspace(0.5, 0.95, 10, device=score_device)

    stats = []
    t0 = time.perf_counter()

    for idx, ip in enumerate(img_paths, 1):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]

        # GT
        lb = labels_dir / f"{ip.stem}.txt"
        gt_xywhn = load_yolo_label_file(lb)
        gt_xyxy = xywhn_to_xyxy_pixels(gt_xywhn, w, h)
        gt = torch.from_numpy(gt_xyxy).to(score_device).float()  # (M,5)

        # Pred
        res = model.predict(
            source=str(ip),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            pred = torch.zeros((0, 6), device=score_device)
        else:
            boxes = res.boxes.xyxy.to(score_device).float()
            confs = res.boxes.conf.to(score_device).float()
            cls = res.boxes.cls.to(score_device).float()
            pred = torch.cat([boxes, confs.unsqueeze(1), cls.unsqueeze(1)], dim=1)

        correct = process_batch(pred, gt, iouv)

        stats.append((
            correct.detach().cpu(),
            pred[:, 4].detach().cpu(),
            pred[:, 5].detach().cpu(),
            gt[:, 0].detach().cpu(),
        ))

        if idx % 50 == 0 or idx == len(img_paths):
            print(f"[CUSTOM] processed {idx}/{len(img_paths)}")

    wall = time.perf_counter() - t0
    result = metrics_from_stats(stats, names, out_dir)
    result["wall_time_sec"] = wall
    return result


# =========================================================
# 6. Main
# =========================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--yolo_weights", required=True)
    p.add_argument("--lr_data_yaml", required=True)

    p.add_argument("--max_images", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--max_det", type=int, default=300)

    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)

    p.add_argument("--out_dir", default="iac_runs/parity_lr")

    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir, labels_dir, names = parse_ultralytics_data_yaml(args.lr_data_yaml)
    all_imgs = list_images(images_dir)
    img_paths = all_imgs[: args.max_images]

    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in {images_dir}")

    subset_txt, subset_yaml = write_subset_files(img_paths, names, out_dir)

    print("\n[PARITY] subset prepared")
    print("  images_dir :", images_dir)
    print("  labels_dir :", labels_dir)
    print("  subset_txt :", subset_txt)
    print("  subset_yaml:", subset_yaml)
    print("  num_images :", len(img_paths))

    print("\n[PARITY] Running official yolo.val() ...")
    official = run_official_val(args.yolo_weights, subset_yaml, args)

    print("\n[PARITY] Running custom evaluator ...")
    custom = run_custom_eval(args.yolo_weights, img_paths, labels_dir, names, args, out_dir)

    delta = {
        "metrics/precision(B)": custom["metrics/precision(B)"] - official["metrics/precision(B)"],
        "metrics/recall(B)": custom["metrics/recall(B)"] - official["metrics/recall(B)"],
        "metrics/mAP50(B)": custom["metrics/mAP50(B)"] - official["metrics/mAP50(B)"],
        "metrics/mAP50-95(B)": custom["metrics/mAP50-95(B)"] - official["metrics/mAP50-95(B)"],
    }

    summary = {
        "meta": {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "yolo_weights": args.yolo_weights,
            "lr_data_yaml": args.lr_data_yaml,
            "subset_txt": str(subset_txt),
            "subset_yaml": str(subset_yaml),
            "num_images": len(img_paths),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "device": args.device,
            "batch": args.batch,
            "workers": args.workers,
        },
        "official_val": official,
        "custom_eval": custom,
        "delta_custom_minus_official": delta,
    }

    out_json = out_dir / "parity_lr_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== PARITY DONE ===")
    print(f"Saved: {out_json}")
    print("\n[OFFICIAL]")
    print(official)
    print("\n[CUSTOM]")
    print(custom)
    print("\n[DELTA custom - official]")
    print(delta)

    print("\n판단 기준(권장):")
    print("  |Δ mAP50-95| <= 0.005")
    print("  |Δ mAP50|    <= 0.005")
    print("  |Δ Precision|<= 0.01")
    print("  |Δ Recall|   <= 0.01")


if __name__ == "__main__":
    main()