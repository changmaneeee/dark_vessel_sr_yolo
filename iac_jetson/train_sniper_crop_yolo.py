#!/usr/bin/env python3
"""
Fine-tune a ROI-crop-specific Sniper YOLO from yolo_8s_rfdn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def str2bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ROI-crop Sniper YOLO.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--base_weights", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=float, default=3.0)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save_period", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", type=str2bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.base_weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        augment=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        save=True,
        save_period=args.save_period,
        val=True,
        plots=True,
        device=args.device,
        workers=args.workers,
        resume=args.resume,
        amp=args.amp,
    )

    save_dir = Path(results.save_dir)
    summary = {
        "save_dir": str(save_dir),
        "best_weights": str(save_dir / "weights" / "best.pt"),
        "last_weights": str(save_dir / "weights" / "last.pt"),
        "results_csv": str(save_dir / "results.csv"),
    }
    (save_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
