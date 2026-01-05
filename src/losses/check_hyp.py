"""hyp 내용 확인"""
import torch
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

yolo = YOLO("yolov8n.pt")

loss_fn = v8DetectionLoss(yolo.model)

print("=" * 60)
print("hyp 내용 확인")
print("=" * 60)

print(f"\ntype(loss_fn.hyp): {type(loss_fn.hyp)}")

if isinstance(loss_fn.hyp, dict):
    print(f"\nhyp keys: {list(loss_fn.hyp.keys())}")
    print(f"\nhyp 전체 내용:")
    for k, v in loss_fn.hyp.items():
        print(f"  {k}: {v}")
else:
    print(f"\nhyp: {loss_fn.hyp}")
    if hasattr(loss_fn.hyp, '__dict__'):
        print(f"hyp.__dict__: {loss_fn.hyp.__dict__}")

# model.args도 확인
print("\n" + "=" * 60)
print("model.args 확인")
print("=" * 60)

if hasattr(yolo.model, 'args'):
    args = yolo.model.args
    print(f"type(args): {type(args)}")
    if isinstance(args, dict):
        print(f"args keys: {list(args.keys())}")
        # box, cls, dfl 있는지
        for key in ['box', 'cls', 'dfl']:
            if key in args:
                print(f"  args['{key}']: {args[key]}")
    else:
        for key in ['box', 'cls', 'dfl']:
            if hasattr(args, key):
                print(f"  args.{key}: {getattr(args, key)}")