#!/usr/bin/env python3
"""
Self-contained unit test for the patched Arch2 selective-skip pipeline.

Why this exists:
- The pipeline file imports many repo-local modules.
- This test injects lightweight stub modules into sys.modules so the
  pipeline can be imported even outside the full repo.
- It verifies the behaviors that matter for the selective-skip patch.

Run:
  python test_arch2_selective_skip_unit.py \
    --arch2_py /path/to/arch2_softgate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unit test selective-skip Arch2 pipeline.")
    parser.add_argument(
        "--arch2_py",
        type=str,
        default=str(Path(__file__).resolve().parent / "arch2_softgate_selective_skip.py"),
        help="Path to the patched arch2_softgate.py file.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stub repo modules
# ---------------------------------------------------------------------------


def install_stub_modules() -> None:
    def ensure_module(name: str) -> types.ModuleType:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        return module

    # Package chain
    for name in [
        "src",
        "src.models",
        "src.models.pipelines",
        "src.models.sr_models",
        "src.models.detectors",
        "src.models.gates",
        "src.losses",
    ]:
        ensure_module(name)

    base_pipeline_mod = ensure_module("src.models.pipelines.base_pipeline")
    rfdn_mod = ensure_module("src.models.sr_models.rfdn")
    yolo_mod = ensure_module("src.models.detectors.yolo_wrapper")
    gate_mod = ensure_module("src.models.gates.soft_gate")
    det_loss_mod = ensure_module("src.losses.detection_loss")
    sr_loss_mod = ensure_module("src.losses.sr_loss")

    class BasePipeline(nn.Module):
        def __init__(self, config: Any):
            super().__init__()
            self.config = config
            device = getattr(config, "device", "cpu")
            self.device = torch.device(device)
            training_config = getattr(config, "training", config)
            self._sr_weight = getattr(training_config, "sr_weight", 0.0)
            self._det_weight = getattr(training_config, "det_weight", 1.0)

        def forward(self, lr_image, **kwargs):  # pragma: no cover - abstract stub
            raise NotImplementedError

        def compute_loss(self, outputs, targets, **kwargs):  # pragma: no cover - abstract stub
            raise NotImplementedError

    class FakeRFDN(nn.Module):
        def __init__(self, in_channels=3, out_channels=3, nf=50, num_modules=4, upscale=4, **kwargs):
            super().__init__()
            self.upscale = upscale
            self.calls = []
            self.bias = nn.Parameter(torch.tensor(64.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls.append(tuple(x.shape))
            out = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
            return out + self.bias

        def load_pretrained(self, path: str, strict: bool = True) -> None:
            self.loaded = path

    class FakeDetectionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            return x

    class FakeYOLOWrapper:
        def __init__(self, model_path: str, num_classes: int, device: Any, verbose: bool = False):
            self.model_path = model_path
            self.num_classes = num_classes
            self.device = device
            self.verbose = verbose
            self.detection_model = FakeDetectionModel()
            self.predict_calls = []
            self.forward_calls = []
            self.mode = "eval"

        def __call__(self, x: torch.Tensor):
            self.forward_calls.append(tuple(x.shape))
            return {"preds": x}

        def train(self):
            self.mode = "train"

        def eval(self):
            self.mode = "eval"

        def predict(self, x: torch.Tensor, **kwargs):
            self.predict_calls.append({"shape": tuple(x.shape), **kwargs})
            return [{"shape": tuple(x.shape), **kwargs}]

        def freeze(self):
            return None

        def unfreeze(self):
            return None

        def set_bn_eval(self):
            return None

    class FakeGate(nn.Module):
        def __init__(self, in_channels=3, base_channels=32, num_layers=4):
            super().__init__()
            self.logit = nn.Parameter(torch.tensor(0.0))
            self.next_output = None

        def set_output(self, gate_values: torch.Tensor) -> None:
            self.next_output = gate_values.clone().detach().float()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            if self.next_output is not None:
                out = self.next_output.to(x.device)
                if out.ndim == 1:
                    out = out.view(-1, 1)
                if out.shape[0] != batch:
                    raise ValueError(f"Expected gate batch {batch}, got {out.shape[0]}")
                return out
            return torch.full((batch, 1), 0.5, device=x.device, dtype=x.dtype)

    class FakeDetectionLoss:
        def __init__(self, detection_model: Any):
            self.detection_model = detection_model

        def __call__(self, preds, targets, image):
            z = image.sum() * 0.0
            return {
                "total": z,
                "box_loss": z,
                "cls_loss": z,
                "dfl_loss": z,
            }

    class FakeSRLoss:
        def __init__(self, *args, **kwargs):
            self.calls = []

        def __call__(self, sr_image: torch.Tensor, hr_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
            self.calls.append((tuple(sr_image.shape), tuple(hr_gt.shape)))
            loss = (sr_image - hr_gt).abs().mean()
            return {"total": loss}

    base_pipeline_mod.BasePipeline = BasePipeline
    rfdn_mod.RFDN = FakeRFDN
    yolo_mod.YOLOWrapper = FakeYOLOWrapper
    gate_mod.LightweightGateV1 = FakeGate
    det_loss_mod.DetectionLoss = FakeDetectionLoss
    sr_loss_mod.SRLoss = FakeSRLoss


def load_arch2_class(arch2_py: Path):
    install_stub_modules()
    spec = importlib.util.spec_from_file_location("arch2_under_test", arch2_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {arch2_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Arch2SoftGate


def make_config(
    *,
    selective: bool = True,
    threshold: float = 0.5,
    blend: bool = False,
    sr_weight: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        device="cpu",
        data=SimpleNamespace(upscale_factor=2),
        training=SimpleNamespace(sr_weight=sr_weight, det_weight=1.0),
        model=SimpleNamespace(
            sr_type="rfdn",
            yolo=SimpleNamespace(weights_path="dummy_yolo.pt", num_classes=1),
            gate=SimpleNamespace(
                in_channels=3,
                base_channels=4,
                num_layers=2,
                weights_path=None,
                use_selective_inference=selective,
                inference_threshold=threshold,
                blend_selected_inference=blend,
            ),
            rfdn={
                "nf": 8,
                "num_modules": 1,
                "pretrain_path": None,
            },
        ),
    )


class TestArch2SelectiveSkip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = parse_args()
        cls.arch2_py = Path(args.arch2_py).resolve()
        cls.Arch2SoftGate = load_arch2_class(cls.arch2_py)

    def build_model(self, **kwargs):
        cfg = make_config(**kwargs)
        model = self.Arch2SoftGate(cfg)
        model.eval()
        return model

    def test_selective_inference_runs_sr_only_for_selected_samples(self):
        model = self.build_model(selective=True, threshold=0.5, blend=False)
        model.gate_network.set_output(torch.tensor([0.90, 0.10, 0.80]))

        x = torch.rand(3, 3, 4, 4)
        out = model.inference(x, conf_threshold=0.33, iou_threshold=0.66)

        self.assertEqual(len(model.sr_model.calls), 1, "SR model should run once on selected subset.")
        self.assertEqual(model.sr_model.calls[0][0], 2, "Only two samples should be sent through SR.")

        expected_mask = torch.tensor([True, False, True])
        self.assertTrue(torch.equal(out["sr_selected_mask"].cpu(), expected_mask))
        self.assertAlmostEqual(out["sr_applied_ratio"], 2.0 / 3.0, places=6)

        predict_call = model.detector.predict_calls[-1]
        self.assertAlmostEqual(predict_call["conf"], 0.33, places=6)
        self.assertAlmostEqual(predict_call["iou"], 0.66, places=6)

        bypass = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        hr = out["hr_image"].detach()

        self.assertTrue(torch.allclose(hr[1], bypass[1], atol=1e-6), "Unselected sample must stay on bypass path.")
        self.assertFalse(torch.allclose(hr[0], bypass[0], atol=1e-6), "Selected sample should differ from bypass path.")
        self.assertFalse(torch.allclose(hr[2], bypass[2], atol=1e-6), "Selected sample should differ from bypass path.")

    def test_fallback_mode_runs_full_batch_sr(self):
        model = self.build_model(selective=False)
        model.gate_network.set_output(torch.tensor([0.10, 0.20, 0.30]))

        x = torch.rand(3, 3, 4, 4)
        out = model.inference(x)

        self.assertEqual(len(model.sr_model.calls), 1)
        self.assertEqual(model.sr_model.calls[0][0], 3, "Fallback mode should run SR on the full batch.")
        self.assertAlmostEqual(out["sr_applied_ratio"], 1.0, places=6)

    def test_compute_loss_uses_lr_image_fallback_for_sr_loss(self):
        model = self.build_model(selective=True, sr_weight=1.0)
        model.gate_network.set_output(torch.tensor([0.90]))

        lr = torch.rand(1, 3, 5, 5)
        out = {
            "hr_image": torch.rand(1, 3, 10, 10),
            "gate": torch.tensor([[0.9]]),
            "lr_image": lr,
        }
        hr_gt = torch.rand(1, 3, 10, 10)

        # Clear call history to isolate compute_loss.
        model.sr_model.calls.clear()
        loss_dict = model.compute_loss(out, targets=None, hr_gt=hr_gt)

        self.assertIn("total", loss_dict)
        self.assertEqual(len(model.sr_model.calls), 1)
        self.assertEqual(model.sr_model.calls[0], tuple(lr.shape), "compute_loss should SR the LR input, not HR image.")
        self.assertEqual(model.sr_loss_fn.calls[-1][0], (1, 3, 10, 10))
        self.assertEqual(model.sr_loss_fn.calls[-1][1], (1, 3, 10, 10))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
