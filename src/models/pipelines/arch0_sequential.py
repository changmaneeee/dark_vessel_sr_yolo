"""Architecture 0: Sequential Pipeline

Simple 2-stage approach: LR → SR → HR → Detection
Baseline architecture for comparison.
"""

import torch
import torch.nn as nn
from base_pipeline import BasePipeline
from ..sr_models.rfdn import RFDN
from ..detectors.yolo_wrapper import YOLOWrapper

class Arch0Sequential(BasePipeline):
    """
    [Sequential Pipeline]
    LR Image → SR Model → HR Image(Tensor) → YOLO Detector → Detection Results
    Base architecture for SR-Detection pipeline.
    """

    def __init__(self, config):
        super().__init__(config)

        # 1. SR Model(RFDN) Initialization
        self.sr_model = RFDN(
            in_channels=3,
            out_channels=3,
            nf = config.model.rfdn.nf,
            num_modules=config.model.rfdn.num_modules,
            upscale=config.data.upscale_factor
        )

        # 2. YOLO Detector Initialization
        self.detector = YOLOWrapper(model_path=config.model.yolo.weights_path, task='detect')

        # 3. YOLO Freeze
        # 4. Before this freeze, ensure YOLO is pre-trained on HR images.
        for param in self.detector.parameters():
            param.requires_grad = False
        print("✓ YOLO detector frozen")

    def forward(self, x):
        """
        [inference/testing mode]
        x: Low Resolution Image Tensor [B, 3, H, W]
        """

        # 1. SR Model: LR → HR
        # Result is not Image, but Tensor
        sr_images = self.sr_model(x)

        # 2. Object Detection
        # In-memory Transfer: Tensor → Detection Results
        detection = self.detector(sr_images)

        return sr_images, detection # Return both SR image tensor and detection results(SR Image for evaluation)

