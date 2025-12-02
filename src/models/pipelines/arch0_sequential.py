"""Architecture 0: Sequential Pipeline

Simple 2-stage approach: LR → SR → HR → Detection
Baseline architecture for comparison.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from src.models.pipelines.base_pipeline import BasePipeline


class Arch0Sequential(BasePipeline):
    """Sequential SR-then-Detect Pipeline (Baseline)

    TODO: Implement sequential pipeline
    - Stage 1: SR (LR → HR)
    - Stage 2: Detection (HR → Boxes)
    - Optional: detach gradients between stages
    """

    def __init__(
        self,
        sr_model: nn.Module,
        detector: nn.Module,
        scale_factor: int = 4,
        detach_sr: bool = True,
        device: str = 'cuda'
    ):
        """
        Args:
            sr_model: SR model
            detector: Detection model
            scale_factor: SR scale factor
            detach_sr: Detach SR from detection gradients
            device: Device
        """
        super().__init__(sr_model, detector, scale_factor, device)
        self.detach_sr = detach_sr

    def forward(self, lr_image: torch.Tensor) -> Dict[str, Any]:
        """Forward pass: LR → SR → Detection

        Args:
            lr_image: LR input [B, 3, H, W]

        Returns:
            outputs: {
                'hr_image': SR output,
                'detections': Detection results
            }
        """
        # TODO: Implement sequential forward
        # 1. SR: LR → HR
        # 2. (Optional) Detach gradients
        # 3. Detection: HR → Boxes
        pass

    def get_loss(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        hr_image: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss

        Args:
            outputs: Forward outputs
            targets: Ground truth
            hr_image: GT HR image (for SR loss)

        Returns:
            loss_dict: {
                'total_loss': combined loss,
                'sr_loss': SR reconstruction loss,
                'det_loss': Detection loss
            }
        """
        # TODO: Implement loss calculation
        # - SR loss (if hr_image provided)
        # - Detection loss
        # - Weighted combination
        pass
