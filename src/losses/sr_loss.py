"""SR Loss Functions

Reconstruction losses for super-resolution training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (smooth L1)

    More robust to outliers than L1/L2.

    TODO: Implement Charbonnier loss
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted HR image [B, C, H, W]
            target: Ground truth HR image [B, C, H, W]

        Returns:
            loss: Charbonnier loss
        """
        # TODO: Implement
        # loss = sqrt((pred - target)^2 + eps^2)
        pass


class PerceptualLoss(nn.Module):
    """Perceptual Loss using VGG features

    Measures high-level perceptual similarity.

    TODO: Implement perceptual loss
    - Load pretrained VGG
    - Extract features at specified layers
    - Compute L1/L2 distance
    """

    def __init__(
        self,
        layers: list = ['relu2_2', 'relu3_3', 'relu4_3'],
        weights: list = [1.0, 1.0, 1.0]
    ):
        super().__init__()
        self.layers = layers
        self.weights = weights

        # TODO: Load VGG and extract feature layers
        pass

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted HR image
            target: Ground truth HR image

        Returns:
            perceptual_loss
        """
        # TODO: Implement
        # 1. Extract features from pred and target
        # 2. Compute distance at each layer
        # 3. Weighted sum
        pass


class SSIMLoss(nn.Module):
    """SSIM Loss

    Structural similarity loss.

    TODO: Implement SSIM loss
    """

    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            1 - SSIM (to minimize)
        """
        # TODO: Implement SSIM
        pass
