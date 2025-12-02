"""RFDN: Residual Feature Distillation Network

Lightweight SR model with residual feature distillation.
Reference: "Residual Feature Distillation Network for Lightweight Image Super-Resolution"
"""

import torch
import torch.nn as nn
from src.models.sr_models.base_sr import BaseSRModel


class RFDN(BaseSRModel):
    """Residual Feature Distillation Network for Super-Resolution

    TODO: Implement RFDN architecture
    - Shallow feature extraction
    - Multiple RFD blocks
    - Upsampling module (PixelShuffle)
    - Reconstruction
    """

    def __init__(self, scale_factor: int = 4, in_channels: int = 3, out_channels: int = 3):
        super().__init__(scale_factor, in_channels, out_channels)

        # TODO: Implement architecture
        pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode LR image to features

        TODO: Implement feature extraction
        """
        # Placeholder
        return x

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features to HR image

        TODO: Implement upsampling and reconstruction
        """
        # Placeholder
        return features
