
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseSRModel(ABC, nn.Module):
    """
    All SR models should inherit from this base class.
    """

    def __init__(self, 
                 scale_factor: int =4,
                 in_channels: int =3, 
                 out_channels: int =3,
                 feature_channels: int = 50):
        
        super().__init__()
        self.scale_factor = scale_factor
        self.feature_channels = feature_channels

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor: #encoder role

        pass

    @abstractmethod
    def forward_reconstruct(self, x: torch.Tensor) -> torch.Tensor: #decoder role
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Standard forward method for SR models.
        """
        features = self.forward_features(x)
        hr_image = self.forward_reconstruct(features)
        return hr_image

    def get_feature_info(self) -> Dict[str, Any]:
        
        return {
            'channels': self.feature_channels,
            'scale' : self.scale_factor
        }