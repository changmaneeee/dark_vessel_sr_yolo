from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union


class BasePipeline(ABC, nn.Module):
    """
    Base class for SR-Detection Pipelines.
    All pipeline architectures should inherit from this base class.
    """

    def __init__(self, config: Any):
        super().__init__()

        self.config = config
        
        self.device = getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        