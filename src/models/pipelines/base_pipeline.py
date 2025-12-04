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
        
        self.sr_model: Optional[nn.Module] = None
        self.detector: Optional[nn.Module] = None

        training_config = getattr(config, 'training', {})
        self._sr_weight: getattr(training_config, 'sr_weight', 0.5)
        self._det_weight: getattr(training_config, 'det_weight', 0.5)

     @abstractmethod
     def forward(self, lr_image: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """
        Input: LR image tensor
        Output: SR image and detection results
        """   
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        output: Tuple[torch.Tensor, Any],
        targets: Any,
        hr_gt: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        pass

    @torch.no_grad()
    def inference(
        self,
        lr_image: torch.Tensor,
        conf_threshold: float = 0.25 # Confidence threshold for detection
    ) ->Dict[str, Any]:

    self.eval()

    sr_image, detections = self.forward(lr_image)

    results= {'detections': detections}
    if return_sr:
        results['sr_image'] = sr_image
    return results

    def freeze_sr(self) -> None:
        """Freeze SR model parameters."""
        if self.sr_model is not None:
            for param in self.sr_model.parameters():
                param.requires_grad = False
            print("SR model frozen!")

    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            **kwargs
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at {path}")        
