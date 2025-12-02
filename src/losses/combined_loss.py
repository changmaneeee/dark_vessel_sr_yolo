"""Combined Loss for Multi-Task Learning

Weighted combination of SR and Detection losses.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class CombinedLoss(nn.Module):
    """Combined SR + Detection Loss

    TODO: Implement combined loss
    - SR reconstruction loss
    - Detection loss
    - Feature matching loss (for Arch5B)
    - Dynamic weighting
    """

    def __init__(
        self,
        sr_loss: nn.Module,
        det_loss: nn.Module,
        sr_alpha: float = 0.7,
        det_beta: float = 0.3,
        feature_gamma: float = 0.0,
        dynamic_weighting: bool = False
    ):
        """
        Args:
            sr_loss: SR loss module
            det_loss: Detection loss module
            sr_alpha: SR loss weight
            det_beta: Detection loss weight
            feature_gamma: Feature matching loss weight
            dynamic_weighting: Use uncertainty-based weighting
        """
        super().__init__()
        self.sr_loss = sr_loss
        self.det_loss = det_loss
        self.sr_alpha = sr_alpha
        self.det_beta = det_beta
        self.feature_gamma = feature_gamma
        self.dynamic_weighting = dynamic_weighting

        # TODO: Implement dynamic weighting (learnable parameters)
        if dynamic_weighting:
            # log_vars for uncertainty weighting
            pass

    def forward(
        self,
        sr_pred: Optional[torch.Tensor],
        sr_target: Optional[torch.Tensor],
        det_pred: Dict[str, torch.Tensor],
        det_target: Dict[str, torch.Tensor],
        features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss

        Args:
            sr_pred: Predicted SR image
            sr_target: Ground truth HR image
            det_pred: Detection predictions
            det_target: Detection targets
            features: Feature pairs for matching loss

        Returns:
            loss_dict: {
                'total_loss': weighted sum,
                'sr_loss': SR loss,
                'det_loss': Detection loss,
                'feature_loss': Feature matching loss
            }
        """
        # TODO: Implement
        # 1. Compute SR loss (if provided)
        # 2. Compute detection loss
        # 3. (Optional) Feature matching loss
        # 4. Weighted combination
        pass

    def feature_matching_loss(
        self,
        sr_features: torch.Tensor,
        det_features: torch.Tensor,
        method: str = "cosine"
    ) -> torch.Tensor:
        """Feature matching loss for Arch5B

        Args:
            sr_features: Features from SR encoder
            det_features: Features from detection backbone
            method: 'cosine', 'l2', 'kl'

        Returns:
            feature_loss
        """
        # TODO: Implement feature matching
        pass
