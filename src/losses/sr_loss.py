"""
=============================================================================
sr_loss.py - Super Resolution Loss Functions
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class L1Loss(nn.Module):
    """기본 L1 (MAE) Loss"""
    
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (Smooth L1의 변형)
    
    L1보다 outlier에 robust하고, L2보다 edge 보존력이 좋음
    RFDN 원 논문에서 사용
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()


class SSIMLoss(nn.Module):
    """
    SSIM (Structural Similarity) Loss
    
    구조적 유사성 측정 - 인간 시각에 더 가까운 품질 평가
    """
    
    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        
        # Gaussian window
        self.window = self._create_window(window_size, channel)
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Gaussian window 생성"""
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([
                torch.exp(torch.tensor(-(x - window_size//2)**2 / float(2*sigma**2)))
                for x in range(window_size)
            ])
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channel = pred.size(1)
        
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)
        
        if channel != self.channel:
            self.window = self._create_window(self.window_size, channel).to(pred.device)
            self.channel = channel
        
        mu1 = F.conv2d(pred, self.window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred*pred, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target*target, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred*target, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()


class SRLoss(nn.Module):
    """
    SR 통합 Loss
    
    L1 + λ_ssim * SSIM 조합
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.0,
        charbonnier: bool = True
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        
        # L1 or Charbonnier
        self.l1_loss = CharbonnierLoss() if charbonnier else L1Loss()
        
        # SSIM (선택)
        self.ssim_loss = SSIMLoss() if ssim_weight > 0 else None
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: SR 출력 [B, 3, H, W]
            target: HR GT [B, 3, H, W]
        
        Returns:
            {
                'total': 전체 loss,
                'l1': L1 loss,
                'ssim': SSIM loss (사용 시)
            }
        """
        # L1 Loss
        l1 = self.l1_loss(pred, target)
        total = self.l1_weight * l1
        
        result = {
            'l1': l1
        }
        
        # SSIM Loss
        if self.ssim_loss is not None:
            ssim = self.ssim_loss(pred, target)
            total = total + self.ssim_weight * ssim
            result['ssim'] = ssim
        
        result['total'] = total
        return result


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("SR Loss 테스트")
    
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    
    # L1 only
    loss_fn = SRLoss(l1_weight=1.0, ssim_weight=0.0)
    result = loss_fn(pred, target)
    print(f"L1 Loss: {result['total'].item():.4f}")
    
    # L1 + SSIM
    loss_fn = SRLoss(l1_weight=1.0, ssim_weight=0.1)
    result = loss_fn(pred, target)
    print(f"L1 + SSIM Loss: {result['total'].item():.4f}")
    
    print("✓ 테스트 완료!")