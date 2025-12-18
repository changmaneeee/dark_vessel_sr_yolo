"""
=============================================================================
combined_loss.py - SR + Detection Combined Loss
=============================================================================

Arch5-B에서 사용하는 통합 Loss
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .sr_loss import SRLoss
from .detection_loss import DetectionLoss


class CombinedLoss(nn.Module):
    """
    SR Loss + Detection Loss 통합
    
    Total Loss = α * SR_Loss + β * Det_Loss
    
    [Phase별 가중치]
    - Phase 1: α=1.0, β=0.0 (SR만 학습)
    - Phase 2: α=0.0, β=1.0 (Detection만, Fusion 학습)
    - Phase 3: α=0.1, β=1.0 (전체 fine-tune)
    """
    
    def __init__(
        self,
        yolo_model: nn.Module,
        sr_weight: float = 0.0,
        det_weight: float = 1.0,
        sr_config: Dict = None,
        phase_schedule: bool = False
    ):
        """
        Args:
            yolo_model: YOLO DetectionModel
            sr_weight: SR loss 가중치 (α)
            det_weight: Detection loss 가중치 (β)
            sr_config: SR loss 설정 (l1_weight, ssim_weight 등)
            phase_schedule: Phase별 가중치 자동 조절 여부
        """
        super().__init__()
        
        self.sr_weight = sr_weight
        self.det_weight = det_weight
        self.phase_schedule = phase_schedule
        self.current_phase = 2  # 기본값
        
        # SR Loss
        sr_config = sr_config or {}
        self.sr_loss = SRLoss(
            l1_weight=sr_config.get('l1_weight', 1.0),
            ssim_weight=sr_config.get('ssim_weight', 0.0),
            charbonnier=sr_config.get('charbonnier', True)
        )
        
        # Detection Loss
        self.det_loss = DetectionLoss(yolo_model)
    
    def set_phase(self, phase: int):
        """
        학습 Phase 설정
        
        Args:
            phase: 1, 2, or 3
        """
        self.current_phase = phase
        
        if self.phase_schedule:
            if phase == 1:
                self.sr_weight = 1.0
                self.det_weight = 0.0
            elif phase == 2:
                self.sr_weight = 0.0
                self.det_weight = 1.0
            elif phase == 3:
                self.sr_weight = 0.1
                self.det_weight = 1.0
        
        print(f"[CombinedLoss] Phase {phase}: α={self.sr_weight}, β={self.det_weight}")
    
    def forward(
        self,
        predictions: Any,
        targets: torch.Tensor,
        sr_pred: Optional[torch.Tensor] = None,
        sr_target: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Combined Loss 계산
        
        Args:
            predictions: YOLO 예측
            targets: Detection GT [N, 6]
            sr_pred: SR 출력 (선택)
            sr_target: HR GT (선택)
            images: 입력 이미지
        
        Returns:
            {
                'total': 전체 loss,
                'sr_loss': SR loss,
                'det_loss': Detection loss,
                'box_loss': Box loss,
                'cls_loss': Cls loss,
                'dfl_loss': DFL loss
            }
        """
        device = targets.device if targets is not None and len(targets) > 0 else 'cpu'
        
        result = {
            'sr_loss': torch.tensor(0.0, device=device),
            'det_loss': torch.tensor(0.0, device=device),
            'box_loss': torch.tensor(0.0, device=device),
            'cls_loss': torch.tensor(0.0, device=device),
            'dfl_loss': torch.tensor(0.0, device=device)
        }
        
        total = torch.tensor(0.0, device=device, requires_grad=True)
        
        # SR Loss
        if self.sr_weight > 0 and sr_pred is not None and sr_target is not None:
            sr_result = self.sr_loss(sr_pred, sr_target)
            result['sr_loss'] = sr_result['total']
            total = total + self.sr_weight * sr_result['total']
        
        # Detection Loss
        if self.det_weight > 0 and targets is not None:
            det_result = self.det_loss(predictions, targets, images)
            result['det_loss'] = det_result['total']
            result['box_loss'] = det_result['box_loss']
            result['cls_loss'] = det_result['cls_loss']
            result['dfl_loss'] = det_result['dfl_loss']
            total = total + self.det_weight * det_result['total']
        
        result['total'] = total
        return result


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("Combined Loss 테스트")
    
    try:
        from ultralytics import YOLO
        
        yolo = YOLO("yolov8n.pt")
        loss_fn = CombinedLoss(yolo.model, sr_weight=0.1, det_weight=1.0)
        
        # Phase 설정 테스트
        loss_fn.set_phase(2)
        loss_fn.set_phase(3)
        
        print("✓ 테스트 완료!")
        
    except Exception as e:
        print(f"테스트 실패: {e}")