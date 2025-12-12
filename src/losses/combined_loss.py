"""
==========================================================
combined_loss.py - SR + Detection 결합 Loss
==========================================================

[역할]
- SR Loss와 Detection Loss를 결합
- 가중치 동적 조절(Phase별 스케줄링)
- DetectionLoss 모듈 활용

[Loss]
L_total = a*L_sr + b*L_detection

where:
    L_sr: L1/Charbonnier + lamda * Perceptual
    L_detection: v8DetectionLoss

[Phase별 스케줄링]
Phase 1 (epoch 0-50): a=0.7, b=0.3
Phase 2 (epoch 50-150): a=0.2, b=0.8
Phase 3 (epoch 150+): a=0.2, b=0.8

"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union

from src.losses.sr_loss import SRLoss
from src.losses.detection_loss import DetectionLoss

class CombinedLoss(nn.Module):
    """
    SR + Detection 결합 Loss
    """
    def __init__(
            self,
            yolo_model: nn.Module,
            sr_weight: float =0.5,
            det_weight: float =0.5,
            use_charbonnier: bool = True,
            perceptual_weight: float =0.0,
            phase_schedule: bool = True,
            phase1_epochs: int = 50,
            phase2_epochs: int = 100,
            alpha_start: float = 0.7,
            alpha_end: float = 0.2
    ):
        super().__init__()
        self.register_buffer('sr_weight', torch.tensor(sr_weight))
        self.register_buffer('det_weight', torch.tensor(det_weight))

        # Phase 설정
        self.phase_schedule = phase_schedule
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end

        # SR Loss
        self.sr_loss_fn = SRLoss(
            l1_weight=1.0,
            perceptual_weight = perceptual_weight,
            use_charbonnier=use_charbonnier
        )

        # Detection Loss
        self.det_loss_fn = DetectionLoss(yolo_model)

        print("[CombinedLoss] Initialed")
        print(f"  SR weight: {sr_weight})")
        print(f"  Detection weight: {det_weight})")
        print(f" -Phase schedule: {phase_schedule}")

    def forward(
            self,
            sr_image: Optional[torch.Tensor] = None,
            hr_gt: Optional[torch.Tensor]= None,
            det_preds: Optional[Union[torch.Tensor, list]]=None,
            det_targets: Optional[torch.Tensor]=None,
            images: Optional[torch.Tensor]=None
    ) -> Dict[str, torch.Tensor]:
        
        device = self._get_device(sr_image, det_preds, images)

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device, requires_grad = True)

        
        #==========================================================================
        # SR Loss
        #==========================================================================

        if sr_image is not None and hr_gt is not None:
            sr_loss_dict = self.sr_loss_fn(sr_image, hr_gt)
            sr_loss = sr_loss_dict['total']

            total_loss = total_loss + self.sr_weight * sr_loss

            loss_dict['sr_loss'] = sr_loss
            loss_dict['pixel_loss'] = sr_loss_dict.get('pixel_loss', sr_loss)
            if 'perceptual_loss' in sr_loss_dict:
                loss_dict['perceptual_loss'] = sr_loss_dict['perceptual_loss']
        else:
            loss_dict['sr_loss'] = torch.tensor(0.0, device=device)


        #==========================================================================
        # Detection Loss
        #==========================================================================

        if det_preds is not None and det_targets is not None and images is not None:
            det_loss_dict = self.det_loss_fn(det_preds, det_targets, images)
            det_loss = det_loss_dict['total']

            total_loss = total_loss + self.det_weight * det_loss

            loss_dict['det_loss'] = det_loss
            loss_dict['box_loss'] = det_loss_dict.get('box_loss', torch.tensor(0.0, device=device))
            loss_dict['cls_loss'] = det_loss_dict.get('cls_loss', torch.tensor(0.0, device=device))
            loss_dict['dfl_loss'] = det_loss_dict.get('dfl_loss', torch.tensor(0.0, device=device))
        else:
            loss_dict['det_loss'] = torch.tensor(0.0, device=device)
            loss_dict['box_loss'] = torch.tensor(0.0, device=device)
            loss_dict['cls_loss'] = torch.tensor(0.0, device=device)
            loss_dict['dfl_loss'] = torch.tensor(0.0, device=device)

        loss_dict['total'] = total_loss
        return loss_dict
    
    def _get_device(self, *tensors) -> torch.device:
        
        for t in tensors:
            if t is not None:
                if isinstance(t, torch.Tensor):
                    return t.device
                elif isinstance(t, list) and len(t) > 0:
                    return t[0].device
        return torch.device('cpu')
    
    #==========================================================================
    # Phase Scheduling
    #==========================================================================

    def update_weight(self, epoch: int) -> Tuple[float, float]:
        """
        Update weight per Phase
        """
        if not self.phase_schedule:
            return self.sr_weight.item(), self.det_weight.item()
        
        if epoch < self.phase1_epochs:
            alpha = self.alpha_start

        elif epoch < self.phase1_epochs + self.phase2_epochs:
            progress = (epoch - self.phase1_epochs) / self.phase2_epochs
            alpha = self.alpha_start - progress * (self.alpha_start - self.alpha_end)
        else:
            alpha = self.alpha_end
        
        beta = 1.0 - alpha

        self.sr_weight.fill_(alpha)
        self.det_weight.fill_(beta)

        return alpha, beta
    
    def get_weights(self) -> Dict[str, float]:
        # Current Weights return
        return {
            'sr_weight': self.sr_weight.item(),
            'det_weight': self.det_weight.item()
        }
    
    def set_weights(
            self,
            sr_weight: Optional[float] = None,
            det_weight: Optional[float] = None
    )-> None:
        
        if sr_weight is not None:
            self.sr_weight.fill_(sr_weight)
        if det_weight is not None:
            self.det_weight.fill_(det_weight)

#========================================================================
#Test
#========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CombinedLoss test")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\n[1. SR Loss만 테스트]")

    try:
        from ultralytics import YOLO

        yolo = YOLO("yolov8nt.py") #???
        model = yolo.model.to(device)

        loss_fn = CombinedLoss(
            yolo_model = model, 
            sr_weight = 1.0,
            det_weight = 0.0
        )

        sr_image = torch.randn(2,3, 256, 256).to(device)
        hr_gt = torch.randn(2,3, 256, 256, device=device)

        loss_dict = loss_fn(sr_image=sr_image, hr_gt=hr_gt)

        print(" SR Loss 결과")
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.item():.6f}")

        print("\n [2. Phase Scheduling test]")

        loss_fn2 = CombinedLoss(
            yolo_model = model,
            phase_schedule=True,
            phase1_epochs=50,
            phase2_epochs=100
        )

        for epoch in [0, 25, 50, 75, 100, 125, 150, 175, 200]:
            sr_w, det_w = loss_fn2.update_weight(epoch)
            print(f" Epoch {epoch}: SR weight={sr_w:.3f}, Det weight={det_w:.3f}")

        print("\n[3. SR + Detection combine test]")

        loss_fn3 = CombinedLoss(
            yolo_model = model,
            sr_weight = 0.3,
            det_weight = 0.7
        )

        images = torch.randn(2,3,640, 640, device=device)
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.2, 0.2],
            [1, 0, 0.3, 0.7, 0.15, 0.25],
        ], device=device)

        model.train()
        preds = model(images)

        sr_image = torch.rand(2,3,640,640, device=device)
        hr_gt = torch.rand(2,3,640,640, device=device)

        loss_dict = loss_fn3(
            sr_image=sr_image, 
            hr_gt = hr_gt,
            det_preds=preds,
            det_targets=targets,
            images=images
        )

        print("  Combined Loss Results")
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"   {k}: {v.item():.6f}")
        print("\n CombinedLoss test done.")
        print(" DetectionLoss Module No duplication")

    except ImportError:
        print("ultralytics package not found. Skipping DetectionLoss test.")
    except Exception as e:
        print(f"test failed: {e}")
        import traceback
        traceback.print_exc()
