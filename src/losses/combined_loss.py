"""
=============================================================================
combined_loss.py - SR + Detection 결합 Loss
=============================================================================

[역할]
- Super-Resolution Loss와 Detection Loss를 결합
- 가중치 동적 조절 (Phase별 스케줄링)
- Joint training (Arch 2/4/5) 지원

[Loss 공식]
L_total = α × L_SR + β × L_detection + γ × L_auxiliary

where:
    L_SR = L1/Charbonnier + λ × Perceptual
    L_detection = λ_box × L_box + λ_cls × L_cls + λ_dfl × L_dfl
    L_auxiliary = Feature consistency 등 (선택)

[Phase별 가중치 스케줄링]
Phase 1 (epoch 0-50):   α=0.7, β=0.3  (SR 안정화)
Phase 2 (epoch 50-150): α→0.2, β→0.8  (점진적 전환)
Phase 3 (epoch 150+):   α=0.2, β=0.8  (Detection 집중)

[사용 예시]
loss_fn = CombinedLoss(
    sr_weight=0.5,
    det_weight=0.5,
    yolo_model=yolo_wrapper.detection_model
)

# 학습 루프
for epoch in range(epochs):
    loss_fn.update_weights(epoch)  # Phase 스케줄링
    
    for batch in dataloader:
        loss_dict = loss_fn(
            sr_image=sr_output,
            hr_gt=hr_target,
            det_preds=yolo_preds,  # Training mode output
            det_batch=det_batch    # batch dictionary
        )
        loss_dict['total'].backward()
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union
import math


class CombinedLoss(nn.Module):
    """
    SR + Detection 결합 Loss
    
    [지원 기능]
    1. SR Loss: L1/Charbonnier + Perceptual (선택)
    2. Detection Loss: v8DetectionLoss
    3. 가중치 동적 조절 (Phase 스케줄링)
    4. Auxiliary Loss (Feature consistency 등)
    """
    
    def __init__(
        self,
        sr_weight: float = 0.5,
        det_weight: float = 0.5,
        aux_weight: float = 0.0,
        yolo_model: Optional[nn.Module] = None,
        use_charbonnier: bool = True,
        perceptual_weight: float = 0.0,
        phase_schedule: bool = True,
        phase1_epochs: int = 50,
        phase2_epochs: int = 100,
        alpha_start: float = 0.7,
        alpha_end: float = 0.2
    ):
        """
        Args:
            sr_weight: SR Loss 가중치 (α)
            det_weight: Detection Loss 가중치 (β)
            aux_weight: Auxiliary Loss 가중치 (γ)
            yolo_model: YOLO DetectionModel (Detection Loss용)
            use_charbonnier: True면 Charbonnier, False면 L1
            perceptual_weight: Perceptual Loss 가중치
            phase_schedule: Phase별 가중치 스케줄링 사용 여부
            phase1_epochs: Phase 1 epoch 수
            phase2_epochs: Phase 2 epoch 수
            alpha_start: Phase 1에서의 SR 가중치
            alpha_end: Phase 3에서의 SR 가중치
        """
        super().__init__()
        
        # 가중치 저장
        self.register_buffer('sr_weight', torch.tensor(sr_weight))
        self.register_buffer('det_weight', torch.tensor(det_weight))
        self.aux_weight = aux_weight
        
        # Phase 스케줄링 설정
        self.phase_schedule = phase_schedule
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        
        # SR Loss
        from src.losses.sr_loss import SRLoss
        self.sr_loss = SRLoss(
            l1_weight=1.0,
            perceptual_weight=perceptual_weight,
            use_charbonnier=use_charbonnier
        )
        
        # Detection Loss (Lazy initialization)
        self.yolo_model = yolo_model
        self._det_loss = None
    
    def _init_det_loss(self):
        """Detection Loss Lazy Initialization"""
        if self._det_loss is not None:
            return
        
        if self.yolo_model is None:
            print("[CombinedLoss] Warning: YOLO model not provided, detection loss disabled")
            return
        
        try:
            from ultralytics.utils.loss import v8DetectionLoss
            self._det_loss = v8DetectionLoss(self.yolo_model)
            print("[CombinedLoss] ✓ v8DetectionLoss initialized")
        except ImportError:
            print("[CombinedLoss] Warning: v8DetectionLoss not available")
    
    def forward(
        self,
        sr_image: Optional[torch.Tensor] = None,
        hr_gt: Optional[torch.Tensor] = None,
        det_preds: Optional[Union[torch.Tensor, list]] = None,
        det_batch: Optional[Dict[str, torch.Tensor]] = None,
        aux_loss: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        결합 Loss 계산
        
        Args:
            sr_image: SR 출력 이미지 [B, 3, H, W]
            hr_gt: HR Ground Truth [B, 3, H, W]
            det_preds: YOLO predictions (training mode output)
            det_batch: Detection batch dictionary
            aux_loss: 추가 auxiliary loss (이미 계산된 것)
        
        Returns:
            loss_dict: {
                'total': 전체 loss,
                'sr_loss': SR loss,
                'det_loss': Detection loss,
                'aux_loss': Auxiliary loss (있으면),
                'box_loss': Box regression loss,
                'cls_loss': Classification loss,
                'dfl_loss': DFL loss
            }
        """
        device = sr_image.device if sr_image is not None else (
            det_preds[0].device if det_preds else torch.device('cpu')
        )
        
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # =====================================================================
        # SR Loss
        # =====================================================================
        if sr_image is not None and hr_gt is not None:
            sr_loss_dict = self.sr_loss(sr_image, hr_gt)
            sr_loss = sr_loss_dict['total']
            
            total_loss = total_loss + self.sr_weight * sr_loss
            
            loss_dict['sr_loss'] = sr_loss
            loss_dict['pixel_loss'] = sr_loss_dict.get('pixel_loss', sr_loss)
            if 'perceptual_loss' in sr_loss_dict:
                loss_dict['perceptual_loss'] = sr_loss_dict['perceptual_loss']
        else:
            loss_dict['sr_loss'] = torch.tensor(0.0, device=device)
        
        # =====================================================================
        # Detection Loss
        # =====================================================================
        if det_preds is not None and det_batch is not None:
            self._init_det_loss()
            
            if self._det_loss is not None:
                try:
                    det_total, det_items = self._det_loss(det_preds, det_batch)
                    
                    total_loss = total_loss + self.det_weight * det_total
                    
                    loss_dict['det_loss'] = det_total
                    loss_dict['box_loss'] = det_items[0] if len(det_items) > 0 else torch.tensor(0.0, device=device)
                    loss_dict['cls_loss'] = det_items[1] if len(det_items) > 1 else torch.tensor(0.0, device=device)
                    loss_dict['dfl_loss'] = det_items[2] if len(det_items) > 2 else torch.tensor(0.0, device=device)
                except Exception as e:
                    print(f"[CombinedLoss] Detection loss error: {e}")
                    loss_dict['det_loss'] = torch.tensor(0.0, device=device)
                    loss_dict['box_loss'] = torch.tensor(0.0, device=device)
                    loss_dict['cls_loss'] = torch.tensor(0.0, device=device)
                    loss_dict['dfl_loss'] = torch.tensor(0.0, device=device)
            else:
                loss_dict['det_loss'] = torch.tensor(0.0, device=device)
        else:
            loss_dict['det_loss'] = torch.tensor(0.0, device=device)
        
        # =====================================================================
        # Auxiliary Loss
        # =====================================================================
        if aux_loss is not None and self.aux_weight > 0:
            total_loss = total_loss + self.aux_weight * aux_loss
            loss_dict['aux_loss'] = aux_loss
        
        loss_dict['total'] = total_loss
        
        return loss_dict
    
    # =========================================================================
    # Phase Scheduling
    # =========================================================================
    
    def update_weights(self, epoch: int) -> Tuple[float, float]:
        """
        Phase별 가중치 업데이트
        
        Args:
            epoch: 현재 epoch
        
        Returns:
            (sr_weight, det_weight)
        """
        if not self.phase_schedule:
            return self.sr_weight.item(), self.det_weight.item()
        
        if epoch < self.phase1_epochs:
            # Phase 1: SR 안정화
            alpha = self.alpha_start
        elif epoch < self.phase1_epochs + self.phase2_epochs:
            # Phase 2: 선형 전환
            progress = (epoch - self.phase1_epochs) / self.phase2_epochs
            alpha = self.alpha_start - progress * (self.alpha_start - self.alpha_end)
        else:
            # Phase 3: Detection 집중
            alpha = self.alpha_end
        
        beta = 1.0 - alpha
        
        # 버퍼 업데이트
        self.sr_weight.fill_(alpha)
        self.det_weight.fill_(beta)
        
        return alpha, beta
    
    def get_weights(self) -> Dict[str, float]:
        """현재 가중치 반환"""
        return {
            'sr_weight': self.sr_weight.item(),
            'det_weight': self.det_weight.item(),
            'aux_weight': self.aux_weight
        }
    
    def set_weights(
        self,
        sr_weight: Optional[float] = None,
        det_weight: Optional[float] = None,
        aux_weight: Optional[float] = None
    ) -> None:
        """가중치 수동 설정"""
        if sr_weight is not None:
            self.sr_weight.fill_(sr_weight)
        if det_weight is not None:
            self.det_weight.fill_(det_weight)
        if aux_weight is not None:
            self.aux_weight = aux_weight


class FeatureConsistencyLoss(nn.Module):
    """
    Feature Consistency Loss (Arch 5-B Auxiliary Loss)
    
    SR feature와 YOLO feature의 일관성 유지
    
    L_consistency = ||SR_feat - YOLO_feat||²
    
    [사용]
    fusion에서 SR feature와 YOLO feature가 비슷한 정보를 담도록 유도
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        sr_features: Dict[str, torch.Tensor],
        yolo_features: Dict[str, torch.Tensor],
        align_channels: bool = True
    ) -> torch.Tensor:
        """
        Args:
            sr_features: SR feature dict {'p3': ..., 'p4': ..., 'p5': ...}
            yolo_features: YOLO feature dict
            align_channels: 채널 수 맞추기 (projection 필요)
        
        Returns:
            consistency loss
        """
        loss = 0.0
        count = 0
        
        for key in sr_features.keys():
            if key in yolo_features:
                sr_feat = sr_features[key]
                yolo_feat = yolo_features[key]
                
                # Spatial size 맞추기
                if sr_feat.shape[2:] != yolo_feat.shape[2:]:
                    sr_feat = nn.functional.interpolate(
                        sr_feat,
                        size=yolo_feat.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # L2 loss (채널 수가 같다고 가정, 다르면 projection 필요)
                if sr_feat.shape[1] == yolo_feat.shape[1]:
                    loss = loss + nn.functional.mse_loss(sr_feat, yolo_feat)
                    count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Combined Loss 테스트")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # SR Loss만 테스트
    print("\n1. SR Loss만 테스트")
    
    loss_fn = CombinedLoss(sr_weight=1.0, det_weight=0.0)
    
    sr_image = torch.rand(2, 3, 256, 256, device=device)
    hr_gt = torch.rand(2, 3, 256, 256, device=device)
    
    loss_dict = loss_fn(sr_image=sr_image, hr_gt=hr_gt)
    
    print("Loss 결과:")
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.6f}")
    
    # Phase 스케줄링 테스트
    print("\n2. Phase 스케줄링 테스트")
    
    loss_fn = CombinedLoss(
        phase_schedule=True,
        phase1_epochs=50,
        phase2_epochs=100,
        alpha_start=0.7,
        alpha_end=0.2
    )
    
    for epoch in [0, 25, 50, 100, 150, 200]:
        sr_w, det_w = loss_fn.update_weights(epoch)
        print(f"  Epoch {epoch:3d}: sr_weight={sr_w:.2f}, det_weight={det_w:.2f}")
    
    # Detection Loss 테스트 (Ultralytics 있을 때)
    print("\n3. Detection Loss 테스트")
    
    try:
        from ultralytics import YOLO
        
        yolo = YOLO("yolov8n.pt")
        loss_fn = CombinedLoss(
            sr_weight=0.5,
            det_weight=0.5,
            yolo_model=yolo.model
        )
        
        # 더미 데이터
        images = torch.randn(2, 3, 640, 640, device=device)
        
        # YOLO forward (training mode)
        yolo.model.train()
        preds = yolo.model(images)
        
        # Batch dictionary
        det_batch = {
            'batch_idx': torch.tensor([0, 0, 1], dtype=torch.float32, device=device),
            'cls': torch.tensor([0, 0, 0], dtype=torch.float32, device=device),
            'bboxes': torch.tensor([
                [0.5, 0.5, 0.2, 0.2],
                [0.3, 0.7, 0.15, 0.15],
                [0.6, 0.4, 0.25, 0.25],
            ], dtype=torch.float32, device=device),
            'img': images,
        }
        
        # SR 더미
        sr_image = torch.rand(2, 3, 640, 640, device=device)
        hr_gt = torch.rand(2, 3, 640, 640, device=device)
        
        loss_dict = loss_fn(
            sr_image=sr_image,
            hr_gt=hr_gt,
            det_preds=preds,
            det_batch=det_batch
        )
        
        print("Combined Loss 결과:")
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.item():.6f}")
        
        print("\n✓ Combined Loss (SR + Detection) 성공!")
        
    except ImportError:
        print("ultralytics 미설치 - Detection Loss 테스트 스킵")
    except Exception as e:
        print(f"테스트 실패: {e}")
    
    print("\n✓ Combined Loss 테스트 완료!")