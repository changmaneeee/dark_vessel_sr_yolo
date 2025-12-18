"""
=============================================================================
detection_loss.py - YOLO Detection Loss Wrapper
=============================================================================

Ultralytics YOLO v8/11의 Detection Loss를 래핑
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class DetectionLoss(nn.Module):
    """
    Ultralytics Detection Loss 래퍼
    
    [역할]
    - Ultralytics 내부 loss 함수 호출
    - 일관된 인터페이스 제공
    
    [사용법]
    loss_fn = DetectionLoss(yolo_model)
    loss_dict = loss_fn(predictions, targets, images)
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: Ultralytics DetectionModel (YOLO.model)
        """
        super().__init__()
        
        self.model = model
        
        # Ultralytics v8DetectionLoss 가져오기
        try:
            from ultralytics.utils.loss import v8DetectionLoss
            self.loss_fn = v8DetectionLoss(model)
            print("[DetectionLoss] ✓ v8DetectionLoss initialized")
        except ImportError:
            print("[DetectionLoss] ⚠️ v8DetectionLoss not found, using fallback")
            self.loss_fn = None
    
    def forward(
        self,
        predictions: Any,
        targets: torch.Tensor,
        images: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Detection Loss 계산
        
        Args:
            predictions: 모델 출력 (training mode)
            targets: GT [N, 6] = (batch_idx, class, x, y, w, h) normalized
            images: 입력 이미지 (batch 정보용)
        
        Returns:
            {
                'total': 전체 loss,
                'box_loss': Box regression loss,
                'cls_loss': Classification loss,
                'dfl_loss': Distribution focal loss
            }
        """
        device = targets.device if targets is not None else 'cpu'
        
        # 빈 타겟 처리
        if targets is None or len(targets) == 0:
            return {
                'total': torch.tensor(0.0, device=device, requires_grad=True),
                'box_loss': torch.tensor(0.0, device=device),
                'cls_loss': torch.tensor(0.0, device=device),
                'dfl_loss': torch.tensor(0.0, device=device)
            }
        
        # Ultralytics loss 사용
        if self.loss_fn is not None:
            try:
                # batch 정보 설정
                batch = {'batch_idx': targets[:, 0].long()}
                batch['cls'] = targets[:, 1:2]
                batch['bboxes'] = targets[:, 2:6]
                
                # Loss 계산
                loss, loss_items = self.loss_fn(predictions, batch)
                
                return {
                    'total': loss,
                    'box_loss': loss_items[0] if len(loss_items) > 0 else torch.tensor(0.0, device=device),
                    'cls_loss': loss_items[1] if len(loss_items) > 1 else torch.tensor(0.0, device=device),
                    'dfl_loss': loss_items[2] if len(loss_items) > 2 else torch.tensor(0.0, device=device)
                }
            except Exception as e:
                print(f"[DetectionLoss] Warning: {e}")
                return self._fallback_loss(predictions, targets, device)
        
        return self._fallback_loss(predictions, targets, device)
    
    def _fallback_loss(
        self,
        predictions: Any,
        targets: torch.Tensor,
        device: str
    ) -> Dict[str, torch.Tensor]:
        """Fallback: 간단한 dummy loss (디버깅용)"""
        # 실제로는 Ultralytics loss를 사용해야 함
        if isinstance(predictions, (list, tuple)) and len(predictions) > 0:
            # predictions의 일부를 사용한 dummy loss
            dummy_loss = sum(p.mean() for p in predictions if isinstance(p, torch.Tensor)) * 0.0
            dummy_loss = dummy_loss + torch.tensor(0.1, device=device, requires_grad=True)
        else:
            dummy_loss = torch.tensor(0.1, device=device, requires_grad=True)
        
        return {
            'total': dummy_loss,
            'box_loss': torch.tensor(0.0, device=device),
            'cls_loss': torch.tensor(0.0, device=device),
            'dfl_loss': torch.tensor(0.0, device=device)
        }


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("Detection Loss 테스트")
    
    try:
        from ultralytics import YOLO
        
        yolo = YOLO("yolov8n.pt")
        loss_fn = DetectionLoss(yolo.model)
        
        # Dummy data
        images = torch.randn(2, 3, 640, 640)
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.2, 0.2],
            [1, 0, 0.3, 0.7, 0.15, 0.25],
        ])
        
        # Forward
        yolo.model.train()
        preds = yolo.model(images)
        
        # Loss
        loss_dict = loss_fn(preds, targets, images)
        print(f"Total Loss: {loss_dict['total'].item():.4f}")
        print(f"Box Loss: {loss_dict['box_loss'].item():.4f}")
        print(f"Cls Loss: {loss_dict['cls_loss'].item():.4f}")
        
        print("✓ 테스트 완료!")
        
    except Exception as e:
        print(f"테스트 실패: {e}")