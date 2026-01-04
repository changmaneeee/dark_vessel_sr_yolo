"""
=============================================================================
detection_loss.py - YOLO Detection Loss Wrapper
=============================================================================
Ultralytics YOLO v8/11의 Detection Loss를 래핑

[수정 내역]
- BatchData 클래스: dict 접근과 속성 접근 모두 지원
- Ultralytics 호환성 완전 확보
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class BatchData:
    """
    Ultralytics 호환 Batch 데이터 클래스
    
    dict처럼도, 속성으로도 접근 가능:
    - batch['cls']  → OK
    - batch.cls     → OK
    """
    
    def __init__(self, **kwargs):
        self._data = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
        setattr(self, key, value)
    
    def __contains__(self, key):
        return key in self._data
    
    def __len__(self):
        return len(self._data)
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()
    
    def get(self, key, default=None):
        return self._data.get(key, default)


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
                # =====================================================
                # [핵심] BatchData 클래스로 dict/속성 접근 모두 지원
                # =====================================================
                batch = BatchData(
                    batch_idx=targets[:, 0].long(),
                    cls=targets[:, 1:2],
                    bboxes=targets[:, 2:6]
                )
                
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
        """
        Fallback: predictions 기반 dummy loss (gradient 연결용)
        
        [중요] 실제 학습에서는 이게 호출되면 안 됨!
        """
        print("[DetectionLoss] ⚠️ Using fallback loss - check your setup!")
        
        # predictions에서 gradient가 흐르는 dummy loss 생성
        if isinstance(predictions, (list, tuple)) and len(predictions) > 0:
            losses = []
            for p in predictions:
                if isinstance(p, torch.Tensor) and p.requires_grad:
                    losses.append(p.mean())
            
            if losses:
                dummy_loss = sum(losses) * 0.01
            else:
                dummy_loss = torch.tensor(0.1, device=device, requires_grad=True)
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
    print("=" * 60)
    print("Detection Loss 테스트")
    print("=" * 60)
    
    # BatchData 테스트
    print("\n[1] BatchData 테스트")
    batch = BatchData(
        batch_idx=torch.tensor([0, 1]),
        cls=torch.tensor([[0], [1]]),
        bboxes=torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.7, 0.15, 0.25]])
    )
    
    # dict 접근
    print(f"  batch['cls']: {batch['cls'].shape}")
    # 속성 접근
    print(f"  batch.cls: {batch.cls.shape}")
    # keys
    print(f"  batch.keys(): {list(batch.keys())}")
    print("  ✓ BatchData 테스트 통과!")
    
    try:
        from ultralytics import YOLO
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n[2] DetectionLoss 테스트 (device: {device})")
        
        # YOLO 로드
        yolo = YOLO("yolov8n.pt")
        yolo.model.to(device)
        
        # Loss 함수 생성
        loss_fn = DetectionLoss(yolo.model)
        
        # Dummy data
        images = torch.randn(2, 3, 640, 640, device=device)
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.2, 0.2],
            [1, 0, 0.3, 0.7, 0.15, 0.25],
        ], device=device)
        
        # Forward (training mode)
        yolo.model.train()
        preds = yolo.model(images)
        
        print(f"\n  Predictions type: {type(preds)}")
        if isinstance(preds, (list, tuple)):
            print(f"  Predictions length: {len(preds)}")
        
        # Loss 계산
        print("\n  Calculating loss...")
        loss_dict = loss_fn(preds, targets, images)
        
        print(f"\n  결과:")
        print(f"    Total Loss: {loss_dict['total'].item():.4f}")
        print(f"    Box Loss: {loss_dict['box_loss'].item():.4f}")
        print(f"    Cls Loss: {loss_dict['cls_loss'].item():.4f}")
        print(f"    DFL Loss: {loss_dict['dfl_loss'].item():.4f}")
        print(f"    requires_grad: {loss_dict['total'].requires_grad}")
        
        # Gradient 테스트
        print("\n  Gradient 테스트...")
        yolo.model.zero_grad()
        loss_dict['total'].backward()
        
        has_grad = False
        for name, param in yolo.model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        print(f"    Gradient flow: {'✓' if has_grad else '✗'}")
        
        # Fallback 아닌지 확인
        is_real_loss = loss_dict['box_loss'].item() > 0 or loss_dict['cls_loss'].item() > 0
        print(f"    Real detection loss: {'✓' if is_real_loss else '✗ (using fallback)'}")
        
        print("\n" + "=" * 60)
        print("✓ 테스트 완료!")
        print("=" * 60)
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()