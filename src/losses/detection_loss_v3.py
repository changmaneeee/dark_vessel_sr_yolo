"""
=============================================================================
detection_loss.py - YOLO Detection Loss Wrapper (v3)
=============================================================================
Ultralytics YOLO v8/11의 Detection Loss를 래핑

[수정 v3]
- BatchData 클래스 개선 (모든 접근 방식 지원)
- 에러 디버깅 상세 출력
- Ultralytics 버전별 호환성
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class BatchData:
    """
    Ultralytics 호환 Batch 데이터 클래스
    
    모든 접근 방식 지원:
    - batch['cls']     → __getitem__
    - batch.cls        → __getattr__
    - batch.get('cls') → get method
    """
    
    def __init__(self, **kwargs):
        # 내부 저장소
        object.__setattr__(self, '_data', kwargs)
        # 속성으로도 접근 가능하게
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
        object.__setattr__(self, key, value)
    
    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        if key == '_data':
            object.__setattr__(self, key, value)
        else:
            self._data[key] = value
            object.__setattr__(self, key, value)
    
    def __contains__(self, key):
        return key in self._data
    
    def __len__(self):
        return len(self._data)
    
    def __repr__(self):
        return f"BatchData({self._data})"
    
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
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.loss_fn = None
        
        try:
            from ultralytics.utils.loss import v8DetectionLoss
            self.loss_fn = v8DetectionLoss(model)
            print("[DetectionLoss] ✓ v8DetectionLoss initialized")
        except ImportError as e:
            print(f"[DetectionLoss] ⚠️ v8DetectionLoss import failed: {e}")
    
    def forward(
        self,
        predictions: Any,
        targets: torch.Tensor,
        images: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Detection Loss 계산"""
        device = targets.device if targets is not None else 'cpu'
        
        # 빈 타겟 처리
        if targets is None or len(targets) == 0:
            return self._zero_loss(device)
        
        if self.loss_fn is None:
            return self._fallback_loss(predictions, targets, device)
        
        try:
            # =====================================================
            # BatchData 생성 - Ultralytics 호환
            # =====================================================
            batch = BatchData(
                batch_idx=targets[:, 0].long(),
                cls=targets[:, 1:2],
                bboxes=targets[:, 2:6]
            )
            
            # 디버깅: batch 내용 확인
            # print(f"[DEBUG] BatchData type: {type(batch)}")
            # print(f"[DEBUG] BatchData keys: {list(batch.keys())}")
            # print(f"[DEBUG] batch.cls shape: {batch.cls.shape}")
            # print(f"[DEBUG] batch['bboxes'] shape: {batch['bboxes'].shape}")
            
            # Loss 계산
            loss, loss_items = self.loss_fn(predictions, batch)
            
            return {
                'total': loss,
                'box_loss': loss_items[0] if len(loss_items) > 0 else torch.tensor(0.0, device=device),
                'cls_loss': loss_items[1] if len(loss_items) > 1 else torch.tensor(0.0, device=device),
                'dfl_loss': loss_items[2] if len(loss_items) > 2 else torch.tensor(0.0, device=device)
            }
            
        except AttributeError as e:
            # 속성 에러 - 어떤 속성이 없는지 출력
            print(f"[DetectionLoss] AttributeError: {e}")
            print(f"[DetectionLoss] → Ultralytics가 기대하는 속성이 BatchData에 없음")
            print(f"[DetectionLoss] → 사용 가능한 속성: {list(batch.keys()) if 'batch' in dir() else 'N/A'}")
            return self._fallback_loss(predictions, targets, device)
            
        except TypeError as e:
            # 타입 에러 - subscript 문제 등
            print(f"[DetectionLoss] TypeError: {e}")
            return self._fallback_loss(predictions, targets, device)
            
        except Exception as e:
            print(f"[DetectionLoss] Unexpected error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_loss(predictions, targets, device)
    
    def _zero_loss(self, device) -> Dict[str, torch.Tensor]:
        """빈 타겟용 zero loss"""
        return {
            'total': torch.tensor(0.0, device=device, requires_grad=True),
            'box_loss': torch.tensor(0.0, device=device),
            'cls_loss': torch.tensor(0.0, device=device),
            'dfl_loss': torch.tensor(0.0, device=device)
        }
    
    def _fallback_loss(
        self,
        predictions: Any,
        targets: torch.Tensor,
        device
    ) -> Dict[str, torch.Tensor]:
        """Fallback loss - gradient 연결용"""
        print("[DetectionLoss] ⚠️ Using fallback loss - check your setup!")
        
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
# 테스트 및 디버깅
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DetectionLoss 테스트 및 디버깅")
    print("=" * 70)
    
    # 1. BatchData 테스트
    print("\n[1] BatchData 테스트")
    batch = BatchData(
        batch_idx=torch.tensor([0, 1]),
        cls=torch.tensor([[0], [1]]),
        bboxes=torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.7, 0.15, 0.25]])
    )
    
    print(f"  type(batch): {type(batch)}")
    print(f"  batch.cls: {batch.cls.shape}")
    print(f"  batch['cls']: {batch['cls'].shape}")
    print(f"  batch.get('cls'): {batch.get('cls').shape}")
    print(f"  'cls' in batch: {'cls' in batch}")
    print(f"  batch.keys(): {list(batch.keys())}")
    print("  ✓ BatchData 테스트 통과!")
    
    # 2. Ultralytics 분석
    print("\n[2] Ultralytics v8DetectionLoss 분석")
    try:
        from ultralytics.utils.loss import v8DetectionLoss
        import inspect
        
        # __call__ 메서드 소스 확인
        source = inspect.getsource(v8DetectionLoss.__call__)
        
        print("  batch 관련 코드:")
        for line in source.split('\n'):
            if 'batch' in line and ('batch.' in line or 'batch[' in line):
                print(f"    {line.strip()}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # 3. 실제 Loss 계산 테스트
    print("\n[3] Loss 계산 테스트")
    try:
        from ultralytics import YOLO
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        yolo = YOLO("yolov8n.pt")
        yolo.model.to(device)
        
        loss_fn = DetectionLoss(yolo.model)
        
        images = torch.randn(2, 3, 640, 640, device=device)
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.2, 0.2],
            [1, 0, 0.3, 0.7, 0.15, 0.25],
        ], device=device)
        
        yolo.model.train()
        preds = yolo.model(images)
        
        loss_dict = loss_fn(preds, targets, images)
        
        print(f"\n  결과:")
        print(f"    Total: {loss_dict['total'].item():.4f}")
        print(f"    Box: {loss_dict['box_loss'].item():.4f}")
        print(f"    Cls: {loss_dict['cls_loss'].item():.4f}")
        print(f"    DFL: {loss_dict['dfl_loss'].item():.4f}")
        
        is_real = loss_dict['box_loss'].item() > 0 or loss_dict['cls_loss'].item() > 0
        print(f"\n    Real loss (not fallback): {'✓' if is_real else '✗'}")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)