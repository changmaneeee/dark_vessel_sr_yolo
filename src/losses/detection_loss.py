"""
=============================================================================
detection_loss.py - Detection Loss 전담 모듈
=============================================================================

[역할 - 단일 책임 원칙(SRP)]
- YOLO Detection Loss 계산만 담당
- Ultralytics v8DetectionLoss 래핑
- batch dictionary 변환 (Adapter 역할)

[v8DetectionLoss 구성]
L_total = λ_box(7.5) × L_box + λ_cls(0.5) × L_cls + λ_dfl(1.5) × L_dfl

- L_box: CIoU Loss (Complete IoU)
- L_cls: BCE with Focal Loss
- L_dfl: Distribution Focal Loss

[사용 예시]
from src.losses import DetectionLoss
from src.models.detectors import YOLOWrapper

# 모델 로드
wrapper = YOLOWrapper("yolo11s.pt")

# Loss 함수 생성 (Wrapper의 모델 전달)
loss_fn = DetectionLoss(wrapper.detection_model)

# 학습 루프
wrapper.train()
preds = wrapper(images)  # raw predictions
loss_dict = loss_fn(preds, targets, images)
loss_dict['total'].backward()
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union


class DetectionLoss(nn.Module):
    """
    YOLO Detection Loss (v8DetectionLoss 래퍼)
    
    [책임]
    - v8DetectionLoss 초기화
    - targets → batch dictionary 변환 (Adapter)
    - Loss 계산 및 반환
    
    [batch dictionary 형식 - Ultralytics 요구사항]
    batch = {
        'batch_idx': [N] float32 - 각 객체의 이미지 인덱스,
        'cls': [N, 1] float32 - 클래스 레이블,
        'bboxes': [N, 4] float32 - normalized xywh,
        'img': [B, 3, H, W] - 이미지
    }
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: DetectionModel (YOLO의 실제 nn.Module)
                  - YOLOWrapper.detection_model
                  - YOLO("model.pt").model
        """
        super().__init__()
        
        self.model = model
        self._loss_fn = None  # Lazy initialization
        self._initialized = False
        
        # Device 추론
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _initialize(self) -> None:
        """v8DetectionLoss Lazy Initialization"""
        if self._initialized:
            return
        
        try:
            from ultralytics.utils.loss import v8DetectionLoss
            self._loss_fn = v8DetectionLoss(self.model)
            self._initialized = True
            print("[DetectionLoss] ✓ v8DetectionLoss initialized")
        except ImportError as e:
            raise ImportError(
                f"v8DetectionLoss import 실패: {e}\n"
                "ultralytics 설치 필요: pip install ultralytics"
            )
    
    def forward(
        self,
        preds: Union[torch.Tensor, list],
        targets: torch.Tensor,
        images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Detection Loss 계산
        
        Args:
            preds: YOLO raw output (training mode)
                  - list[Tensor]: [P3_pred, P4_pred, P5_pred]
                  - 각 텐서: [B, reg_max*4 + nc, H, W]
            targets: Ground truth - YOLO 형식
                    [N, 6] = (batch_idx, class, x_center, y_center, w, h)
                    좌표는 0~1 normalized
            images: 입력 이미지 [B, 3, H, W]
                   (batch 크기 확인용)
        
        Returns:
            loss_dict: {
                'total': 전체 loss (backward용),
                'box_loss': Box regression loss,
                'cls_loss': Classification loss,
                'dfl_loss': Distribution Focal Loss
            }
        
        [중요]
        - model.train() 상태에서 호출해야 함
        - preds는 model(images)의 training mode 출력이어야 함
        """
        self._initialize()
        
        # 학습 모드가 아니면 0 반환
        if not self.model.training:
            return self._zero_loss()
        
        # 타겟이 없으면 0 반환
        if targets is None or len(targets) == 0:
            return self._zero_loss(requires_grad=True)
        
        # Device 이동
        targets = targets.to(self.device)
        images = images.to(self.device)
        
        # =====================================================================
        # Batch Dictionary 생성 (Adapter 역할)
        # =====================================================================
        # Ultralytics v8DetectionLoss가 요구하는 형식으로 변환
        batch = self._create_batch_dict(targets, images)
        
        # =====================================================================
        # Loss 계산
        # =====================================================================
        try:
            total_loss, loss_items = self._loss_fn(preds, batch)
            
            return {
                'total': total_loss,
                'box_loss': loss_items[0] if len(loss_items) > 0 else torch.tensor(0.0, device=self.device),
                'cls_loss': loss_items[1] if len(loss_items) > 1 else torch.tensor(0.0, device=self.device),
                'dfl_loss': loss_items[2] if len(loss_items) > 2 else torch.tensor(0.0, device=self.device)
            }
            
        except Exception as e:
            print(f"[DetectionLoss] Warning: Loss computation failed: {e}")
            return self._zero_loss(requires_grad=True)
    
    def _create_batch_dict(
        self, 
        targets: torch.Tensor, 
        images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        YOLO 형식 targets를 Ultralytics batch dictionary로 변환
        
        Args:
            targets: [N, 6] = (batch_idx, class, x, y, w, h)
            images: [B, 3, H, W]
        
        Returns:
            batch: Ultralytics v8DetectionLoss가 요구하는 형식
        
        [Ultralytics 내부 요구사항]
        - batch_idx: float32
        - cls: float32, shape [N, 1] 또는 [N]
        - bboxes: float32, normalized xywh
        - img: 이미지 텐서
        """
        return {
            'batch_idx': targets[:, 0].float(),        # [N]
            'cls': targets[:, 1].float().view(-1, 1),  # [N, 1]
            'bboxes': targets[:, 2:6].float(),         # [N, 4]
            'img': images,                              # [B, 3, H, W]
        }
    
    def _zero_loss(self, requires_grad: bool = False) -> Dict[str, torch.Tensor]:
        """0 loss 반환 (평가 모드 또는 타겟 없을 때)"""
        zero = torch.tensor(0.0, device=self.device, requires_grad=requires_grad)
        return {
            'total': zero,
            'box_loss': torch.tensor(0.0, device=self.device),
            'cls_loss': torch.tensor(0.0, device=self.device),
            'dfl_loss': torch.tensor(0.0, device=self.device)
        }
    
    def compute_from_images(
        self,
        model: nn.Module,
        images: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        편의 메서드: 이미지에서 바로 Loss 계산
        
        내부에서 forward → loss 계산
        
        Args:
            model: YOLO DetectionModel (또는 Wrapper)
            images: 입력 이미지 [B, 3, H, W]
            targets: GT [N, 6]
        
        Returns:
            loss_dict
        """
        model.train()
        preds = model(images)
        return self.forward(preds, targets, images)


class DetectionLossWithModel(nn.Module):
    """
    모델을 내장한 Detection Loss
    
    모델 forward + loss 계산을 한번에
    (compute_from_images의 모듈 버전)
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.loss_fn = DetectionLoss(model)
    
    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        이미지 → Forward → Loss
        
        Args:
            images: [B, 3, H, W]
            targets: [N, 6]
        
        Returns:
            loss_dict
        """
        self.model.train()
        preds = self.model(images)
        return self.loss_fn(preds, targets, images)


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DetectionLoss 테스트")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    try:
        from ultralytics import YOLO
        
        # 1. 모델 로드
        print("\n[1. 모델 로드]")
        yolo = YOLO("yolov8n.pt")
        model = yolo.model.to(device)
        print("✓ YOLO 모델 로드 완료")
        
        # 2. DetectionLoss 생성
        print("\n[2. DetectionLoss 생성]")
        loss_fn = DetectionLoss(model)
        print("✓ DetectionLoss 생성 완료")
        
        # 3. 더미 데이터
        print("\n[3. 더미 데이터 생성]")
        images = torch.randn(2, 3, 640, 640, device=device)
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.2, 0.2],   # 이미지 0, 클래스 0
            [0, 0, 0.3, 0.7, 0.15, 0.15], # 이미지 0, 클래스 0
            [1, 0, 0.6, 0.4, 0.25, 0.25], # 이미지 1, 클래스 0
        ], device=device)
        print(f"  Images: {images.shape}")
        print(f"  Targets: {targets.shape}")
        
        # 4. Forward + Loss 계산
        print("\n[4. Loss 계산]")
        model.train()
        preds = model(images)
        print(f"  Preds type: {type(preds)}")
        if isinstance(preds, list):
            print(f"  Preds shapes: {[p.shape for p in preds]}")
        
        loss_dict = loss_fn(preds, targets, images)
        
        print("\n  Loss 결과:")
        for name, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"    {name}: {value.item():.6f}")
        
        # 5. Gradient 확인
        print("\n[5. Gradient 확인]")
        if loss_dict['total'].requires_grad:
            loss_dict['total'].backward()
            print("  ✓ Backward 성공!")
            
            # 파라미터 gradient 확인
            has_grad = False
            for param in model.parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grad = True
                    break
            
            if has_grad:
                print("  ✓ Gradient가 모델 파라미터로 흐름!")
            else:
                print("  ⚠ Gradient가 없음 (freeze 상태?)")
        else:
            print("  ✗ requires_grad=False")
        
        print("\n" + "=" * 70)
        print("✓ DetectionLoss 테스트 완료!")
        print("=" * 70)
        
    except ImportError:
        print("ultralytics 미설치 - 테스트 스킵")
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()