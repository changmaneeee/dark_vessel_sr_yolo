"""
=============================================================================
yolo_wrapper.py - Ultralytics YOLO 완벽 통합 래퍼
=============================================================================

[역할 - 단일 책임 원칙(SRP)]
1. 모델 로드
2. Forward Pass (추론/학습)
3. Feature Extraction (P3/P4/P5)
4. Freeze/Unfreeze 관리

[제외된 기능]
- Loss 계산 → detection_loss.py로 이관
  이유: Loss 로직이 변경될 때 이 파일을 수정할 필요 없음

[지원 모델]
- YOLOv8 (yolov8n/s/m/l/x)
- YOLO11 (yolo11n/s/m/l/x)

[사용 예시]
# 모델 로드
wrapper = YOLOWrapper("yolo11s.pt")

# Feature 추출 (Arch 5-B용)
features = wrapper.extract_features(images)
p3, p4, p5 = features['p3'], features['p4'], features['p5']

# 추론
wrapper.eval()
detections = wrapper.predict(images)

# Loss 계산은 별도 모듈에서
from src.losses import DetectionLoss
loss_fn = DetectionLoss(wrapper.detection_model)
loss = loss_fn(preds, targets, images)

[참고 문서]
- Ultralytics Docs: https://docs.ultralytics.com/reference/utils/loss/
- GitHub: https://github.com/ultralytics/ultralytics
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path


class YOLOWrapper(nn.Module):
    """
    Ultralytics YOLO 래퍼 (Forward & Feature Extraction 전담)
    
    [책임]
    - 모델 실행 (Forward)
    - Feature 추출 (P3/P4/P5)
    - Freeze/Unfreeze 관리
    
    [비책임]
    - Loss 계산 → DetectionLoss 클래스에서 담당
    """
    
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        num_classes: int = None,  # None이면 모델 기본값 사용
        device: str = 'cuda',
        verbose: bool = False
    ):
        """
        Args:
            model_path: YOLO 모델 경로 또는 이름
                       - "yolov8n.pt", "yolov8s.pt", ...
                       - "yolo11n.pt", "yolo11s.pt", ...
                       - 커스텀 학습 모델 경로
            num_classes: 클래스 수 (None이면 모델 기본값)
            device: 실행 장치
            verbose: 로깅 출력 여부
        """
        super(YOLOWrapper, self).__init__()
        
        self.model_path = model_path
        self.device = device
        self.verbose = verbose
        
        # Feature 저장용 (hook에서 사용)
        self._features: Dict[str, torch.Tensor] = {}
        self._feature_channels: Optional[Dict[str, int]] = None
        
        # =====================================================================
        # Ultralytics YOLO 로드
        # =====================================================================
        print(f"[YOLOWrapper] Loading model: {model_path}")
        
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics 패키지가 필요합니다. "
                "pip install ultralytics"
            )
        
        # YOLO 고수준 래퍼 로드
        self.yolo = YOLO(model_path, verbose=verbose)
        
        # DetectionModel 추출 (실제 nn.Module)
        # YOLO.model이 DetectionModel 인스턴스
        self.detection_model = self.yolo.model
        
        # 클래스 수 설정
        detect_head = self.detection_model.model[-1]
        self.num_classes = num_classes if num_classes else detect_head.nc
        
        # Feature 레이어 인덱스 (Detect.f에서 가져옴)
        # 일반적으로 [15, 18, 21] (P3, P4, P5)
        self.feature_indices = detect_head.f
        print(f"[YOLOWrapper] Feature indices (P3, P4, P5): {self.feature_indices}")
        
        # stride 정보 (P3=8, P4=16, P5=32)
        self.strides = detect_head.stride.tolist()
        print(f"[YOLOWrapper] Strides: {self.strides}")
        
        # Device 이동
        self.detection_model = self.detection_model.to(device)
        
        # =====================================================================
        # Loss 함수 초기화
        # =====================================================================
        self._loss_fn = None  # Lazy initialization
        
        # Feature 채널 정보 (모델에 따라 다름)
        # 실제 forward로 확인 필요
        self._feature_channels = None
        
        print(f"[YOLOWrapper] ✓ Model loaded successfully")
        print(f"[YOLOWrapper]   - Classes: {self.num_classes}")
        print(f"[YOLOWrapper]   - Device: {device}")
    
    # =========================================================================
    # Forward Pass
    # =========================================================================
    
    def forward(self, x: torch.Tensor) ->Any:
        """
        Forward pass
        
        Args:
            x: 입력 이미지 [B, 3, H, W], 값 범위 0~1
            return_features: Feature도 함께 반환할지 여부
        
        Returns:
            training=True: raw predictions (list of tensors)
            training=False: decoded predictions
        """
        return self.detection_model(x)
    
    def forward_with_features(
            self,
            x: torch.Tensor
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """
        Forward + Feature 추출을 한번에

        Args:
            x: 입력 이미지 [B, 3, H, W]

        Returns: predictions, features_dict
        """

        features = self.extract_features(x, detach=False)
        preds = self.detection_model(x)
        return preds, features

    # =========================================================================
    # Feature Extraction(For Arch 5-B)
    # =========================================================================
    
    def extract_features(
        self, 
        x: torch.Tensor,
        detach: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        P3, P4, P5 Multi-scale Feature 추출
        
        [Ultralytics 내부 구조]
        - Detect head의 .f 속성: feature 레이어 인덱스 [15, 18, 21]
        - forward hook으로 중간 레이어 출력 캡처
        
        Args:
            x: 입력 이미지 [B, 3, H, W]
            detach: True면 gradient 끊음 (feature만 사용)
                   False면 gradient 유지 (joint training)
        
        Returns:
            features: {
                'p3': [B, C3, H/8, W/8],   # stride 8, 작은 객체
                'p4': [B, C4, H/16, W/16], # stride 16, 중간 객체
                'p5': [B, C5, H/32, W/32]  # stride 32, 큰 객체
            }
        
        [채널 수 (모델에 따라 다름)]
        - YOLOv8n/YOLO11n: C3=64, C4=128, C5=256
        - YOLOv8s/YOLO11s: C3=128, C4=256, C5=512
        - YOLOv8m/YOLO11m: C3=192, C4=384, C5=576
        """
        self._features = {}
        hooks = []
        
        # Hook 함수 생성
        def make_hook(name: str):
            def hook(module, input, output):
                if detach:
                    self._features[name] = output.detach()
                else:
                    self._features[name] = output
            return hook
        
        # Feature 레이어 인덱스
        # self.feature_indices = [15, 18, 21] (Detect.f에서 가져옴)
        feature_names = ['p3', 'p4', 'p5']
        
        try:
            # Hook 등록
            layers = self.detection_model.model
            for idx, name in zip(self.feature_indices, feature_names):
                if idx < len(layers):
                    hook = layers[idx].register_forward_hook(make_hook(name))
                    hooks.append(hook)
            
            # Forward pass
            _ = self.detection_model(x)
            
        finally:
            # Hook 제거 (메모리 누수 방지)
            for hook in hooks:
                hook.remove()
        
        if self._feature_channels is None and self._features:
            self._feature_channels = {
                name: feat.size(1) for name, feat in self._features.items()
            }
        
        return self._features
    
    def get_feature_channels(self) -> Dict[str, int]:
        """
        각 Feature level의 채널 수 반환
        
        Returns:
            {'p3': C3, 'p4': C4, 'p5': C5}
        """
        if self._feature_channels is None:
            # Dummy forward로 채널 확인
            dummy = torch.zeros(1, 3, 640, 640, device=self.device)
            self.extract_features(dummy, detach=True)
        
        return self._feature_channels.copy()
    
    # =========================================================================
    # Prediction (추론)
    # =========================================================================
    
    def predict(
        self,
        x: torch.Tensor,
        conf: float = 0.25,
        iou: float = 0.45
    ) -> List[Dict[str, torch.Tensor]]:
        """
        NMS 포함 추론
        
        Args:
            x: 입력 이미지 [B, 3, H, W]
            conf: Confidence threshold
            iou: NMS IoU threshold
        
        Returns:
            List of detection dicts for each image:
            [{
                'boxes': [N, 4] (xyxy format, pixel coordinates),
                'scores': [N],
                'classes': [N]
            }, ...]
        """
        self.eval()
        
        # Ultralytics predict 사용
        results = self.yolo.predict(
            source=x,
            conf=conf,
            iou=iou,
            verbose=self.verbose
        )
        
        # 결과 변환
        outputs = []
        for result in results:
            boxes = result.boxes
            outputs.append({
                'boxes': boxes.xyxy if boxes.xyxy.numel() > 0 else torch.zeros(0, 4, device=self.device),
                'scores': boxes.conf if boxes.conf.numel() > 0 else torch.zeros(0, device=self.device),
                'classes': boxes.cls if boxes.cls.numel() > 0 else torch.zeros(0, device=self.device)
            })
        
        return outputs
    
    # =========================================================================
    # Evaluation (평가)
    # =========================================================================
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        단일 배치에 대한 Detection 결과와 GT 비교
        
        Args:
            images: 입력 이미지 [B, 3, H, W]
            targets: GT [N, 6] = (batch_idx, class, x, y, w, h) normalized
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
        
        Returns:
            dict: {
                'predictions': 예측 결과,
                'targets': GT (pixel 좌표로 변환),
                'num_images': 배치 크기
            }
        """
        self.eval()
        
        with torch.no_grad():
            preds = self.predict(images, conf=conf_threshold, iou=iou_threshold)
        
        # GT 변환 (normalized xywh → pixel xyxy)
        batch_size = images.size(0)
        H, W = images.size(2), images.size(3)
        
        gt_per_image = []
        for b in range(batch_size):
            mask = targets[:, 0] == b
            gt_boxes_norm = targets[mask, 2:6]  # x, y, w, h (normalized)
            gt_classes = targets[mask, 1]
            
            # xywh normalized → xyxy pixel
            gt_xyxy = self._xywhn_to_xyxy(gt_boxes_norm, W, H)
            
            gt_per_image.append({
                'boxes': gt_xyxy,
                'classes': gt_classes
            })
        
        return {
            'predictions': preds,
            'targets': gt_per_image,
            'num_images': batch_size
        }
    
    def _xywhn_to_xyxy(
        self, 
        boxes: torch.Tensor, 
        img_w: int, 
        img_h: int
    ) -> torch.Tensor:
        """
        Normalized xywh → Pixel xyxy 변환
        
        Args:
            boxes: [N, 4] normalized (x_center, y_center, w, h)
            img_w, img_h: 이미지 크기
        
        Returns:
            [N, 4] pixel (x1, y1, x2, y2)
        """
        if boxes.numel() == 0:
            return torch.zeros((0, 4), device=boxes.device)
        
        x_center = boxes[:, 0] * img_w
        y_center = boxes[:, 1] * img_h
        w = boxes[:, 2] * img_w
        h = boxes[:, 3] * img_h
        
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    # =========================================================================
    # Freeze / Unfreeze
    # =========================================================================
    
    def freeze(self) -> None:
        """전체 모델 Freeze"""
        for param in self.detection_model.parameters():
            param.requires_grad = False
        print("[YOLOWrapper] ✓ Model frozen (all parameters)")
    
    def unfreeze(self) -> None:
        """전체 모델 Unfreeze"""
        for param in self.detection_model.parameters():
            param.requires_grad = True
        print("[YOLOWrapper] ✓ Model unfrozen (all parameters)")
    
    def freeze_backbone(self, num_layers: int = 10) -> None:
        """
        Backbone만 Freeze (Head는 학습)
        
        Args:
            num_layers: Freeze할 레이어 수 (기본 10)
        """
        for i, layer in enumerate(self.detection_model.model):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"[YOLOWrapper] ✓ Backbone frozen (layers 0-{num_layers-1})")
    
    def freeze_except_head(self) -> None:
        """Detect head를 제외한 모든 레이어 Freeze"""
        # 마지막 레이어(Detect)만 학습 가능
        for param in self.detection_model.parameters():
            param.requires_grad = False
        
        for param in self.detection_model.model[-1].parameters():
            param.requires_grad = True
        
        print("[YOLOWrapper] ✓ All layers frozen except Detect head")
    
    def set_bn_eval(self) -> None:
        """
        BatchNorm을 eval 모드로 설정
        
        [중요] requires_grad=False만으로는 BN의 running_mean/var 업데이트를 막을 수 없음!
        freeze할 때 이 메서드도 호출해야 함
        """
        for module in self.detection_model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
        print("[YOLOWrapper] ✓ BatchNorm layers set to eval mode")
    
    # =========================================================================
    # Utility
    # =========================================================================
    
    def count_parameters(self) -> Dict[str, int]:
        """파라미터 수 계산"""
        total = sum(p.numel() for p in self.detection_model.parameters())
        trainable = sum(p.numel() for p in self.detection_model.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_path': self.model_path,
            'num_classes': self.num_classes,
            'device': str(self.device),
            'feature_indices': self.feature_indices,
            'strides': self.strides,
            'parameters': self.count_parameters(),
            'feature_channels': self.get_feature_channels() if self._feature_channels else 'Not computed yet'
        }
    
    def train(self, mode: bool = True):
        """학습 모드 설정 (override)"""
        super().train(mode)
        self.detection_model.train(mode)
        return self
    
    def eval(self):
        """평가 모드 설정 (override)"""
        super().eval()
        self.detection_model.eval()
        return self


# =============================================================================
# 테스트 코드
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("YOLOWrapper 테스트 (SRP 적용 - Loss 제외)")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    try:
        wrapper = YOLOWrapper("yolov8n.pt", device=device)
        print(f"\n모델 정보: {wrapper.get_model_info()}")
        
        # Feature 추출 테스트
        print("\n[Feature 추출 테스트]")
        dummy = torch.randn(2, 3, 640, 640, device=device)
        features = wrapper.extract_features(dummy)
        for name, feat in features.items():
            print(f"  {name}: {feat.shape}")
        
        # Forward 테스트
        print("\n[Forward 테스트]")
        wrapper.train()
        preds = wrapper(dummy)
        print(f"  Output type: {type(preds)}")
        if isinstance(preds, list):
            print(f"  Output shapes: {[p.shape for p in preds]}")
        
        print("\n✓ YOLOWrapper 테스트 완료!")
        print("  (compute_loss 제거됨 - DetectionLoss 사용하세요)")
        
    except Exception as e:
        print(f"테스트 실패: {e}")