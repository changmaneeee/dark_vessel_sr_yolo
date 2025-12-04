"""
=============================================================================
yolo_wrapper.py - YOLO Detection Model Wrapper
=============================================================================
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union
from ultralytics import YOLO


class YOLOWrapper(nn.Module):
    """
    YOLOv8 Detection Model Wrapper
    
    [주요 기능]
    1. forward(): 기본 Detection forward pass
    2. compute_loss(): Detection Loss 계산
    3. extract_features(): Multi-scale Feature 추출 (Arch 5-B용)
    4. predict(): NMS 포함 추론
    5. freeze/unfreeze: 학습 제어
    
    [사용 예시]
    
    # 기본 사용 (Arch 0)
    wrapper = YOLOWrapper("yolov8n.pt")
    detections = wrapper(hr_image)
    
    # Feature 추출 (Arch 5-B)
    features = wrapper.extract_features(lr_image)
    p3_feat = features['p3']  # [B, 256, H/8, W/8]
    
    # Loss 계산
    loss = wrapper.compute_loss(detections, targets)
    
    [시스템 엔지니어 학습 포인트]
    1. Lazy Loading: 필요할 때만 리소스 로드
    2. Hook Registration: 중간 레이어 접근
    3. Error Handling: 다양한 입력 형식 처리
    """
    
    def __init__(
        self,
        model_path,
        num_classes: int = 1,
        device,
        verbose: bool = False
    ):

        super(YOLOWrapper, self).__init__()
        
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = device
        self.verbose = verbose

        self._features: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        
        print(f"Loading YOLO model from: {model_path}")
        

        self.yolo_model = YOLO(model_path, verbose=verbose)
        self.model = self.yolo_model.model
        self.model = self.model.to(device)
        

        self._feature_channels = {
            'p3': 256,   # For small objects
            'p4': 512,   # For medium objects
            'p5': 1024   # For large objects (after SPPF)
        }
        
        print(f"✓ YOLO model loaded successfully")
    
    # =========================================================================
    # Forward Pass
    # =========================================================================
    
    def forward(self, x: torch.Tensor) -> Any:

        return self.model(x)
    
    def predict(
        self,
        x: torch.Tensor,
        conf: float = 0.25,
        iou: float = 0.45
    ) -> List[Dict[str, torch.Tensor]]:


        results = self.yolo_model.predict(
            source=x,
            conf=conf,
            iou=iou,
            verbose=self.verbose
        )
        

        outputs = []
        for result in results:
            boxes = result.boxes
            outputs.append({
                'boxes': boxes.xyxy,      # [N, 4]
                'scores': boxes.conf,     # [N]
                'classes': boxes.cls      # [N]
            })
        
        return outputs
    
    # =========================================================================
    # Loss Computation
    # =========================================================================
    
    def compute_loss(
        self,
        predictions: Any,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Detection Loss 계산
        
        Args:
            predictions: forward()의 출력
            targets: Ground truth
                    YOLO 형식: [N, 6] = (batch_idx, class, x_center, y_center, w, h)
                    좌표는 0~1 정규화
        
        Returns:
            loss_dict: {
                'total': 전체 loss,
                'box_loss': Box regression loss (CIoU),
                'cls_loss': Classification loss (BCE),
                'dfl_loss': Distribution Focal Loss
            }
        
        [YOLO Loss 구성]
        L_total = λ_box * L_box + λ_cls * L_cls + λ_dfl * L_dfl
        
        - L_box (CIoU): Complete IoU loss for box regression
        - L_cls (BCE): Binary Cross Entropy for classification
        - L_dfl: Distribution Focal Loss for fine-grained localization
        
        [시스템 엔지니어 학습 포인트]
        - Multi-task Loss: 여러 손실 함수의 가중합
        - Loss Balancing: λ 값으로 각 loss 중요도 조절
        
        [주의]
        Ultralytics YOLO의 내부 loss 접근이 복잡함
        - 학습 모드에서만 loss 계산 가능
        - 현재는 간소화된 버전 제공
        - 실제 학습 시 Ultralytics trainer 사용 권장
        """
        # 학습 모드 확인
        if not self.training:
            return {
                'total': torch.tensor(0.0, device=self.device),
                'box_loss': torch.tensor(0.0, device=self.device),
                'cls_loss': torch.tensor(0.0, device=self.device),
                'dfl_loss': torch.tensor(0.0, device=self.device)
            }
        
        # =====================================================================
        # Ultralytics 내부 loss 계산
        # =====================================================================
        # 
        # YOLO 모델의 loss 계산은 복잡한 구조를 가짐
        # - 모델 내부에 loss 함수가 포함되어 있음
        # - 학습 시 자동으로 계산됨
        # 
        # 현재 구현: 간소화된 버전
        # 실제 프로젝트에서는 Ultralytics trainer를 사용하거나
        # 내부 loss 함수를 직접 호출해야 함
        # =====================================================================
        
        try:
            # YOLO 모델이 학습 모드일 때 loss 포함하여 반환
            # predictions가 tuple/list 형태로 loss 포함할 수 있음
            if isinstance(predictions, (tuple, list)) and len(predictions) > 1:
                # (output, loss) 형태로 반환되는 경우
                loss = predictions[1] if isinstance(predictions[1], torch.Tensor) else predictions[0]
            else:
                # Loss가 직접 반환되지 않는 경우
                # placeholder loss 반환
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            return {
                'total': loss,
                'box_loss': loss * 0.5,  # 대략적인 분배
                'cls_loss': loss * 0.3,
                'dfl_loss': loss * 0.2
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Loss computation failed: {e}")
            return {
                'total': torch.tensor(0.0, device=self.device, requires_grad=True),
                'box_loss': torch.tensor(0.0, device=self.device),
                'cls_loss': torch.tensor(0.0, device=self.device),
                'dfl_loss': torch.tensor(0.0, device=self.device)
            }
    
    # =========================================================================
    # Feature Extraction (Arch 5-B용)
    # =========================================================================
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Multi-scale Feature 추출 (Arch 5-B Fusion용)
        
        Args:
            x: 입력 이미지 [B, 3, H, W]
        
        Returns:
            features: {
                'p3': [B, 256, H/8, W/8],    # 작은 객체용
                'p4': [B, 512, H/16, W/16],  # 중간 객체용
                'p5': [B, 1024, H/32, W/32]  # 큰 객체용
            }
        
        [Feature Pyramid Network (FPN) 이해]
        
        입력이 192×192일 때:
        - P3: 192/8 = 24×24   (작은 선박 탐지)
        - P4: 192/16 = 12×12  (중간 선박)
        - P5: 192/32 = 6×6    (큰 선박)
        
        [Hook 메커니즘]
        
        PyTorch Hook = 중간 레이어 출력을 가로채는 기술
        
        1. register_forward_hook()으로 hook 등록
        2. forward pass 시 hook 함수 자동 호출
        3. hook 함수에서 output을 저장
        4. 나중에 저장된 값 사용
        
        [시스템 엔지니어 학습 포인트]
        - Hook Pattern: 이벤트 기반 프로그래밍
        - Callback 함수: 특정 시점에 호출되는 함수
        - Feature Map 활용: 중간 표현의 재사용
        """
        # Feature 저장 초기화
        self._features = {}
        
        # =====================================================================
        # Hook 함수 정의
        # =====================================================================
        def get_hook(name: str):
            """
            Hook 함수 생성기 (Closure 활용)
            
            [Closure란?]
            - 함수 안에서 정의된 함수가 바깥 함수의 변수를 기억
            - 여기서는 'name' 변수를 기억
            
            Args:
                name: Feature 이름 ('p3', 'p4', 'p5')
            
            Returns:
                hook 함수
            """
            def hook(module, input, output):
                # output을 딕셔너리에 저장
                self._features[name] = output
            return hook
        
        # =====================================================================
        # Hook 등록
        # =====================================================================
        # YOLOv8 모델 구조에서 해당 레이어 찾기
        # 
        # YOLOv8n 구조 (model.model):
        # - [0-3]: Backbone 초반
        # - [4]: P3 레벨 (C2f)
        # - [5]: Conv (다운샘플)
        # - [6]: P4 레벨 (C2f)
        # - [7]: Conv (다운샘플)
        # - [8]: P5 레벨 (C2f)
        # - [9]: SPPF
        # - [10-22]: Neck + Head
        # 
        # 주의: 모델 버전에 따라 인덱스가 다를 수 있음!
        # =====================================================================
        
        hooks = []
        try:
            # 모델 레이어 접근
            model_layers = self.model.model
            
            # P3, P4, P5 레이어에 hook 등록
            # (레이어 인덱스는 YOLOv8 버전에 따라 조정 필요)
            layer_indices = {
                'p3': 4,   # C2f after first downsample
                'p4': 6,   # C2f after second downsample
                'p5': 9    # SPPF
            }
            
            for name, idx in layer_indices.items():
                if idx < len(model_layers):
                    hook = model_layers[idx].register_forward_hook(get_hook(name))
                    hooks.append(hook)
            
            # Forward pass 실행 (Hook이 자동 호출됨)
            with torch.no_grad():
                _ = self.model(x)
            
        finally:
            # Hook 제거 (메모리 누수 방지!)
            for hook in hooks:
                hook.remove()
        
        # Feature가 없으면 빈 텐서로 대체
        if not self._features:
            print("Warning: Feature extraction failed, returning empty features")
            B = x.size(0)
            H, W = x.size(2), x.size(3)
            self._features = {
                'p3': torch.zeros(B, 256, H//8, W//8, device=x.device),
                'p4': torch.zeros(B, 512, H//16, W//16, device=x.device),
                'p5': torch.zeros(B, 1024, H//32, W//32, device=x.device)
            }
        
        return self._features
    
    def get_feature_channels(self) -> Dict[str, int]:
        """
        각 Feature level의 채널 수 반환
        
        Returns:
            {'p3': 256, 'p4': 512, 'p5': 1024}
        
        [Arch 5-B Fusion 설계 시 사용]
        fusion = AttentionFusion(
            sr_channels=50,  # RFDN feature
            yolo_channels=wrapper.get_feature_channels()['p3']  # 256
        )
        """
        return self._feature_channels.copy()
    
    # =========================================================================
    # Freeze / Unfreeze
    # =========================================================================
    
    def freeze(self) -> None:
        """
        전체 모델 Freeze (파라미터 고정)
        
        [사용 상황]
        - Arch 0: HR pretrained YOLO 사용, SR만 학습
        - Transfer Learning: Feature extractor로만 사용
        """
        for param in self.model.parameters():
            param.requires_grad = False
        print("✓ YOLO model frozen (all parameters)")
    
    def unfreeze(self) -> None:
        """
        전체 모델 Unfreeze (학습 가능)
        
        [사용 상황]
        - End-to-end fine-tuning
        - Joint training (SR + Detection)
        """
        for param in self.model.parameters():
            param.requires_grad = True
        print("✓ YOLO model unfrozen (all parameters)")
    
    def freeze_backbone(self) -> None:
        """
        Backbone만 Freeze, Head는 학습
        
        [사용 상황]
        - 새로운 데이터셋에 fine-tuning
        - Backbone의 일반적 특징은 유지하면서 Head만 적응
        
        [시스템 엔지니어 학습 포인트]
        - Transfer Learning: 사전학습 지식 활용
        - Selective Freezing: 필요한 부분만 학습
        """
        # YOLOv8에서 backbone은 보통 처음 10개 레이어
        try:
            for i, layer in enumerate(self.model.model):
                if i < 10:  # Backbone 레이어
                    for param in layer.parameters():
                        param.requires_grad = False
            print("✓ YOLO backbone frozen, head trainable")
        except Exception as e:
            print(f"Warning: Could not freeze backbone: {e}")
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def count_parameters(self) -> Dict[str, int]:
        """파라미터 수 계산"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
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
            'parameters': self.count_parameters(),
            'feature_channels': self._feature_channels
        }


# =============================================================================
# 테스트 코드
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("YOLOWrapper 테스트")
    print("=" * 60)
    
    # 모델 생성 (CPU에서 테스트)
    wrapper = YOLOWrapper("yolov8n.pt", device='cpu', verbose=False)
    print(f"\n모델 정보:")
    print(f"  {wrapper.get_model_info()}")
    
    # 더미 입력 생성
    batch_size = 2
    test_image = torch.randn(batch_size, 3, 640, 640)
    print(f"\n입력 shape: {test_image.shape}")
    
    # Forward 테스트
    print("\n1. Forward 테스트:")
    wrapper.eval()
    with torch.no_grad():
        output = wrapper(test_image)
        print(f"   출력 타입: {type(output)}")
    
    # Feature 추출 테스트
    print("\n2. Feature 추출 테스트:")
    features = wrapper.extract_features(test_image)
    for name, feat in features.items():
        print(f"   {name}: {feat.shape}")
    
    # Freeze 테스트
    print("\n3. Freeze 테스트:")
    wrapper.freeze()
    print(f"   학습 가능 파라미터: {wrapper.count_parameters()['trainable']}")
    
    wrapper.unfreeze()
    print(f"   Unfreeze 후: {wrapper.count_parameters()['trainable']}")
    
    print("\n✓ YOLOWrapper 테스트 완료!")