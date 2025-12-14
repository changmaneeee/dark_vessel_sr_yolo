"""
=============================================================================
arch0_sequential.py - Architecture 0: Sequential Pipeline
=============================================================================

[역할]
- 가장 단순한 SR-Detection 파이프라인
- 다른 아키텍처의 "기준점(Baseline)" 역할
- 코드 구조의 "Reference Implementation"

[파이프라인 흐름]

LR Image (192×192, 6m GSD)
    │
    │  "저해상도라 선박이 2~5 픽셀밖에 안 돼..."
    │
    ▼
┌─────────┐
│  RFDN   │  Super-Resolution (×4 확대)
│  (SR)   │  
└────┬────┘
     │
     │  "4배 확대해서 8~20 픽셀로!"
     │
     ▼
HR Image (768×768, 1.5m GSD)
     │
     │  "이제 YOLO가 찾을 수 있어"
     │
     ▼
┌─────────┐
│  YOLO   │  Object Detection
│  v8     │  
└────┬────┘
     │
     ▼
Detection Results
    - boxes: [N, 4] (xyxy)
    - scores: [N]
    - classes: [N]

[학습 전략: A방식 (개별 학습 후 연결)]

1. RFDN 사전학습 (DIV2K → Airbus fine-tune)
   - Input: LR 이미지, Target: HR 이미지
   - Loss: L1 + Perceptual

2. YOLO 사전학습 (COCO → Airbus fine-tune)
   - Input: HR 이미지, Target: Bounding boxes
   - Loss: YOLO Loss (Box + Cls + DFL)

3. 파이프라인 연결
   - RFDN: 사전학습 가중치 로드
   - YOLO: 사전학습 가중치 로드 + Freeze
   - (선택) SR만 추가 fine-tune

[성능 기대치]
- SR 품질: PSNR ~28dB, SSIM ~0.85 (Airbus 기준)
- Detection: mAP@0.5 ~0.7 (HR에서의 성능 유지)
- 속도: ~200ms/image (Jetson Xavier NX)

[다른 아키텍처와 비교]
- Arch 0: 최고 성능, 가장 느림 (SR 전체 수행)
- Arch 2: 약간 낮은 성능, 빠름 (Gate로 SR 스킵)
- Arch 4: 비슷한 성능, 중간 속도 (2-pass)
- Arch 5-B: 비슷한 성능, 빠름 (Feature Fusion)

[시스템 엔지니어 학습 포인트]
1. 추상 클래스 구현: BasePipeline의 abstract method 구현
2. Dependency Injection: SR, YOLO를 config로 주입
3. Loss Composition: 여러 loss의 가중합
4. Freeze/Unfreeze 전략: Transfer Learning의 핵심
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

# Base 클래스와 모델들 import
from src.models.pipelines.base_pipeline import BasePipeline
from src.models.sr_models.rfdn import RFDN
from src.models.detectors.yolo_wrapper import YOLOWrapper


class Arch0Sequential(BasePipeline):
    """
    Architecture 0: Sequential SR-Detection Pipeline
    
    [상속]
    BasePipeline을 상속하여 표준 인터페이스 준수:
    - forward(): LR → (SR → HR → YOLO) → Detection
    - compute_loss(): SR Loss + Detection Loss
    
    [구성 요소]
    - sr_model (RFDN): 저해상도 → 고해상도 변환
    - detector (YOLOWrapper): 고해상도에서 객체 탐지
    
    [Config 구조]
    config:
      model:
        rfdn:
          nf: 50           # Feature 채널 수
          num_modules: 4   # RFDB 블록 수
        yolo:
          weights_path: "yolov8n.pt"  # YOLO 모델 경로
          num_classes: 1              # 클래스 수 (선박=1)
      data:
        upscale_factor: 4  # SR 배율
        lr_size: 192       # LR 이미지 크기
        hr_size: 768       # HR 이미지 크기
      training:
        sr_weight: 0.5         # SR Loss 가중치 (α)
        det_weight: 0.5        # Detection Loss 가중치 (β)
        freeze_detector: true  # YOLO Freeze 여부
    
    [사용 예시]
    
    # 1. Config 로드
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/arch0_sequential.yaml")
    
    # 2. 모델 생성
    model = Arch0Sequential(config)
    model = model.to('cuda')
    
    # 3. 학습
    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=1e-4)
    
    for batch in dataloader:
        lr, hr_gt, targets = batch
        
        # Forward
        sr_image, detections = model(lr)
        
        # Loss
        loss_dict = model.compute_loss((sr_image, detections), targets, hr_gt)
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()
    
    # 4. 추론
    model.eval()
    results = model.inference(test_lr)
    """
    
    def __init__(self, config: Any):
        """
        Args:
            config: 설정 객체 (OmegaConf 또는 dict-like)
        
        [초기화 순서]
        1. BasePipeline 초기화 (config 저장, loss weights 설정)
        2. RFDN (SR 모델) 생성
        3. YOLOWrapper (Detector) 생성
        4. (옵션) Detector Freeze
        5. (옵션) Pretrained weights 로드
        """
        # =====================================================================
        # 1. BasePipeline 초기화
        # =====================================================================
        super().__init__(config)
        
        def get_val(obj, key, default=None):
            """SimpleNamespace와 dict 둘 다 지원"""
            if hasattr(obj, key):
                return getattr(obj, key)
            elif isinstance(obj, dict):
                return obj.get(key, default)
            return default
        
        # =====================================================================
        # Config 파싱
        # =====================================================================
        model_config = get_val(config, 'model', config)
        data_config = get_val(config, 'data', SimpleNamespace())
        training_config = get_val(config, 'training', SimpleNamespace())
        
        # RFDN 설정
        rfdn_config = get_val(model_config, 'rfdn', SimpleNamespace())
        self.nf = get_val(rfdn_config, 'nf', 50)
        self.num_modules = get_val(rfdn_config, 'num_modules', 4)
        
        # YOLO 설정
        yolo_config = get_val(model_config, 'yolo', SimpleNamespace())
        self.yolo_weights = get_val(yolo_config, 'weights_path', 'yolov8n.pt')
        self.num_classes = get_val(yolo_config, 'num_classes', 1)
        
        # Data 설정
        self.upscale_factor = get_val(data_config, 'upscale_factor', 4)
        
        # Training 설정
        self.freeze_detector_flag = get_val(training_config, 'freeze_detector', True)
        self._sr_weight = get_val(training_config, 'sr_weight', 0.5)
        self._det_weight = get_val(training_config, 'det_weight', 0.5)
        # =====================================================================
        # 2. SR 모델 (RFDN) 생성
        # =====================================================================
        print(f"\n[Arch0] Initializing RFDN...")
        print(f"  - nf (feature channels): {self.nf}")
        print(f"  - num_modules (RFDB blocks): {self.num_modules}")
        print(f"  - upscale factor: {self.upscale_factor}")
        
        self.sr_model = RFDN(
            in_channels=3,
            out_channels=3,
            nf=self.nf,
            num_modules=self.num_modules,
            upscale=self.upscale_factor
        )
        
        # =====================================================================
        # 3. YOLO Detector 생성
        # =====================================================================
        print(f"\n[Arch0] Initializing YOLO...")
        print(f"  - weights: {self.yolo_weights}")
        print(f"  - num_classes: {self.num_classes}")
        
        self.detector = YOLOWrapper(
            model_path=self.yolo_weights,
            num_classes=self.num_classes,
            device=self.device,
            verbose=False
        )
        
        # =====================================================================
        # 4. Detector Freeze (옵션)
        # =====================================================================
        # [왜 Freeze하는가?]
        # - YOLO는 이미 HR 이미지로 학습됨
        # - SR이 HR에 가까운 출력을 내면 YOLO는 잘 동작
        # - YOLO를 함께 학습하면 오히려 성능 저하 가능
        #   (SR의 artifact에 적응해버림)
        
        if self.freeze_detector_flag:
            self.detector.freeze()
            print("✓ YOLO detector frozen (using pretrained weights)")
        else:
            print("✓ YOLO detector trainable (joint training)")
        
        # =====================================================================
        # 모델 정보 출력
        # =====================================================================
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[Arch0] Model Summary:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - SR weight (α): {self._sr_weight}")
        print(f"  - Det weight (β): {self._det_weight}")
    
    # =========================================================================
    # Forward Pass (필수 구현)
    # =========================================================================
    
    def forward(self, lr_image: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """
        LR 이미지 → SR → YOLO → Detection
        
        Args:
            lr_image: 저해상도 입력 [B, 3, H, W]
                     예: [4, 3, 192, 192]
        
        Returns:
            sr_image: SR 복원 이미지 [B, 3, H*4, W*4]
                     예: [4, 3, 768, 768]
            detections: YOLO detection 결과
        
        [데이터 흐름]
        
        lr_image [B, 3, 192, 192]
             │
             │ "6m GSD, 선박이 2~5픽셀"
             │
             ▼ self.sr_model(lr_image)
             │
             │ 내부: RFDN.forward()
             │   1. forward_features() → [B, 50, 192, 192]
             │   2. forward_reconstruct() → [B, 3, 768, 768]
             │
        sr_image [B, 3, 768, 768]
             │
             │ "1.5m GSD, 선박이 8~20픽셀"
             │
             ▼ self.detector(sr_image)
             │
             │ 내부: YOLOWrapper.forward()
             │   - YOLO 모델 실행
             │   - Detection 결과 반환
             │
        detections
             │
             ▼
        return (sr_image, detections)
        
        [주의]
        - sr_image는 SR Loss 계산과 시각화에 사용
        - detections는 Detection Loss 계산에 사용
        - 둘 다 반환해야 compute_loss에서 사용 가능
        """
        # Step 1: Super-Resolution
        # LR (192×192) → HR (768×768)
        sr_image = self.sr_model(lr_image)
        
        # Step 2: Object Detection
        # HR 이미지에서 선박 탐지
        detections = self.detector(sr_image)
        
        return sr_image, detections
    
    # =========================================================================
    # Loss Computation (필수 구현)
    # =========================================================================
    
    def compute_loss(
        self,
        outputs: Tuple[torch.Tensor, Any],
        targets: torch.Tensor,
        hr_gt: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        SR Loss + Detection Loss 계산
        
        Args:
            outputs: forward()의 출력 (sr_image, detections)
                    - sr_image: SR 복원 이미지 [B, 3, H*4, W*4]
                    - detections: YOLO 추론 결과 (Loss 계산에는 안 씀)
            targets: Detection GT (YOLO 형식)
                    [N, 6] = (batch_idx, class, x_center, y_center, w, h)
                    좌표는 0~1 정규화
            hr_gt: Ground Truth HR 이미지 [B, 3, H*4, W*4]
        
        Returns:
            loss_dict: {
                'total': α * L_SR + β * L_det,
                'sr_loss': L1 loss,
                'det_loss': YOLO loss
            }
        
        [Loss 구성]
        
        L_total = α × L_SR + β × L_detection
        
        where:
            α = self._sr_weight (기본 0.5)
            β = self._det_weight (기본 0.5)
        
        L_SR = L1(sr_image, hr_gt)
             = (1/N) × Σ|sr_image - hr_gt|
        
        L_detection = YOLO_Loss (Ultralytics v8DetectionLoss)
                    = λ_box × L_box + λ_cls × L_cls + λ_dfl × L_dfl
        
        [중요 변경점]
        YOLO Loss 계산 시:
        - 기존: forward() 결과(detections)로 loss 계산 (불가능)
        - 변경: sr_image를 다시 YOLO에 넣어서 loss 계산
        
        이유: Ultralytics Loss는 raw predictions이 필요하고,
              forward() 결과는 이미 후처리된 결과임
        
        [Arch 0 학습 전략]
        
        일반적으로 Arch 0에서는:
        1. YOLO는 Freeze (pretrained 사용)
        2. SR만 학습 (sr_weight만 영향)
        3. det_loss는 모니터링용으로만 사용
        
        따라서:
        - α = 1.0, β = 0.0 으로 설정하거나
        - YOLO Freeze 상태에서 β > 0 이어도 gradient 안 흐름
        """
        # outputs 분리
        sr_image, _ = outputs  # detections는 loss 계산에 안 씀
        
        # =====================================================================
        # SR Loss 계산
        # =====================================================================
        if hr_gt is not None:
            # L1 Loss: 픽셀별 절대 차이의 평균
            # 
            # [왜 L1인가?]
            # - L2보다 outlier에 덜 민감
            # - 선명한 이미지 생성 경향
            # - 계산이 빠름
            # 
            # [더 좋은 방법]
            # - Charbonnier Loss: L1의 smooth 버전
            # - Perceptual Loss: VGG feature 공간에서 비교
            # - 현재는 단순 L1, 나중에 확장 가능
            
            sr_loss = F.l1_loss(sr_image, hr_gt)
        else:
            # HR GT 없으면 SR Loss = 0
            sr_loss = torch.tensor(0.0, device=sr_image.device)
        
        # =====================================================================
        # Detection Loss 계산
        # =====================================================================
        # [중요] sr_image를 YOLO에 다시 넣어서 loss 계산
        # 
        # 왜? forward()의 detections는 후처리된 결과라서
        #     Ultralytics Loss 함수가 요구하는 raw predictions이 아님
        # 
        # YOLO가 Freeze 상태면 gradient 안 흐르지만
        # loss 값은 계산되어 모니터링 가능
        # =====================================================================
        
        det_loss_dict = self.detector.compute_loss(sr_image, targets)
        det_loss = det_loss_dict['total']
        
        # =====================================================================
        # Total Loss 계산
        # =====================================================================
        # L_total = α × L_SR + β × L_det
        
        total_loss = self._sr_weight * sr_loss + self._det_weight * det_loss
        
        return {
            'total': total_loss,
            'sr_loss': sr_loss,
            'det_loss': det_loss,
            # 세부 detection loss (모니터링용)
            'box_loss': det_loss_dict.get('box_loss', torch.tensor(0.0)),
            'cls_loss': det_loss_dict.get('cls_loss', torch.tensor(0.0)),
            'dfl_loss': det_loss_dict.get('dfl_loss', torch.tensor(0.0))
        }
    
    # =========================================================================
    # Pretrained Weights Loading
    # =========================================================================
    
    def load_pretrained_sr(self, checkpoint_path: str, strict: bool = True) -> None:
        """
        SR 모델 (RFDN)의 사전학습 가중치 로드
        
        Args:
            checkpoint_path: RFDN 체크포인트 경로
            strict: 엄격한 키 매칭
        
        [사용 예시]
        model = Arch0Sequential(config)
        model.load_pretrained_sr("pretrained/rfdn_div2k.pth")
        """
        self.sr_model.load_pretrained(checkpoint_path, strict=strict)
    
    def load_pretrained_detector(self, checkpoint_path: str) -> None:
        """
        Detector (YOLO)의 사전학습 가중치 로드
        
        Args:
            checkpoint_path: YOLO 체크포인트 경로
        
        [주의]
        - YOLO는 __init__에서 이미 로드됨
        - 추가로 fine-tuned 모델을 로드할 때 사용
        
        [사용 예시]
        model = Arch0Sequential(config)
        model.load_pretrained_detector("runs/train/airbus/weights/best.pt")
        """
        # 새로운 YOLO 생성하여 교체
        self.detector = YOLOWrapper(
            model_path=checkpoint_path,
            num_classes=self.num_classes,
            device=self.device,
            verbose=False
        )
        
        if self.freeze_detector_flag:
            self.detector.freeze()
    
    # =========================================================================
    # Architecture Info (Override)
    # =========================================================================
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """아키텍처 정보 반환 (Override)"""
        info = super().get_architecture_info()
        info.update({
            'architecture': 'Arch0_Sequential',
            'description': 'LR → SR (RFDN) → HR → YOLO → Detection',
            'rfdn_config': {
                'nf': self.nf,
                'num_modules': self.num_modules,
                'upscale': self.upscale_factor
            },
            'yolo_config': {
                'weights': self.yolo_weights,
                'num_classes': self.num_classes,
                'frozen': self.freeze_detector_flag
            }
        })
        return info


# =============================================================================
# Factory Function
# =============================================================================

def create_arch0_pipeline(config: Any) -> Arch0Sequential:
    """
    Arch0 파이프라인 생성 팩토리 함수
    
    [왜 Factory Function?]
    - 복잡한 초기화 로직을 숨김
    - 나중에 버전별 분기 가능
    - 테스트에서 Mock 객체 주입 용이
    
    Args:
        config: 설정 객체
    
    Returns:
        Arch0Sequential 인스턴스
    
    [사용 예시]
    from src.models.pipelines import create_arch0_pipeline
    model = create_arch0_pipeline(config)
    """
    return Arch0Sequential(config)


# =============================================================================
# 테스트 코드
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Arch0Sequential 테스트")
    print("=" * 70)
    
    # =========================================================================
    # 1. Config 생성 (테스트용)
    # =========================================================================
    from types import SimpleNamespace
    
    # SimpleNamespace로 config-like 객체 생성
    config = SimpleNamespace(
        model=SimpleNamespace(
            rfdn=SimpleNamespace(nf=50, num_modules=4),
            yolo=SimpleNamespace(weights_path="/home/changmin/test_for_darkvessel_ai_parameter_and_data/yolo11n.pt", num_classes=1)
        ),
        data=SimpleNamespace(upscale_factor=4, lr_size=192, hr_size=768),
        training=SimpleNamespace(
            sr_weight=0.5,
            det_weight=0.5,
            freeze_detector=True
        ),
        device='cpu'  # 테스트는 CPU에서
    )
    
    print("\n[Config]")
    print(f"  RFDN: nf={config.model.rfdn.nf}, modules={config.model.rfdn.num_modules}")
    print(f"  YOLO: {config.model.yolo.weights_path}")
    print(f"  Training: sr_weight={config.training.sr_weight}, det_weight={config.training.det_weight}")
    
    # =========================================================================
    # 2. 모델 생성
    # =========================================================================
    print("\n" + "=" * 70)
    print("모델 생성")
    print("=" * 70)
    
    model = Arch0Sequential(config)
    
    # =========================================================================
    # 3. Forward 테스트
    # =========================================================================
    print("\n" + "=" * 70)
    print("Forward 테스트")
    print("=" * 70)
    
    batch_size = 2
    lr_image = torch.randn(batch_size, 3, 192, 192)
    hr_gt = torch.randn(batch_size, 3, 768, 768)
    
    print(f"\n입력 LR shape: {lr_image.shape}")
    
    model.eval()
    with torch.no_grad():
        sr_image, detections = model(lr_image)
        print(f"출력 SR shape: {sr_image.shape}")
        print(f"Detection type: {type(detections)}")
    
    # =========================================================================
    # 4. Loss 테스트
    # =========================================================================
    print("\n" + "=" * 70)
    print("Loss 테스트")
    print("=" * 70)
    
    model.train()
    sr_image, detections = model(lr_image)
    
    # 더미 targets (실제로는 YOLO 형식 필요)
    targets = torch.zeros(1, 6)  # [batch_idx, class, x, y, w, h]
    
    loss_dict = model.compute_loss((sr_image, detections), targets, hr_gt)
    
    print(f"\nLoss 결과:")
    for name, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {name}: {value.item():.6f}")
    
    # =========================================================================
    # 5. 아키텍처 정보
    # =========================================================================
    print("\n" + "=" * 70)
    print("아키텍처 정보")
    print("=" * 70)
    
    info = model.get_architecture_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Arch0Sequential 테스트 완료!")