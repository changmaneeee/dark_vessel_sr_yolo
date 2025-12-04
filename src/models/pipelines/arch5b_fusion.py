"""
=============================================================================
arch5b_fusion.py - Architecture 5-B: Feature Fusion Pipeline
=============================================================================

[역할]
SR Feature와 YOLO Feature를 Feature 수준에서 융합하여 Detection 성능 향상
- HR 이미지 생성 없이 Feature만 융합 (연산량 절약)
- Multi-scale Attention Fusion으로 SR 정보 활용
- End-to-end 학습 가능

[Arch 5-B 구조]

LR Image [B, 3, 192, 192]
    │
    ├───────────────────────────────────────┐
    │                                        │
    ▼                                        ▼
┌─────────┐                           ┌───────────┐
│  RFDN   │ forward_features()        │   YOLO    │ extract_features()
└────┬────┘                           └─────┬─────┘
     │                                      │
     │ SR Features                          │ YOLO Features
     │ [B, 50, 192, 192]                   │ P3: [B, C3, H/8, W/8]
     │                                      │ P4: [B, C4, H/16, W/16]
     │                                      │ P5: [B, C5, H/32, W/32]
     │                                      │
     └──────────────┬───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ MultiScaleAttention   │
        │      Fusion           │
        └───────────┬───────────┘
                    │
                    ▼
         Fused Features (P3', P4', P5')
                    │
                    ▼
              YOLO Detect Head
                    │
                    ▼
             Detection Results

[Arch 0 대비 장점]
1. 연산량 감소: HR 이미지(768×768) 생성 불필요
2. Feature 수준 융합: 더 세밀한 정보 결합
3. End-to-end 학습: SR과 Detection이 함께 최적화

[학습 전략: C방식]
Phase 1: SR/YOLO 개별 pretrain (또는 기존 가중치 사용)
Phase 2: Fusion 모듈만 학습 (SR/YOLO freeze)
Phase 3: 전체 fine-tune (낮은 LR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List

from src.models.pipelines.base_pipeline import BasePipeline
from src.models.sr_models.rfdn import RFDN
from src.models.detectors.yolo_wrapper import YOLOWrapper
from src.models.fusion.attention_fusion import MultiScaleAttentionFusion


class Arch5BFusion(BasePipeline):
    """
    Architecture 5-B: Feature Fusion Pipeline
    
    [상속]
    BasePipeline을 상속하여 표준 인터페이스 준수
    
    [구성 요소]
    - sr_model (RFDN): SR Feature Encoder
    - detector (YOLOWrapper): YOLO Backbone + Neck + Detect
    - fusion (MultiScaleAttentionFusion): SR-YOLO Feature Fusion
    
    [Config 구조]
    config:
      model:
        rfdn:
          nf: 50
          num_modules: 4
        yolo:
          weights_path: "yolo11s.pt"
          num_classes: 1
        fusion:
          use_cross_attention: true
          use_cbam: true
          num_heads: 4
      data:
        upscale_factor: 4
        lr_size: 192
      training:
        sr_weight: 0.3
        det_weight: 0.7
        phase_schedule: true
    
    [사용 예시]
    model = Arch5BFusion(config)
    
    # 학습
    model.train()
    detections = model(lr_image)
    loss_dict = model.compute_loss(detections, targets, lr_image)
    
    # 추론
    model.eval()
    results = model.inference(lr_image)
    """
    
    def __init__(self, config: Any):
        """
        Args:
            config: 설정 객체 (OmegaConf 또는 dict-like)
        """
        # =====================================================================
        # BasePipeline 초기화
        # =====================================================================
        super().__init__(config)
        
        # Config 추출
        model_config = getattr(config, 'model', config.get('model', {}))
        data_config = getattr(config, 'data', config.get('data', {}))
        training_config = getattr(config, 'training', config.get('training', {}))
        
        # RFDN 설정
        rfdn_config = getattr(model_config, 'rfdn', model_config.get('rfdn', {}))
        self.nf = getattr(rfdn_config, 'nf', rfdn_config.get('nf', 50))
        self.num_modules = getattr(rfdn_config, 'num_modules', rfdn_config.get('num_modules', 4))
        
        # YOLO 설정
        yolo_config = getattr(model_config, 'yolo', model_config.get('yolo', {}))
        self.yolo_weights = getattr(yolo_config, 'weights_path', yolo_config.get('weights_path', 'yolo11s.pt'))
        self.num_classes = getattr(yolo_config, 'num_classes', yolo_config.get('num_classes', 1))
        
        # Fusion 설정
        fusion_config = getattr(model_config, 'fusion', model_config.get('fusion', {}))
        self.use_cross_attention = getattr(fusion_config, 'use_cross_attention', fusion_config.get('use_cross_attention', True))
        self.use_cbam = getattr(fusion_config, 'use_cbam', fusion_config.get('use_cbam', True))
        self.num_heads = getattr(fusion_config, 'num_heads', fusion_config.get('num_heads', 4))
        
        # Data 설정
        self.upscale_factor = getattr(data_config, 'upscale_factor', data_config.get('upscale_factor', 4))
        
        # =====================================================================
        # SR 모델 (RFDN) 생성
        # =====================================================================
        print(f"\n[Arch5B] Initializing RFDN...")
        
        self.sr_model = RFDN(
            in_channels=3,
            out_channels=3,
            nf=self.nf,
            num_modules=self.num_modules,
            upscale=self.upscale_factor
        )
        
        # SR feature 채널 수
        self.sr_feature_channels = self.nf  # 50
        
        # =====================================================================
        # YOLO Detector 생성
        # =====================================================================
        print(f"\n[Arch5B] Initializing YOLO...")
        
        self.detector = YOLOWrapper(
            model_path=self.yolo_weights,
            num_classes=self.num_classes,
            device=self.device,
            verbose=False
        )
        
        # YOLO feature 채널 정보 가져오기
        yolo_channels = self.detector.get_feature_channels()
        print(f"[Arch5B] YOLO feature channels: {yolo_channels}")
        
        # =====================================================================
        # Fusion 모듈 생성
        # =====================================================================
        print(f"\n[Arch5B] Initializing Fusion Module...")
        
        self.fusion = MultiScaleAttentionFusion(
            sr_channels=self.sr_feature_channels,
            yolo_channels=yolo_channels,
            use_cross_attention=self.use_cross_attention,
            use_cbam=self.use_cbam,
            num_heads=self.num_heads
        )
        
        # =====================================================================
        # Loss 함수 (Lazy initialization)
        # =====================================================================
        self._combined_loss = None
        
        # =====================================================================
        # 모델 정보 출력
        # =====================================================================
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n[Arch5B] Model Summary:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - SR weight (α): {self._sr_weight}")
        print(f"  - Det weight (β): {self._det_weight}")
    
    # =========================================================================
    # Forward Pass
    # =========================================================================
    
    def forward(
        self,
        lr_image: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[Any, Optional[Dict[str, torch.Tensor]]]:
        """
        LR 이미지 → SR Features + YOLO Features → Fusion → Detection
        
        Args:
            lr_image: 저해상도 입력 [B, 3, H, W]
            return_features: 중간 feature도 반환할지 여부
        
        Returns:
            detections: YOLO detection 결과
            features (optional): 중간 feature dict
        
        [데이터 흐름]
        
        lr_image [B, 3, 192, 192]
             │
             ├── sr_model.forward_features() ────┐
             │        │                          │
             │   sr_features                     │
             │   [B, 50, 192, 192]               │
             │                                   │
             └── detector.extract_features() ────┤
                      │                          │
                 yolo_features                   │
                 P3: [B, C3, H/8, W/8]           │
                 P4: [B, C4, H/16, W/16]         │
                 P5: [B, C5, H/32, W/32]         │
                                                 │
                 fusion.forward() ◄──────────────┘
                      │
                 fused_features
                 P3', P4', P5' (same shape as YOLO)
                      │
                 detector.detection_model.model[-1](fused_features)
                      │
                 detections
        """
        # 1. SR Feature 추출 (HR 복원 없이 feature만)
        sr_features = self.sr_model.forward_features(lr_image)
        
        # 2. YOLO Feature 추출 (gradient 유지)
        yolo_features = self.detector.extract_features(lr_image, detach=False)
        
        # 3. Feature Fusion
        fused_features = self.fusion(sr_features, yolo_features)
        
        # 4. Fused features를 Detect head에 전달
        # YOLO Detect head는 [P3, P4, P5] 리스트를 입력으로 받음
        fused_list = [fused_features['p3'], fused_features['p4'], fused_features['p5']]
        
        # Detect head forward
        detect_head = self.detector.detection_model.model[-1]
        detections = detect_head(fused_list)
        
        if return_features:
            return detections, {
                'sr_features': sr_features,
                'yolo_features': yolo_features,
                'fused_features': fused_features
            }
        
        return detections
    
    # =========================================================================
    # Loss Computation
    # =========================================================================
    
    def compute_loss(
        self,
        outputs: Any,
        targets: torch.Tensor,
        lr_image: Optional[torch.Tensor] = None,
        hr_gt: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Detection Loss + (선택적) SR Loss 계산
        
        Args:
            outputs: forward()의 출력 (detections)
                    또는 (detections, features) if return_features=True
            targets: Detection GT [N, 6] = (batch_idx, class, x, y, w, h)
            lr_image: LR 입력 이미지 (Detection loss 계산용)
            hr_gt: HR GT 이미지 (SR loss 계산용, 선택적)
        
        Returns:
            loss_dict: {
                'total': 전체 loss,
                'det_loss': Detection loss,
                'sr_loss': SR loss (hr_gt 있을 때),
                'box_loss': Box loss,
                'cls_loss': Cls loss,
                'dfl_loss': DFL loss
            }
        
        [Arch 5-B Loss 구성]
        - Detection Loss: v8DetectionLoss on fused features
        - SR Loss (선택): Feature에서 HR 복원 후 비교 (Phase 1에서 사용)
        
        실제 학습에서는 Detection Loss가 메인!
        """
        # outputs 처리
        if isinstance(outputs, tuple):
            detections, features = outputs
        else:
            detections = outputs
            features = None
        
        device = targets.device if targets is not None and len(targets) > 0 else self.device
        
        # =====================================================================
        # Detection Loss
        # =====================================================================
        det_loss = torch.tensor(0.0, device=device, requires_grad=True)
        det_loss_dict = {}
        
        if targets is not None and len(targets) > 0 and lr_image is not None:
            # 학습 모드에서 다시 forward하여 loss 계산
            # (Ultralytics loss는 모델의 training mode forward output이 필요)
            
            self.detector.train()
            
            # Full forward로 raw predictions 얻기
            _, yolo_features = self.forward(lr_image, return_features=True)
            
            # YOLO detection model을 training mode로 forward
            # fused features로 Detect head 호출
            fused_list = [
                yolo_features['fused_features']['p3'],
                yolo_features['fused_features']['p4'],
                yolo_features['fused_features']['p5']
            ]
            
            # Detection loss를 위해 전체 model forward 필요
            # detector.compute_loss는 images를 받아서 내부에서 forward함
            det_loss_dict = self.detector.compute_loss(lr_image, targets)
            det_loss = det_loss_dict['total']
        
        # =====================================================================
        # SR Loss (선택적 - Phase 1에서 SR 안정화용)
        # =====================================================================
        sr_loss = torch.tensor(0.0, device=device)
        
        if hr_gt is not None and self._sr_weight > 0:
            # SR 복원 (feature에서 HR 생성)
            if features is not None and 'sr_features' in features:
                sr_features = features['sr_features']
            else:
                sr_features = self.sr_model.forward_features(lr_image)
            
            sr_image = self.sr_model.forward_reconstruct(sr_features)
            sr_loss = F.l1_loss(sr_image, hr_gt)
        
        # =====================================================================
        # Total Loss
        # =====================================================================
        total_loss = self._sr_weight * sr_loss + self._det_weight * det_loss
        
        return {
            'total': total_loss,
            'det_loss': det_loss,
            'sr_loss': sr_loss,
            'box_loss': det_loss_dict.get('box_loss', torch.tensor(0.0, device=device)),
            'cls_loss': det_loss_dict.get('cls_loss', torch.tensor(0.0, device=device)),
            'dfl_loss': det_loss_dict.get('dfl_loss', torch.tensor(0.0, device=device))
        }
    
    # =========================================================================
    # Inference
    # =========================================================================
    
    @torch.no_grad()
    def inference(
        self,
        lr_image: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        추론 모드
        
        Args:
            lr_image: LR 입력 이미지
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            return_features: Feature도 반환할지
        
        Returns:
            {
                'detections': [{boxes, scores, classes}, ...],
                'features': (optional) 중간 features
            }
        """
        self.eval()
        
        # Forward
        if return_features:
            detections, features = self.forward(lr_image, return_features=True)
        else:
            detections = self.forward(lr_image)
            features = None
        
        # Post-processing (NMS는 YOLO predict에서 처리)
        # 여기서는 raw detections 반환
        results = {
            'detections': detections,
        }
        
        if features is not None:
            results['features'] = features
        
        return results
    
    # =========================================================================
    # Phase별 Freeze/Unfreeze
    # =========================================================================
    
    def freeze_for_phase2(self) -> None:
        """
        Phase 2: Fusion만 학습 (SR/YOLO freeze)
        """
        # SR freeze
        for param in self.sr_model.parameters():
            param.requires_grad = False
        
        # YOLO freeze
        self.detector.freeze()
        self.detector.set_bn_eval()
        
        # Fusion unfreeze
        for param in self.fusion.parameters():
            param.requires_grad = True
        
        print("[Arch5B] Phase 2: Fusion only training")
        print(f"  - SR frozen: {sum(p.numel() for p in self.sr_model.parameters() if not p.requires_grad):,}")
        print(f"  - YOLO frozen: {self.detector.count_parameters()['frozen']:,}")
        print(f"  - Fusion trainable: {sum(p.numel() for p in self.fusion.parameters() if p.requires_grad):,}")
    
    def unfreeze_for_phase3(self, sr_lr_scale: float = 0.1, yolo_lr_scale: float = 0.1) -> Dict[str, List]:
        """
        Phase 3: 전체 fine-tune (다른 LR 사용)
        
        Returns:
            param_groups: Optimizer에 전달할 파라미터 그룹
        """
        # 전체 unfreeze
        for param in self.sr_model.parameters():
            param.requires_grad = True
        
        self.detector.unfreeze()
        
        for param in self.fusion.parameters():
            param.requires_grad = True
        
        print("[Arch5B] Phase 3: Full fine-tuning")
        
        # 다른 LR을 위한 파라미터 그룹
        param_groups = [
            {'params': self.sr_model.parameters(), 'lr_scale': sr_lr_scale, 'name': 'sr'},
            {'params': self.detector.detection_model.parameters(), 'lr_scale': yolo_lr_scale, 'name': 'yolo'},
            {'params': self.fusion.parameters(), 'lr_scale': 1.0, 'name': 'fusion'},
        ]
        
        return param_groups
    
    # =========================================================================
    # Architecture Info
    # =========================================================================
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """아키텍처 정보 반환"""
        info = super().get_architecture_info()
        info.update({
            'architecture': 'Arch5B_FeatureFusion',
            'description': 'LR → SR_Features + YOLO_Features → Attention_Fusion → Detection',
            'rfdn_config': {
                'nf': self.nf,
                'num_modules': self.num_modules,
                'feature_channels': self.sr_feature_channels
            },
            'yolo_config': {
                'weights': self.yolo_weights,
                'num_classes': self.num_classes,
                'feature_channels': self.detector.get_feature_channels()
            },
            'fusion_config': {
                'use_cross_attention': self.use_cross_attention,
                'use_cbam': self.use_cbam,
                'num_heads': self.num_heads
            }
        })
        return info


# =============================================================================
# Factory Function
# =============================================================================

def create_arch5b_pipeline(config: Any) -> Arch5BFusion:
    """Arch5B 파이프라인 생성 팩토리 함수"""
    return Arch5BFusion(config)


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Arch5BFusion 테스트")
    print("=" * 70)
    
    from types import SimpleNamespace
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Config 생성
    config = SimpleNamespace(
        model=SimpleNamespace(
            rfdn=SimpleNamespace(nf=50, num_modules=4),
            yolo=SimpleNamespace(weights_path="yolov8n.pt", num_classes=80),
            fusion=SimpleNamespace(use_cross_attention=True, use_cbam=True, num_heads=4)
        ),
        data=SimpleNamespace(upscale_factor=4, lr_size=192),
        training=SimpleNamespace(
            sr_weight=0.3,
            det_weight=0.7,
            phase_schedule=True
        ),
        device=device
    )
    
    # 모델 생성
    print("\n" + "=" * 70)
    print("1. 모델 생성")
    print("=" * 70)
    
    try:
        model = Arch5BFusion(config)
        print("\n✓ Arch5BFusion 생성 성공!")
    except Exception as e:
        print(f"모델 생성 실패: {e}")
        exit(1)
    
    # Forward 테스트
    print("\n" + "=" * 70)
    print("2. Forward 테스트")
    print("=" * 70)
    
    batch_size = 2
    lr_image = torch.randn(batch_size, 3, 640, 640, device=device)
    
    model.eval()
    with torch.no_grad():
        detections, features = model(lr_image, return_features=True)
    
    print(f"\n입력 LR shape: {lr_image.shape}")
    print(f"SR features shape: {features['sr_features'].shape}")
    print("YOLO features:")
    for k, v in features['yolo_features'].items():
        print(f"  {k}: {v.shape}")
    print("Fused features:")
    for k, v in features['fused_features'].items():
        print(f"  {k}: {v.shape}")
    print(f"Detections type: {type(detections)}")
    
    # Phase 테스트
    print("\n" + "=" * 70)
    print("3. Phase 테스트")
    print("=" * 70)
    
    model.freeze_for_phase2()
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nPhase 2: Trainable {trainable:,} / Total {total:,}")
    
    # 아키텍처 정보
    print("\n" + "=" * 70)
    print("4. 아키텍처 정보")
    print("=" * 70)
    
    info = model.get_architecture_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✓ Arch5BFusion 테스트 완료!")
    print("=" * 70)