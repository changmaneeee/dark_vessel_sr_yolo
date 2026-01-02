"""
=============================================================================
arch5b_fusion.py - Architecture 5-B: Feature Fusion Pipeline
=============================================================================

[역할]
SR Feature와 YOLO Feature를 Feature 수준에서 융합하여 Detection 성능 향상
- HR 이미지 생성 없이 Feature만 융합 (연산량 절약)
- Multi-scale Attention Fusion으로 SR 정보 활용
- End-to-end 학습 가능

[지원 SR 모델]
- RFDN: 경량, 빠름 (기본)
- MambaSR: 고성능, Mamba 기반

[Arch 5-B 구조]

LR Image [B, 3, 192, 192]
    │
    ├───────────────────────────────────────┐
    │                                        │
    ▼                                        ▼
┌─────────────┐                       ┌───────────┐
│ SR Model    │ encode()              │   YOLO    │ extract_features()
│ (RFDN/Mamba)│                       └─────┬─────┘
└──────┬──────┘                             │
       │                                    │
       │ SR Features                        │ YOLO Features
       │ [B, C_sr, H, W]                   │ P3, P4, P5
       │                                    │
       └──────────────┬─────────────────────┘
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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List

from src.models.pipelines.base_pipeline import BasePipeline
from src.models.sr_models.rfdn import RFDN
from src.models.detectors.yolo_wrapper import YOLOWrapper
from src.models.fusion.attention_fusion import MultiScaleAttentionFusion
from src.losses.combined_loss import CombinedLoss
from src.losses.detection_loss import DetectionLoss


class Arch5BFusion(BasePipeline):
    """
    Architecture 5-B: Feature Fusion Pipeline
    
    [상속]
    BasePipeline을 상속하여 표준 인터페이스 준수
    
    [구성 요소]
    - sr_model (RFDN or MambaSR): SR Feature Encoder
    - detector (YOLOWrapper): YOLO Backbone + Neck + Detect
    - fusion (MultiScaleAttentionFusion): SR-YOLO Feature Fusion
    """
    
    # 지원하는 SR 모델 타입
    SUPPORTED_SR_TYPES = ['rfdn', 'mamba']
    
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
        
        # Data 설정
        self.upscale_factor = getattr(data_config, 'upscale_factor', data_config.get('upscale_factor', 4))
        
        # SR 타입 결정
        self.sr_type = getattr(model_config, 'sr_type', model_config.get('sr_type', 'rfdn')).lower()
        
        if self.sr_type not in self.SUPPORTED_SR_TYPES:
            raise ValueError(f"Unsupported SR type: {self.sr_type}. Supported: {self.SUPPORTED_SR_TYPES}")
        
        # YOLO 설정
        yolo_config = getattr(model_config, 'yolo', model_config.get('yolo', {}))
        self.yolo_weights = getattr(yolo_config, 'weights_path', yolo_config.get('weights_path', 'yolo11s.pt'))
        self.num_classes = getattr(yolo_config, 'num_classes', yolo_config.get('num_classes', 1))
        
        # Fusion 설정
        fusion_config = getattr(model_config, 'fusion', model_config.get('fusion', {}))
        self.use_cross_attention = getattr(fusion_config, 'use_cross_attention', fusion_config.get('use_cross_attention', True))
        self.use_cbam = getattr(fusion_config, 'use_cbam', fusion_config.get('use_cbam', True))
        self.num_heads = getattr(fusion_config, 'num_heads', fusion_config.get('num_heads', 4))
        
        # =====================================================================
        # SR 모델 생성 (RFDN or MambaSR)
        # =====================================================================
        print(f"\n[Arch5B] 선택된 SR 모델: {self.sr_type.upper()}")
        
        if self.sr_type == 'mamba':
            self._init_mamba_sr(model_config)
        else:  # rfdn (기본)
            self._init_rfdn_sr(model_config)
        
        print(f"[Arch5B] SR Feature 채널: {self.sr_feature_channels}")
        
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
        # Loss 함수
        # =====================================================================
        self.loss_fn = CombinedLoss(
            yolo_model=self.detector.detection_model,
            sr_weight=self._sr_weight,
            det_weight=self._det_weight,
            phase_schedule=True
        )

        self.det_loss_fn = DetectionLoss(self.detector.detection_model)
        
        # =====================================================================
        # 모델 정보 출력
        # =====================================================================
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n[Arch5B] Model Summary:")
        print(f"  - SR Model: {self.sr_type.upper()}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - SR weight (α): {self._sr_weight}")
        print(f"  - Det weight (β): {self._det_weight}")
    
    # =========================================================================
    # SR 모델 초기화 헬퍼
    # =========================================================================
    
    def _init_rfdn_sr(self, model_config):
        """RFDN SR 모델 초기화"""
        rfdn_config = getattr(model_config, 'rfdn', model_config.get('rfdn', {}))
        self.nf = getattr(rfdn_config, 'nf', rfdn_config.get('nf', 50))
        self.num_modules = getattr(rfdn_config, 'num_modules', rfdn_config.get('num_modules', 4))
        
        self.sr_model = RFDN(
            nf=self.nf,
            num_modules=self.num_modules,
            upscale=self.upscale_factor
        )
        
        # RFDN pretrained 로드
        pretrain_path = getattr(rfdn_config, 'pretrain_path', rfdn_config.get('pretrain_path', None))
        if pretrain_path:
            self.sr_model.load_pretrained(pretrain_path)
            print(f"[Arch5B] RFDN pretrained 로드: {pretrain_path}")
        
        self.sr_feature_channels = self.nf
    
    def _init_mamba_sr(self, model_config):
        """MambaSR 모델 초기화"""
        # MambaSR import (여기서 해야 mamba_ssm 없는 환경에서도 RFDN 사용 가능)
        from src.models.sr_models.mamba_sr import MambaSR
        
        mamba_config = getattr(model_config, 'mamba', model_config.get('mamba', {}))
        
        # MambaSR 생성
        self.sr_model = MambaSR(
            scale_factor=self.upscale_factor,
            img_size=getattr(mamba_config, 'img_size', mamba_config.get('img_size', 64)),
            embed_dim=getattr(mamba_config, 'embed_dim', mamba_config.get('embed_dim', 48)),
            d_state=getattr(mamba_config, 'd_state', mamba_config.get('d_state', 8)),
            depths=getattr(mamba_config, 'depths', mamba_config.get('depths', [5, 5, 5, 5])),
            num_heads=getattr(mamba_config, 'num_heads', mamba_config.get('num_heads', [4, 4, 4, 4])),
            window_size=getattr(mamba_config, 'window_size', mamba_config.get('window_size', 16)),
        )
        
        # Pretrained 로드
        pretrain_path = getattr(mamba_config, 'pretrain_path', mamba_config.get('pretrain_path', None))
        if pretrain_path:
            self.sr_model.load_pretrained(pretrain_path)
            print(f"[Arch5B] MambaSR pretrained 로드: {pretrain_path}")
        
        self.sr_feature_channels = self.sr_model.feature_channels
    
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
        """
        # 1. SR Feature 추출 (HR 복원 없이 feature만)
        # RFDN: forward_features() / MambaSR: encode()
        if self.sr_type == 'mamba':
            sr_features = self.sr_model.encode(lr_image)
        else:
            sr_features = self.sr_model.forward_features(lr_image)
        
        # 2. YOLO Feature 추출 (gradient 유지)
        yolo_features = self.detector.extract_features(lr_image, detach=False)
        
        # 3. Feature Fusion
        fused_features = self.fusion(sr_features, yolo_features)
        
        # 4. Fused features를 Detect head에 전달
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
        
        return detections, None
    
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
        """
        # outputs 처리
        if isinstance(outputs, tuple):
            detections, features = outputs
        else:
            detections = outputs
            features = None
        
        device = targets.device if targets is not None and len(targets) > 0 else lr_image.device
               
        # =====================================================================
        # Detection Loss
        # =====================================================================
        det_loss_dict = {
            'total': torch.tensor(0.0, device=device),
            'box_loss': torch.tensor(0.0, device=device),
            'cls_loss': torch.tensor(0.0, device=device),
            'dfl_loss': torch.tensor(0.0, device=device)
        }
        
        if targets is not None and len(targets) > 0 and lr_image is not None:
            self.detector.train()
            preds = self.detector(lr_image)
            det_loss_dict = self.det_loss_fn(preds, targets, lr_image)

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
                if self.sr_type == 'mamba':
                    sr_features = self.sr_model.encode(lr_image)
                else:
                    sr_features = self.sr_model.forward_features(lr_image)
            
            # HR 복원
            if self.sr_type == 'mamba':
                sr_image = self.sr_model.decode(sr_features)
            else:
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
        """추론 모드"""
        self.eval()
        
        detections, features = self.forward(lr_image, return_features=True)

        return {
            'detections': detections,
            'features': features
        }
    
    # =========================================================================
    # Phase별 Freeze/Unfreeze
    # =========================================================================
    
    def freeze_for_phase2(self) -> None:
        """Phase 2: Fusion만 학습 (SR/YOLO freeze)"""
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
        print(f"  - SR ({self.sr_type}) frozen")
        print(f"  - YOLO frozen")
        print(f"  - Fusion trainable: {sum(p.numel() for p in self.fusion.parameters() if p.requires_grad):,}")
    
    def unfreeze_for_phase3(self) -> Dict[str, List]:
        """Phase 3: 전체 fine-tune"""
        # 전체 unfreeze
        for param in self.sr_model.parameters():
            param.requires_grad = True
        
        self.detector.unfreeze()
        
        for param in self.fusion.parameters():
            param.requires_grad = True
        
        print("[Arch5B] Phase 3: Full fine-tuning")
        
        return {
            'sr': list(self.sr_model.parameters()),
            'detector': list(self.detector.detection_model.parameters()),
            'fusion': list(self.fusion.parameters())
        }
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Architecture information"""
        info = super().get_architecture_info()
        info.update({
            'architecture': 'Arch5B_FeatureFusion',
            'sr_type': self.sr_type,
            'components': {
                'sr_model': self.sr_type.upper(),
                'detector': 'YOLO',
                'fusion': 'MultiScaleAttentionFusion',
                'loss': 'CombinedLoss'
            }
        })
        return info


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
    
    # Config - RFDN 테스트
    config_rfdn = SimpleNamespace(
        model=SimpleNamespace(
            sr_type='rfdn',  # RFDN 사용
            rfdn=SimpleNamespace(nf=50, num_modules=4),
            yolo=SimpleNamespace(weights_path="yolov8n.pt", num_classes=80),
            fusion=SimpleNamespace(use_cross_attention=True, use_cbam=True, num_heads=4)
        ),
        data=SimpleNamespace(upscale_factor=4),
        training=SimpleNamespace(sr_weight=0.3, det_weight=0.7),
        device=device
    )
    
    try:
        print("\n[1. RFDN 테스트]")
        model = Arch5BFusion(config_rfdn)
        print("✓ Arch5BFusion (RFDN) 생성 성공")
        
        # Mamba 테스트 (mamba_ssm 있을 때만)
        try:
            print("\n[2. MambaSR 테스트]")
            config_mamba = SimpleNamespace(
                model=SimpleNamespace(
                    sr_type='mamba',  # MambaSR 사용
                    mamba=SimpleNamespace(
                        embed_dim=48,
                        depths=[5, 5, 5, 5],
                        d_state=8,
                        window_size=16
                    ),
                    yolo=SimpleNamespace(weights_path="yolov8n.pt", num_classes=80),
                    fusion=SimpleNamespace(use_cross_attention=True, use_cbam=True, num_heads=4)
                ),
                data=SimpleNamespace(upscale_factor=4),
                training=SimpleNamespace(sr_weight=0.3, det_weight=0.7),
                device=device
            )
            model_mamba = Arch5BFusion(config_mamba)
            print("✓ Arch5BFusion (MambaSR) 생성 성공")
        except ImportError as e:
            print(f"⚠️ MambaSR 테스트 스킵 (mamba_ssm 미설치): {e}")
        
        print("\n" + "=" * 70)
        print("✓ Arch5BFusion 테스트 완료!")
        print("=" * 70)
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()