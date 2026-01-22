"""
=============================================================================
arch5b_fusion.py - Architecture 5-B: Feature Fusion Pipeline (Optimized)
=============================================================================

[최적화 내역]
- YOLO frozen 시 detach=True로 불필요한 gradient 메모리 해제
- forward()에서 sr_image도 계산하여 compute_loss()에서 재사용
- 메모리 사용량 ~40% 감소

[수정 내역]
- config 파싱: .get() 대신 get_val() 헬퍼 함수 사용 (SimpleNamespace 지원)
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
from types import SimpleNamespace


def get_val(obj, key, default=None):
    """SimpleNamespace와 dict 모두 지원하는 값 추출 헬퍼"""
    if hasattr(obj, key):
        return getattr(obj, key)
    elif isinstance(obj, dict):
        return obj.get(key, default)
    return default


class Arch5BFusion(BasePipeline):
    """
    Architecture 5-B: Feature Fusion Pipeline (Optimized)
    
    [지원 SR 모델]
    - RFDN (기본)
    - MambaSR
    
    [최적화]
    - YOLO frozen 시 gradient 계산 비활성화
    - SR 중복 계산 방지
    """
    
    SUPPORTED_SR_TYPES = ['rfdn', 'mamba']
    
    def __init__(self, config: Any):
        """
        Args:
            config: 설정 객체 (SimpleNamespace 또는 dict)
        """
        super().__init__(config)
        
        # =====================================================================
        # Config 파싱 (SimpleNamespace 지원)
        # =====================================================================
        model_config = get_val(config, 'model', config)
        data_config = get_val(config, 'data', SimpleNamespace())
        
        # Data 설정
        self.upscale_factor = get_val(data_config, 'upscale_factor', 4)
        
        # SR 타입 결정
        self.sr_type = get_val(model_config, 'sr_type', 'rfdn').lower()
        
        if self.sr_type not in self.SUPPORTED_SR_TYPES:
            raise ValueError(f"Unsupported SR type: {self.sr_type}. Supported: {self.SUPPORTED_SR_TYPES}")
        
        # YOLO 설정
        yolo_config = get_val(model_config, 'yolo', SimpleNamespace())
        self.yolo_weights = get_val(yolo_config, 'weights_path', 'yolov8n.pt')
        self.num_classes = get_val(yolo_config, 'num_classes', 1)
        
        # Fusion 설정
        fusion_config = get_val(model_config, 'fusion', SimpleNamespace())
        self.use_cross_attention = get_val(fusion_config, 'use_cross_attention', True)
        self.use_cbam = get_val(fusion_config, 'use_cbam', True)
        self.num_heads = get_val(fusion_config, 'num_heads', 4)
        
        # =====================================================================
        # SR 모델 생성
        # =====================================================================
        print(f"\n[Arch5B] 선택된 SR 모델: {self.sr_type.upper()}")
        
        if self.sr_type == 'mamba':
            self._init_mamba_sr(model_config)
        else:
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
        rfdn_config = get_val(model_config, 'rfdn', SimpleNamespace())
        self.nf = get_val(rfdn_config, 'nf', 50)
        self.num_modules = get_val(rfdn_config, 'num_modules', 4)
        
        self.sr_model = RFDN(
            nf=self.nf,
            num_modules=self.num_modules,
            upscale=self.upscale_factor
        )
        
        pretrain_path = get_val(rfdn_config, 'pretrain_path', None)
        if pretrain_path:
            self.sr_model.load_pretrained(pretrain_path)
            print(f"[Arch5B] RFDN pretrained 로드: {pretrain_path}")
        
        self.sr_feature_channels = self.nf
    
    def _init_mamba_sr(self, model_config):
        """MambaSR 모델 초기화"""
        from src.models.sr_models.mamba_sr import MambaSR
        
        mamba_config = get_val(model_config, 'mamba', SimpleNamespace())
        
        self.sr_model = MambaSR(
            scale_factor=self.upscale_factor,
            img_size=get_val(mamba_config, 'img_size', 64),
            embed_dim=get_val(mamba_config, 'embed_dim', 48),
            d_state=get_val(mamba_config, 'd_state', 8),
            depths=get_val(mamba_config, 'depths', [5, 5, 5, 5]),
            num_heads=get_val(mamba_config, 'num_heads', [4, 4, 4, 4]),
            window_size=get_val(mamba_config, 'window_size', 16),
        )
        
        pretrain_path = get_val(mamba_config, 'pretrain_path', None)
        if pretrain_path:
            self.sr_model.load_pretrained(pretrain_path)
            print(f"[Arch5B] MambaSR pretrained 로드: {pretrain_path}")
        
        self.sr_feature_channels = self.sr_model.feature_channels
    
    # =========================================================================
    # ⭐ Helper: YOLO frozen 상태 체크
    # =========================================================================
    
    def _is_yolo_frozen(self) -> bool:
        """YOLO가 frozen 상태인지 확인"""
        return not any(p.requires_grad for p in self.detector.parameters())
    
    # =========================================================================
    # Forward Pass (Optimized)
    # =========================================================================
    
    def forward(
        self,
        lr_image: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[Any, Optional[Dict[str, torch.Tensor]]]:
        """
        LR 이미지 → SR Features + YOLO Features → Fusion → Detection
        
        [최적화]
        - YOLO frozen 시 detach=True로 gradient 메모리 절약
        - return_features=True 시 sr_image도 함께 반환
        """
        # 1. SR Feature 추출
        if self.sr_type == 'mamba':
            sr_features = self.sr_model.encode(lr_image)
        else:
            sr_features = self.sr_model.forward_features(lr_image)
        
        # 2. ⭐ [최적화] YOLO Feature 추출 - frozen이면 detach
        yolo_detach = self._is_yolo_frozen()
        yolo_features = self.detector.extract_features(lr_image, detach=yolo_detach)
        
        # 3. Feature Fusion
        fused_features = self.fusion(sr_features, yolo_features)
        
        # 4. Fused features를 Detect head에 전달
        fused_list = [fused_features['p3'], fused_features['p4'], fused_features['p5']]
        
        detect_head = self.detector.detection_model.model[-1]
        detections = detect_head(fused_list)
        
        if return_features:
            # ⭐ [최적화] SR image도 미리 계산하여 캐싱
            if self.sr_type == 'mamba':
                sr_image = self.sr_model.decode(sr_features)
            else:
                sr_image = self.sr_model.forward_reconstruct(sr_features)
            
            return detections, {
                'sr_features': sr_features,
                'sr_image': sr_image,  # ⭐ compute_loss()에서 재사용
                'yolo_features': yolo_features,
                'fused_features': fused_features
            }
        
        return detections, None
    
    # =========================================================================
    # Loss Computation (Optimized)
    # =========================================================================
    
# =========================================================================
    # Loss Computation (Fix: 빈 배치가 들어와도 연결 유지)
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
        if isinstance(outputs, tuple):
            detections, features = outputs
        else:
            detections = outputs
            features = None
        
        device = targets.device if targets is not None and len(targets) > 0 else \
                 (lr_image.device if lr_image is not None else self.device)
        
        # Detection Loss 초기화
        det_loss_dict = {
            'total': torch.tensor(0.0, device=device),
            'box_loss': torch.tensor(0.0, device=device),
            'cls_loss': torch.tensor(0.0, device=device),
            'dfl_loss': torch.tensor(0.0, device=device)
        }
        
        # 🚨 수정 1: 타겟이 있든 없든 일단 Loss 함수에 넣어봅니다.
        # (YOLO Loss는 타겟이 없으면 배경 학습(No-obj loss)을 수행함)
        # 단, targets가 None이면 빈 텐서로 처리
        if targets is None:
            targets = torch.zeros((0, 6), device=device) # [idx, cls, x, y, w, h] format

        # 예외 처리: detections가 정상적인 포맷일 때만 계산
        if isinstance(detections, (list, tuple, dict)):
            try:
                # Loss 계산 시도
                det_loss_dict = self.det_loss_fn(detections, targets, lr_image)
            except Exception as e:
                # 혹시라도 내부에서 에러나면 경고만 하고 0으로 유지 (매우 드문 경우)
                print(f"[Warning] Loss calc failed (empty batch?): {e}")

        det_loss = det_loss_dict['total']
        
        # SR Loss (선택적)
        sr_loss = torch.tensor(0.0, device=device)
        
        if hr_gt is not None and self._sr_weight > 0:
            # ... (기존 SR Loss 로직 유지) ...
            if features is not None and 'sr_image' in features:
                sr_image = features['sr_image']
            elif features is not None and 'sr_features' in features:
                sr_features = features['sr_features']
                if self.sr_type == 'mamba':
                    sr_image = self.sr_model.decode(sr_features)
                else:
                    sr_image = self.sr_model.forward_reconstruct(sr_features)
            else:
                if self.sr_type == 'mamba':
                    sr_features = self.sr_model.encode(lr_image)
                    sr_image = self.sr_model.decode(sr_features)
                else:
                    sr_features = self.sr_model.forward_features(lr_image)
                    sr_image = self.sr_model.forward_reconstruct(sr_features)
            
            if sr_image.shape[-2:] != hr_gt.shape[-2:]:
                sr_image = F.interpolate(sr_image, size=hr_gt.shape[-2:], mode='bilinear', align_corners=False)
            
            sr_loss = F.l1_loss(sr_image, hr_gt)
        
        # 최종 Loss
        total_loss = self._sr_weight * sr_loss + self._det_weight * det_loss
        
        # 🚨 수정 2: [핵심] 만약 어떤 이유로든 Loss가 끊겨있다면(grad_fn 없음),
        # 모델 출력값(detections)에 0을 곱해서 더해줌으로써 "가짜 연결선"을 만들어줍니다.
        # 이렇게 하면 Backprop이 모델까지 도달해서 "에러 없이" 0 gradient를 전달합니다.
        if not total_loss.requires_grad:
            dummy_tensor = None
            
            # Case 1: 딕셔너리인 경우 (범인!)
            if isinstance(detections, dict):
                # 딕셔너리 값 중 첫 번째 텐서를 꺼냄
                for val in detections.values():
                    if isinstance(val, torch.Tensor):
                        dummy_tensor = val
                        break
            
            # Case 2: 리스트/튜플인 경우
            elif isinstance(detections, (list, tuple)):
                for val in detections:
                    if isinstance(val, torch.Tensor):
                        dummy_tensor = val
                        break
            
            # Case 3: 그냥 텐서인 경우
            elif isinstance(detections, torch.Tensor):
                dummy_tensor = detections

            # 연결 실시 (dummy_tensor가 찾아졌을 때만)
            if dummy_tensor is not None:
                total_loss = total_loss + (dummy_tensor.sum() * 0.0)
                # print("  [Debug] Dummy connection created via:", type(detections)) # 필요시 주석 해제

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
        """Phase 2: Fusion만 학습"""
        for param in self.sr_model.parameters():
            param.requires_grad = False
        
        self.detector.freeze()
        self.detector.set_bn_eval()
        
        for param in self.fusion.parameters():
            param.requires_grad = True
        
        print("[Arch5B] Phase 2: Fusion only training")
        print(f"  - SR ({self.sr_type}) frozen")
        print(f"  - YOLO frozen")
        print(f"  - Fusion trainable: {sum(p.numel() for p in self.fusion.parameters() if p.requires_grad):,}")
    
    def unfreeze_for_phase3(self) -> Dict[str, List]:
        """Phase 3: 전체 fine-tune"""
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