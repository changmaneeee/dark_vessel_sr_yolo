"""
=============================================================================
arch0_sequential.py - Architecture 0: Sequential Pipeline
=============================================================================

[지원 SR 모델]
- RFDN: 경량, 빠름 (기본)
- MambaSR: 고성능, Mamba 기반
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from src.models.pipelines.base_pipeline import BasePipeline
from src.models.sr_models.rfdn import RFDN
from src.models.detectors.yolo_wrapper import YOLOWrapper
from src.losses.detection_loss import DetectionLoss
from types import SimpleNamespace


class Arch0Sequential(BasePipeline):
    """
    Architecture 0: Sequential SR-Detection Pipeline
    
    [지원 SR 모델]
    - RFDN (기본)
    - MambaSR
    """
    
    SUPPORTED_SR_TYPES = ['rfdn', 'mamba']
    
    def __init__(self, config: Any):
        super().__init__(config)
        
        def get_val(obj, key, default=None):
            if hasattr(obj, key):
                return getattr(obj, key)
            elif isinstance(obj, dict):
                return obj.get(key, default)
            return default
        
        # Config 파싱
        model_config = get_val(config, 'model', config)
        data_config = get_val(config, 'data', SimpleNamespace())
        training_config = get_val(config, 'training', SimpleNamespace())
        
        # Data 설정
        self.upscale_factor = get_val(data_config, 'upscale_factor', 4)
        
        # SR 타입 결정
        self.sr_type = get_val(model_config, 'sr_type', 'rfdn').lower()
        
        if self.sr_type not in self.SUPPORTED_SR_TYPES:
            print(f"[Arch0] ⚠️ Unknown SR type '{self.sr_type}', falling back to RFDN")
            self.sr_type = 'rfdn'
        
        # YOLO 설정
        yolo_config = get_val(model_config, 'yolo', SimpleNamespace())
        self.yolo_weights = get_val(yolo_config, 'weights_path', 'yolov8n.pt')
        self.num_classes = get_val(yolo_config, 'num_classes', 1)
        
        # Training 설정
        self.freeze_detector_flag = get_val(training_config, 'freeze_detector', True)
        
        # =====================================================================
        # SR 모델 생성
        # =====================================================================
        print(f"\n[Arch0] 선택된 SR 모델: {self.sr_type.upper()}")
        
        if self.sr_type == 'mamba':
            self._init_mamba_sr(model_config)
        else:
            self._init_rfdn_sr(model_config)
        
        # =====================================================================
        # YOLO Detector 생성
        # =====================================================================
        print(f"\n[Arch0] Initializing YOLO...")
        
        self.detector = YOLOWrapper(
            model_path=self.yolo_weights,
            num_classes=self.num_classes,
            device=self.device,
            verbose=False
        )
        self.detection_loss_fn = DetectionLoss(self.detector.detection_model)

        if self.freeze_detector_flag:
            self.detector.freeze()
            print("✓ YOLO detector frozen")
        
        # 모델 정보
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[Arch0] Model Summary:")
        print(f"  - SR Model: {self.sr_type.upper()}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
    
    def _init_rfdn_sr(self, model_config):
        """RFDN 초기화"""
        rfdn_config = getattr(model_config, 'rfdn', {})
        self.nf = getattr(rfdn_config, 'nf', 50)
        self.num_modules = getattr(rfdn_config, 'num_modules', 4)
        
        self.sr_model = RFDN(
            in_channels=3,
            out_channels=3,
            nf=self.nf,
            num_modules=self.num_modules,
            upscale=self.upscale_factor
        )
    
    def _init_mamba_sr(self, model_config):
        """MambaSR 초기화"""
        from src.models.sr_models.mamba_sr import MambaSR
        
        mamba_config = getattr(model_config, 'mamba', {})
        
        self.sr_model = MambaSR(
            scale_factor=self.upscale_factor,
            img_size=getattr(mamba_config, 'img_size', 64),
            embed_dim=getattr(mamba_config, 'embed_dim', 48),
            d_state=getattr(mamba_config, 'd_state', 8),
            depths=getattr(mamba_config, 'depths', [5, 5, 5, 5]),
            num_heads=getattr(mamba_config, 'num_heads', [4, 4, 4, 4]),
            window_size=getattr(mamba_config, 'window_size', 16),
        )
        
        pretrain_path = getattr(mamba_config, 'pretrain_path', None)
        if pretrain_path:
            self.sr_model.load_pretrained(pretrain_path)
    
    def forward(self, lr_image: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """LR → SR → YOLO"""
        sr_image = self.sr_model(lr_image)
        detections = self.detector(sr_image)
        return sr_image, detections
    
    def compute_loss(
        self,
        outputs: Tuple[torch.Tensor, Any],
        targets: torch.Tensor,
        hr_gt: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Loss 계산"""
        sr_image, _ = outputs
        
        # SR Loss
        if hr_gt is not None:
            sr_loss = F.l1_loss(sr_image, hr_gt)
        else:
            sr_loss = torch.tensor(0.0, device=sr_image.device)

        # Detection Loss
        self.detector.detection_model.model.train()
        preds = self.detector.detection_model.model(sr_image)
        det_loss_dict = self.detection_loss_fn(preds, targets, sr_image)
        det_loss = det_loss_dict['total']
        
        total_loss = self._sr_weight * sr_loss + self._det_weight * det_loss
        
        return {
            'total': total_loss,
            'sr_loss': sr_loss,
            'det_loss': det_loss,
            'box_loss': det_loss_dict.get('box_loss', torch.tensor(0.0)),
            'cls_loss': det_loss_dict.get('cls_loss', torch.tensor(0.0)),
            'dfl_loss': det_loss_dict.get('dfl_loss', torch.tensor(0.0))
        }
    
    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update({
            'architecture': 'Arch0_Sequential',
            'sr_type': self.sr_type,
        })
        return info


def create_arch0_pipeline(config: Any) -> Arch0Sequential:
    return Arch0Sequential(config)