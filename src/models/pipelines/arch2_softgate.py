"""
=============================================================================
arch2_softgate.py - Architecture 2: SoftGate Pipeline
=============================================================================

[수정 내역]
- compute_loss에서 gradient 연결 개선
- Loss 계산 시 requires_grad 유지
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List

from src.models.pipelines.base_pipeline import BasePipeline
from src.models.sr_models.rfdn import RFDN
from src.models.detectors.yolo_wrapper import YOLOWrapper
from src.models.gates.soft_gate import LightweightGate
from src.losses.detection_loss import DetectionLoss
from src.losses.sr_loss import SRLoss
from types import SimpleNamespace


def get_val(obj, key, default=None):
    """SimpleNamespace와 dict 모두 지원하는 값 추출 헬퍼"""
    if hasattr(obj, key):
        return getattr(obj, key)
    elif isinstance(obj, dict):
        return obj.get(key, default)
    return default


class Arch2SoftGate(BasePipeline):
    """
    Architecture 2: SoftGate Pipeline
    
    [지원 SR 모델]
    - RFDN (기본)
    - MambaSR
    """

    SUPPORTED_SR_TYPES = ['rfdn', 'mamba']

    def __init__(self, config: Any):
        super().__init__(config)

        # Config 파싱
        model_config = get_val(config, 'model', config)
        data_config = get_val(config, 'data', SimpleNamespace())
        
        # Data 설정
        self.upscale_factor = get_val(data_config, 'upscale_factor', 4)
        
        # SR 타입 결정
        self.sr_type = get_val(model_config, 'sr_type', 'rfdn').lower()
        
        if self.sr_type not in self.SUPPORTED_SR_TYPES:
            print(f"[Arch2] ⚠️ Unknown SR type '{self.sr_type}', falling back to RFDN")
            self.sr_type = 'rfdn'
        
        # YOLO 설정
        yolo_config = get_val(model_config, 'yolo', SimpleNamespace())
        self.yolo_weights = get_val(yolo_config, 'weights_path', 'yolov8n.pt')
        self.num_classes = get_val(yolo_config, 'num_classes', 1)

        # Gate 설정
        gate_config = get_val(model_config, 'gate', SimpleNamespace())
        self.gate_basechannels = get_val(gate_config, 'base_channels', 32)
        self.gate_num_layers = get_val(gate_config, 'num_layers', 4)

        # =====================================================================
        # Gate Network 생성
        # =====================================================================
        print(f"\n[Arch2] Initializing Gate Network...")
        self.gate_network = LightweightGate(
            in_channels=3,
            base_channels=self.gate_basechannels,   
            num_layers=self.gate_num_layers
        )

        # =====================================================================
        # SR 모델 생성
        # =====================================================================
        print(f"[Arch2] 선택된 SR 모델: {self.sr_type.upper()}")
        
        if self.sr_type == 'mamba':
            self._init_mamba_sr(model_config)
        else:
            self._init_rfdn_sr(model_config)

        # =====================================================================
        # YOLO Detector 생성
        # =====================================================================
        print(f"[Arch2] Initializing YOLO...")
        self.detector = YOLOWrapper(
            model_path=self.yolo_weights,
            num_classes=self.num_classes,
            device=self.device,
            verbose=False
        )

        # =====================================================================
        # Loss Functions
        # =====================================================================
        self.det_loss_fn = DetectionLoss(self.detector.detection_model)
        self.sr_loss_fn = SRLoss(l1_weight=1.0, charbonnier=True)

        # Gate 통계 추적
        self.register_buffer('gate_running_mean', torch.tensor(0.5))
        self.register_buffer('gate_count', torch.tensor(0))

        # 모델 정보 출력
        total_params = sum(p.numel() for p in self.parameters())
        gate_params = sum(p.numel() for p in self.gate_network.parameters())
        sr_params = sum(p.numel() for p in self.sr_model.parameters())

        print(f"\n[Arch2] ✓ Initialized")
        print(f"  - SR Model: {self.sr_type.upper()}")
        print(f"  - Gate params: {gate_params:,}")
        print(f"  - SR params: {sr_params:,}")
        print(f"  - Total params: {total_params:,}")

    # =========================================================================
    # SR 모델 초기화 헬퍼
    # =========================================================================
    
    def _init_rfdn_sr(self, model_config):
        """RFDN 초기화"""
        rfdn_config = get_val(model_config, 'rfdn', {})
        if isinstance(rfdn_config, dict):
            self.nf = rfdn_config.get('nf', 50)
            self.num_modules = rfdn_config.get('num_modules', 4)
            pretrain_path = rfdn_config.get('pretrain_path', None)
        else:
            self.nf = get_val(rfdn_config, 'nf', 50)
            self.num_modules = get_val(rfdn_config, 'num_modules', 4)
            pretrain_path = get_val(rfdn_config, 'pretrain_path', None)
        
        self.sr_model = RFDN(
            in_channels=3,
            out_channels=3,
            nf=self.nf,
            num_modules=self.num_modules,
            upscale=self.upscale_factor
        )
        
        if pretrain_path:
            self.sr_model.load_pretrained(pretrain_path)
            print(f"[Arch2] RFDN pretrained 로드: {pretrain_path}")
    
    def _init_mamba_sr(self, model_config):
        """MambaSR 초기화"""
        from src.models.sr_models.mamba_sr import MambaSR
        
        mamba_config = get_val(model_config, 'mamba', {})
        if isinstance(mamba_config, dict):
            img_size = mamba_config.get('img_size', 64)
            embed_dim = mamba_config.get('embed_dim', 48)
            d_state = mamba_config.get('d_state', 8)
            depths = mamba_config.get('depths', [5, 5, 5, 5])
            num_heads = mamba_config.get('num_heads', [4, 4, 4, 4])
            window_size = mamba_config.get('window_size', 16)
            pretrain_path = mamba_config.get('pretrain_path', None)
        else:
            img_size = get_val(mamba_config, 'img_size', 64)
            embed_dim = get_val(mamba_config, 'embed_dim', 48)
            d_state = get_val(mamba_config, 'd_state', 8)
            depths = get_val(mamba_config, 'depths', [5, 5, 5, 5])
            num_heads = get_val(mamba_config, 'num_heads', [4, 4, 4, 4])
            window_size = get_val(mamba_config, 'window_size', 16)
            pretrain_path = get_val(mamba_config, 'pretrain_path', None)
        
        self.sr_model = MambaSR(
            scale_factor=self.upscale_factor,
            img_size=img_size,
            embed_dim=embed_dim,
            d_state=d_state,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            pretrained_path=pretrain_path
        )
        
        print(f"[Arch2] MambaSR 초기화 완료")

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(
            self,
            lr_image: torch.Tensor,
            return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """Forward pass"""
        B = lr_image.size(0)

        # 1. Gate 예측
        gate = self.gate_network(lr_image)

        # 2. SR 수행
        sr_image = self.sr_model(lr_image)
        
        # 3. 단순 업샘플 (bypass path)
        upsampled = F.interpolate(
            lr_image,
            scale_factor=self.upscale_factor,
            mode='bilinear',
            align_corners=False
        )
        
        # 4. Soft blending
        gate_expanded = gate.view(B, 1, 1, 1)
        hr_image = gate_expanded * sr_image + (1 - gate_expanded) * upsampled

        # 5. Detection
        if self.training:
            self.detector.train()
            detections = self.detector(hr_image)
        else:
            self.detector.eval()
            detections = self.detector.predict(hr_image)

        # Gate 통계 업데이트
        if self.training:
            with torch.no_grad():
                batch_mean = gate.mean()
                self.gate_running_mean = 0.99 * self.gate_running_mean + 0.01 * batch_mean
                self.gate_count += 1

        result = {
            'hr_image': hr_image,
            'gate': gate, 
            'detections': detections
        }

        if return_intermediates:
            result['sr_image'] = sr_image
            result['upsampled'] = upsampled
            result['lr_image'] = lr_image
            
        return result

    # =========================================================================
    # Loss Computation
    # =========================================================================
    
    def compute_loss(
            self,
            outputs: Dict[str, Any],
            targets: torch.Tensor,
            hr_gt: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Loss 계산
        
        [수정됨] Gradient 연결 개선
        """
        hr_image = outputs['hr_image']
        gate = outputs['gate']

        device = hr_image.device

        # Detection Loss 초기화
        det_loss_dict = {
            'total': torch.tensor(0.0, device=device),
            'box_loss': torch.tensor(0.0, device=device),
            'cls_loss': torch.tensor(0.0, device=device),
            'dfl_loss': torch.tensor(0.0, device=device)
        }

        # Detection Loss 계산
        if targets is not None and len(targets) > 0:
            self.detector.train()
            preds = self.detector(hr_image)
            det_loss_dict = self.det_loss_fn(preds, targets, hr_image)

        det_loss = det_loss_dict['total']

        # SR Loss (선택적) - gradient 보장
        sr_loss = torch.tensor(0.0, device=device)
        
        if hr_gt is not None and self._sr_weight > 0:
            if 'sr_image' in outputs:
                sr_image = outputs['sr_image']
            else:
                sr_image = self.sr_model(outputs.get('lr_image', hr_image))
            
            sr_loss_dict = self.sr_loss_fn(sr_image, hr_gt)
            sr_loss = sr_loss_dict['total']

        # Gate Regularization (선택적) - gradient 연결용 dummy loss
        # Gate가 학습되도록 hr_image에 연결
        gate_reg_loss = gate.mean() * 0.0  # gradient 연결만, 값은 0

        # Total Loss - hr_image를 통해 gate/sr에 gradient 전파
        total_loss = self._det_weight * det_loss + self._sr_weight * sr_loss + gate_reg_loss
        
        # [핵심] hr_image 사용하여 gradient 연결 보장
        # hr_image = gate * sr_image + (1-gate) * upsampled 이므로
        # det_loss가 hr_image 기반이면 gate와 sr_model 모두에 gradient 전파됨
        
        return {
            'total': total_loss,
            'det_loss': det_loss,
            'sr_loss': sr_loss,
            'gate_reg_loss': gate_reg_loss,
            'box_loss': det_loss_dict.get('box_loss', torch.tensor(0.0, device=device)),
            'cls_loss': det_loss_dict.get('cls_loss', torch.tensor(0.0, device=device)),
            'dfl_loss': det_loss_dict.get('dfl_loss', torch.tensor(0.0, device=device)),
            'gate_mean': gate.mean().detach(),
            'gate_std': gate.std().detach()
        }

    # =========================================================================
    # Inference
    # =========================================================================
    
    @torch.no_grad()
    def inference(
        self,
        lr_image: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Dict[str, Any]:
        """추론 모드"""
        self.eval()
        
        result = self.forward(lr_image, return_intermediates=True)
        sr_applied = (result['gate'] > 0.5).float().mean().item()
        
        return {
            'detections': result['detections'],
            'gate': result['gate'],
            'hr_image': result['hr_image'],
            'sr_applied_ratio': sr_applied
        }
    
    # =========================================================================
    # Phase Control
    # =========================================================================
    
    def freeze_sr_and_yolo(self) -> None:
        """Phase 1: Gate만 학습"""
        for param in self.sr_model.parameters():
            param.requires_grad = False
        
        self.detector.freeze()
        self.detector.set_bn_eval()
        
        for param in self.gate_network.parameters():
            param.requires_grad = True
        
        print("[Arch2] Phase 1: Gate only training")
        print(f"  - SR ({self.sr_type}): frozen")
        print(f"  - YOLO: frozen")
        print(f"  - Gate: trainable")
    
    def unfreeze_all(self) -> None:
        """Phase 2: 전체 학습"""
        for param in self.sr_model.parameters():
            param.requires_grad = True
        
        self.detector.unfreeze()
        
        for param in self.gate_network.parameters():
            param.requires_grad = True
        
        print("[Arch2] Phase 2: Full training")
    
    def get_parameter_groups(
        self,
        base_lr: float = 1e-4,
        gate_lr_scale: float = 1.0,
        sr_lr_scale: float = 0.1,
        yolo_lr_scale: float = 0.1
    ) -> List[Dict]:
        """파라미터 그룹 반환"""
        return [
            {
                'params': self.gate_network.parameters(),
                'lr': base_lr * gate_lr_scale,
                'name': 'gate'
            },
            {
                'params': self.sr_model.parameters(),
                'lr': base_lr * sr_lr_scale,
                'name': 'sr'
            },
            {
                'params': self.detector.detection_model.parameters(),
                'lr': base_lr * yolo_lr_scale,
                'name': 'yolo'
            }
        ]
    
    # =========================================================================
    # Info
    # =========================================================================
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """아키텍처 정보"""
        info = super().get_architecture_info()
        
        gate_params = sum(p.numel() for p in self.gate_network.parameters())
        sr_params = sum(p.numel() for p in self.sr_model.parameters())
        yolo_params = sum(p.numel() for p in self.detector.detection_model.parameters())
        
        info.update({
            'architecture': 'Arch2_SoftGate',
            'sr_type': self.sr_type,
            'description': 'Conditional SR with learnable gate network',
            'components': {
                'gate': f'LightweightGate ({gate_params:,} params)',
                'sr_model': f'{self.sr_type.upper()} ({sr_params:,} params)',
                'detector': f'YOLO ({yolo_params:,} params)'
            },
            'gate_running_mean': self.gate_running_mean.item(),
            'upscale_factor': self.upscale_factor
        })
        
        return info
    
    def get_gate_stats(self) -> Dict[str, float]:
        """현재 Gate 통계 반환"""
        return {
            'running_mean': self.gate_running_mean.item(),
            'count': self.gate_count.item()
        }