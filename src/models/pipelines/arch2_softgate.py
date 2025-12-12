"""
=============================================================================
arch2_softgate.py - Architecture 2: SoftGate Pipeline
=============================================================================

[Arch 2 개념]

"모든 이미지에 SR을 적용할 필요가 있을까?"

- 품질 좋은 이미지: SR 없이도 탐지 가능 → SR 스킵 (연산 절약)
- 품질 나쁜 이미지: SR 필요 → SR 적용

[아키텍처]

LR Image [B, 3, H, W]
    │
    ├──────────────────────────┐
    │                          │
    ▼                          ▼
┌─────────┐              ┌───────────┐
│  Gate   │              │   RFDN    │
│ Network │              │ (SR Model)│
└────┬────┘              └─────┬─────┘
     │                         │
     │ gate ∈ [0,1]           │ SR Output
     │                         │
     └──────────┬──────────────┘
                │
                ▼
        ┌───────────────┐
        │  Soft Blend   │
        │               │
        │ out = gate×SR │
        │ + (1-g)×Up    │
        └───────┬───────┘
                │
                ▼
         HR Image (또는 LR 업샘플)
                │
                ▼
        ┌───────────────┐
        │     YOLO      │
        │   Detector    │
        └───────┬───────┘
                │
                ▼
          Detections

[Soft Gate 수식]
output = gate × SR(LR) + (1 - gate) × Bilinear_Upsample(LR)

- gate ≈ 1: SR 결과 사용 (품질 나쁜 이미지)
- gate ≈ 0: 단순 업샘플 (품질 좋은 이미지, SR 스킵)

[장점]
1. 연산량 절약: 품질 좋은 이미지는 SR 스킵
2. End-to-end 학습: Detection loss가 Gate까지 역전파
3. 유연한 판단: Gate가 자동으로 "탐지에 유리한 방향" 학습

[vs Arch 0]
- Arch 0: 항상 SR 적용 (고정)
- Arch 2: 조건부 SR 적용 (적응적)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List

from src.models.pipelines.base_pipeline import BasePipeline
from src.models.sr_models.rfdn import RFDN
from src.models.detectors.yolo_wrapper import YOLOWrapper
from src.models.gates.soft_gate import LightweightGate, softGateMoudle
from src.losses.detection_loss import DetectionLoss
from src.losses.sr_loss import SRLoss

class Arch2SoftGate(BasePipeline):


    def __init__(self, config: Any):
        """
        Args:
            sr_model: SR model
            detector: Detection model
            gate_network: Gate network for fusion
            scale_factor: SR scale factor
            device: Device
        """
        super().__init__(config)
        
        model_config = getattr(config, 'model', config.get('model', {}))
        data_config = getattr(config, 'data', config.get('data',{}))

        rfdn_config = getattr(model_config, 'yolo', model_config.get('rfdn', {}))
        self.nf = getattr(rfdn_config, 'nf', rfdn_config.get('nf', 50))
        self.num_modules = getattr(rfdn_config, 'num_modules', rfdn_config.get('num_modules',4))

        yolo_config = getattr(model_config, 'yolo', model_config.get('yolo', {}))
        self.yolo_weights = getattr(yolo_config, 'weights_path', yolo_config.get('weights_path','yolo11s.pt'))
        self.num_classes = getattr(yolo_config, 'num_classes', yolo_config.get('num_classes', 1))

        gate_config = getattr(model_config, 'gate', model_config.get('gate', {}))
        self.gate_basechannels = getattr(gate_config, 'base_channels', gate_config.get('base_channels',32))
        self.gate_num_layers = getattr(gate_config, 'num_layers', gate_config.get('num_layers', 4))

        self.upscale_factor = getattr(data_config, 'upscale_factor', data_config.get('upscale_factor', 4))

        self.gate_network = LightweightGate(
            in_channels=3,
            base_channels=self.gate_basechannels,   
            num_layers=self.gate_num_layers
        )

        self.sr_model = RFDN(
            in_channels=3,
            out_channels=3,
            nf=self.nf,
            num_modules=self.num_modules,
            upscale=self.upscale_factor
        )

        self.detector = YOLOWrapper(
            model_path = self.yolo_weights,
            num_classes = self.num_classes,
            device = self.device,
            verbose=False
        )

        self.det_loss_fn = DetectionLoss(self.detector.yolo_model)
        self.sr_loss_fn = SRLoss(l1_weight=1.0, use_charbonnier=True)

        self.register_buffer('gate_running_mean', torch.tensor(0.5))
        self.register_buffer('gate_count', torch.tensor(0))

        total_params = sum(p.numel() for p in self.parameters())
        gate_params = sum(p.numel() for p in self.gate_network.parameters())
        sr_params = sum(p.numel() for p in self.sr_model.parameters())

        print(f"\n[Arch2] ✓ Initialized")
        print(f"  - Gate params: {gate_params:,}")
        print(f"  - SR params: {sr_params:,}")
        print(f"  - Total params: {total_params:,}")

#============================================================================#
# Forward
#============================================================================#

    def forward(
            self,
            lr_image: torch.Tensor,
            return_intermediates: bool = False
    ) -> Dict[str, Any]:
        B = lr_image.size(0)

        gate = self.gate_network(lr_image)

        sr_image = self.sr_model(lr_image)
        upsampled = F.interpolate(
            lr_image,
            scale_factor=self.upscale_factor,
            mode = 'bilinear',
            align_corners = False
        )
        gate_expaned = gate.view(B, 1, 1, 1)
        hr_image = gate_expaned * sr_image + (1 - gate_expaned) * upsampled

        if self.training:
            self.detector.train()
            detections = self.detector(hr_image)
        else:
            self.detector.eval()
            detections = self.detector.predict(hr_image)

        if self.training:
            with torch.no_grad():
                batch_mean = gate.mean()
                self.gate_running_mean = 0.99 * self.gate_running_mean + 0.01 * batch_mean
                self.gate_count +=1

        result = {
            'hr_image': hr_image,
            'gate': gate, 
            'detections': detections
        }

        if return_intermediates:
            result['sr_image'] = sr_image
            result['upsampled'] = upsampled
        return result
    
    def forward_with_gate_control(
            self,
            lr_image: torch.Tensor,
            force_gate: Optional[float] = None,
    )-> Dict[str, Any]:
        
        B = lr_image.size(0)

        if force_gate is not None:
            gate = torch.full((B,1), force_gate, device=lr_image.device)
        else:
            gate = self.gate_network(lr_image)
        
        sr_image = self.sr_model(lr_image)
        upsampled = F.interpolate(
            lr_image, 
            scale_factor = self.upscale_factor,
            mode = 'bilinear',
            align_corners = False
        )
        
        gate_expaned = gate.view(B,1,1,1)
        hr_image = gate_expaned * sr_image + (1 - gate_expaned) * upsampled

        self.detector.eval()
        detections = self.detector.predict(hr_image)

        return {
            'hr_image': hr_image,
            'gate': gate,
            'detections': detections,
            'sr_image': sr_image,
            'upsampled': upsampled
        }
#============================================================================#
# Loss Computation
#============================================================================#
    def compute_loss(
            self,
            outputs: Dict[str,Any],
            targets: torch.Tensor,
            hr_gt: Optional[torch.Tensor]= None
    )->Dict[str, torch.Tensor]:
        hr_image = outputs['hr_image']
        gate = outputs['gate']
        detections = outputs['detections']

        device = hr_image.device

        det_loss_dict = {
            'total': torch.tensor(0.0, device=device),
            'box_loss': torch.tensor(0.0, device=device),
            'cls_loss': torch.tensor(0.0, device=device),
            'dfl_loss': torch.tensor(0.0, device=device)
        }

        if targets is not None and len(targets) >0:
            self.detector.train()
            preds = self.detector(hr_image)

            det_loss_dict = self.det_loss_fn(preds, targets, hr_image)

        det_loss = det_loss_dict['total']

        sr_loss = det_loss_dict['total']
        # =====================================================================
        # SR Loss (선택적)
        # =====================================================================
        sr_loss = torch.tensor(0.0, device=device)
        
        if hr_gt is not None and self._sr_weight > 0:
            # SR 출력과 GT 비교
            if 'sr_image' in outputs:
                sr_image = outputs['sr_image']
            else:
                sr_image = self.sr_model(outputs.get('lr_image', hr_image))
            
            sr_loss_dict = self.sr_loss_fn(sr_image, hr_gt)
            sr_loss = sr_loss_dict['total']
        # =====================================================================
        # Gate Regularization (선택적)
        # =====================================================================
        gate_reg_loss = torch.tensor(0.0, device=device)
        
        # Gate가 너무 한쪽으로 치우치지 않도록 약간의 정규화
        # 목표: gate 값이 0.3 ~ 0.7 사이에 분포하도록 유도 (선택적)
        # gate_reg_loss = F.mse_loss(gate.mean(), torch.tensor(0.5, device=device))
        
        # =====================================================================
        # Total Loss
        # =====================================================================
        total_loss = self._det_weight * det_loss + self._sr_weight * sr_loss
        
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
        """
        추론 모드
        
        Returns:
            {
                'detections': 탐지 결과,
                'gate': Gate 값,
                'hr_image': 출력 이미지,
                'sr_applied_ratio': SR 적용 비율
            }
        """
        self.eval()
        
        result = self.forward(lr_image, return_intermediate=True)
        
        # SR 적용 비율 계산
        sr_applied = (result['gate'] > 0.5).float().mean().item()
        
        return {
            'detections': result['detections'],
            'gate': result['gate'],
            'hr_image': result['hr_image'],
            'sr_applied_ratio': sr_applied
        }
    
    # =========================================================================
    # Analysis Methods
    # =========================================================================
    
    def analyze_gate_behavior(
        self,
        dataloader,
        device: str = 'cuda',
        num_batches: int = 50
    ) -> Dict[str, Any]:
        """
        Gate 동작 분석
        
        Returns:
            {
                'gate_mean': 평균 gate 값,
                'gate_std': 표준편차,
                'sr_ratio': SR 적용 비율 (gate > 0.5),
                'gate_histogram': gate 분포
            }
        """
        self.eval()
        gates = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                if isinstance(batch, (list, tuple)):
                    lr_image = batch[0]
                else:
                    lr_image = batch
                
                lr_image = lr_image.to(device)
                gate = self.gate_network(lr_image)
                gates.append(gate.cpu())
        
        gates = torch.cat(gates, dim=0).squeeze()
        
        # 히스토그램
        hist, bin_edges = torch.histogram(gates, bins=10, range=(0., 1.))
        
        return {
            'gate_mean': gates.mean().item(),
            'gate_std': gates.std().item(),
            'sr_ratio': (gates > 0.5).float().mean().item(),
            'bypass_ratio': (gates < 0.5).float().mean().item(),
            'gate_min': gates.min().item(),
            'gate_max': gates.max().item(),
            'histogram': {
                'counts': hist.tolist(),
                'bins': bin_edges.tolist()
            }
        }
    
    def compare_with_without_sr(
        self,
        lr_image: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, Any]:
        """
        SR 적용/미적용 결과 비교 (분석용)
        
        Returns:
            {
                'with_sr': SR 적용 결과,
                'without_sr': SR 미적용 결과,
                'soft_gate': Soft gate 결과,
                'gate_value': 실제 gate 값
            }
        """
        self.eval()
        
        with torch.no_grad():
            # SR 강제 적용
            result_with_sr = self.forward_with_gate_control(lr_image, force_gate=1.0)
            
            # SR 강제 스킵
            result_without_sr = self.forward_with_gate_control(lr_image, force_gate=0.0)
            
            # 정상 동작
            result_soft = self.forward_with_gate_control(lr_image, force_gate=None)
        
        return {
            'with_sr': result_with_sr,
            'without_sr': result_without_sr,
            'soft_gate': result_soft,
            'gate_value': result_soft['gate'].squeeze().tolist()
        }
    
    # =========================================================================
    # Phase Control
    # =========================================================================
    
    def freeze_sr_and_yolo(self) -> None:
        """
        Phase 1: Gate만 학습
        
        SR과 YOLO를 pretrain된 상태로 freeze하고
        Gate만 학습시킬 때 사용
        """
        # SR freeze
        for param in self.sr_model.parameters():
            param.requires_grad = False
        
        # YOLO freeze
        self.detector.freeze()
        self.detector.set_bn_eval()
        
        # Gate unfreeze
        for param in self.gate_network.parameters():
            param.requires_grad = True
        
        print("[Arch2] Phase 1: Gate only training")
        print(f"  - SR: frozen")
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
        """
        파라미터 그룹 반환 (다른 LR 적용용)
        
        Args:
            base_lr: 기본 학습률
            gate_lr_scale: Gate LR 배율
            sr_lr_scale: SR LR 배율
            yolo_lr_scale: YOLO LR 배율
        
        Returns:
            optimizer에 전달할 파라미터 그룹 리스트
        """
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
            'description': 'Conditional SR with learnable gate network',
            'components': {
                'gate': f'LightweightGate ({gate_params:,} params)',
                'sr_model': f'RFDN ({sr_params:,} params)',
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


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Arch2 SoftGate 테스트")
    print("=" * 70)
    
    from types import SimpleNamespace
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Config
    config = SimpleNamespace(
        model=SimpleNamespace(
            rfdn=SimpleNamespace(nf=50, num_modules=4),
            yolo=SimpleNamespace(weights_path="yolov8n.pt", num_classes=80),
            gate=SimpleNamespace(base_channels=32, num_layers=4)
        ),
        data=SimpleNamespace(upscale_factor=4),
        training=SimpleNamespace(sr_weight=0.3, det_weight=0.7),
        device=device
    )
    
    try:
        # 1. 모델 생성
        print("\n[1. 모델 생성]")
        model = Arch2SoftGate(config)
        print("✓ Arch2SoftGate 생성 성공")
        
        # 2. Forward
        print("\n[2. Forward 테스트]")
        lr_image = torch.randn(2, 3, 160, 160, device=device)
        
        model.eval()
        with torch.no_grad():
            result = model(lr_image, return_intermediate=True)
        
        print(f"  LR input: {lr_image.shape}")
        print(f"  HR output: {result['hr_image'].shape}")
        print(f"  Gate values: {result['gate'].squeeze().tolist()}")
        print(f"  SR image: {result['sr_image'].shape}")
        print(f"  Upsampled: {result['upsampled'].shape}")
        
        # 3. Gate 강제 제어
        print("\n[3. Gate 강제 제어 테스트]")
        
        result_sr = model.forward_with_gate_control(lr_image, force_gate=1.0)
        result_bypass = model.forward_with_gate_control(lr_image, force_gate=0.0)
        
        sr_det_count = sum(len(d['boxes']) for d in result_sr['detections'])
        bypass_det_count = sum(len(d['boxes']) for d in result_bypass['detections'])
        
        print(f"  SR 강제 적용: {sr_det_count} detections")
        print(f"  SR 스킵: {bypass_det_count} detections")
        
        # 4. Loss 계산
        print("\n[4. Loss 테스트]")
        
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.2, 0.2],
            [1, 0, 0.3, 0.7, 0.15, 0.25],
        ], device=device)
        
        model.train()
        result = model(lr_image, return_intermediate=True)
        loss_dict = model.compute_loss(result, targets)
        
        print("  Loss 결과:")
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {v.item():.6f}")
        
        # 5. Gradient Flow
        print("\n[5. Gradient Flow 테스트]")
        
        if loss_dict['total'].requires_grad:
            loss_dict['total'].backward()
            
            gate_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                               for p in model.gate_network.parameters())
            sr_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                             for p in model.sr_model.parameters())
            
            print(f"  ✓ Gate gradient: {gate_has_grad}")
            print(f"  ✓ SR gradient: {sr_has_grad}")
        
        # 6. Phase 테스트
        print("\n[6. Phase 테스트]")
        model.freeze_sr_and_yolo()
        
        gate_trainable = sum(p.numel() for p in model.gate_network.parameters() if p.requires_grad)
        sr_trainable = sum(p.numel() for p in model.sr_model.parameters() if p.requires_grad)
        
        print(f"  Gate trainable: {gate_trainable:,}")
        print(f"  SR trainable: {sr_trainable:,}")
        
        # 7. 아키텍처 정보
        print("\n[7. 아키텍처 정보]")
        info = model.get_architecture_info()
        print(f"  Architecture: {info['architecture']}")
        print(f"  Components: {info['components']}")
        
        print("\n" + "=" * 70)
        print("✓ Arch2 SoftGate 테스트 완료!")
        print("=" * 70)
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()

