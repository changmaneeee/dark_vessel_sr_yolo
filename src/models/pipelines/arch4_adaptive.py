"""
=============================================================================
arch4_adaptive.py - Architecture 4: Adaptive 2-Pass Pipeline (Final Corrected)
=============================================================================
[Arch0와 동일한 좌표계 적용]
1. Pass 1 입력: Upsampled Image (640px) 사용
   - Arch0가 SR(640px) 이미지를 입력받는 것과 동일한 스케일 환경 조성.
   - LR 모델(192px)이라도 큰 이미지를 주면 Recall이 상승함.
2. 좌표 스케일링 삭제:
   - 입력이 640px이므로 출력 좌표도 640px 기준.
   - Arch0처럼 별도의 * scale 연산 없이 바로 사용 가능.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from torchvision.ops import nms, batched_nms
from pathlib import Path

from src.models.pipelines.base_pipeline import BasePipeline
from src.models.sr_models.rfdn import RFDN
from src.models.detectors.yolo_wrapper import YOLOWrapper
from src.losses.detection_loss import DetectionLoss
from src.losses.sr_loss import SRLoss
from types import SimpleNamespace


class Arch4Adaptive(BasePipeline):
    """
    Architecture 4: Adaptive 2-Pass Pipeline
    """

    SUPPORTED_SR_TYPES = ['rfdn', 'mamba']

    def __init__(self, config: Any):
        super().__init__(config)

        def get_val(obj, key, default=None):
            if hasattr(obj, key): return getattr(obj, key)
            elif isinstance(obj, dict): return obj.get(key, default)
            return default

        # Config 파싱
        model_config = get_val(config, 'model', config)
        data_config = get_val(config, 'data', SimpleNamespace())

        self.upscale_factor = get_val(data_config, 'upscale_factor', 4)
        
        # SR 타입 결정
        self.sr_type = get_val(model_config, 'sr_type', 'rfdn').lower()
        if self.sr_type not in self.SUPPORTED_SR_TYPES:
            print(f"[Arch4] ⚠️ Unknown SR type '{self.sr_type}', falling back to RFDN")
            self.sr_type = 'rfdn'

        # YOLO 설정
        yolo_config = get_val(model_config, 'yolo', SimpleNamespace())
        self.yolo_weights_hr = get_val(yolo_config, 'weights_path', 'yolov8n.pt')
        self.yolo_weights_lr = get_val(yolo_config, 'weights_path_lr', None)
        self.num_classes = get_val(yolo_config, 'num_classes', 1)
        
        if self.yolo_weights_lr is None:
            self.yolo_weights_lr = self.yolo_weights_hr
            self.use_dual_yolo = False
        else:
            self.use_dual_yolo = True

        # Adaptive 설정
        adaptive_config = get_val(model_config, 'adaptive', SimpleNamespace())
        self.pass1_conf_threshold = get_val(adaptive_config, 'pass1_conf_threshold', 0.01)
        self.high_conf_threshold = get_val(adaptive_config, 'high_conf_threshold', 0.40)
        self.final_conf_threshold = get_val(adaptive_config, 'final_conf_threshold', 0.25)
        self.nms_iou_threshold = get_val(adaptive_config, 'nms_iou_threshold', 0.45)
        self.sr_on_zero_detection = get_val(adaptive_config, 'sr_on_zero_detection', False)

        # SR 모델 생성
        print(f"\n[Arch4] 선택된 SR 모델: {self.sr_type.upper()}")
        if self.sr_type == 'mamba': self._init_mamba_sr(model_config)
        else: self._init_rfdn_sr(model_config)

        # YOLO Detector 생성
        print(f"[Arch4] Initializing YOLO...")
        self.detector_hr = YOLOWrapper(
            model_path=self.yolo_weights_hr, num_classes=self.num_classes, device=self.device, verbose=False
        )
        print(f"[Arch4] ✓ YOLO (HR/Pass2): {self.yolo_weights_hr}")
        
        if self.use_dual_yolo:
            self.detector_lr = YOLOWrapper(
                model_path=self.yolo_weights_lr, num_classes=self.num_classes, device=self.device, verbose=False
            )
            print(f"[Arch4] ✓ YOLO (LR/Pass1): {self.yolo_weights_lr}")
        else:
            self.detector_lr = self.detector_hr
            print(f"[Arch4] ✓ YOLO (공유): {self.yolo_weights_hr}")
        
        self.detector = self.detector_hr # 기본값

        # Loss Functions
        self.det_loss_fn = DetectionLoss(self.detector_hr.detection_model)
        self.sr_loss_fn = SRLoss(l1_weight=1.0, charbonnier=True)
        self._det_weight = 1.0
        self._sr_weight = 1.0

        # 통계 추적
        self.register_buffer('total_images', torch.tensor(0))
        self.register_buffer('confirmed_count', torch.tensor(0))
        self.register_buffer('full_sr_count', torch.tensor(0))
        self.register_buffer('zero_det_count', torch.tensor(0))

        self.to(self.device)

    def _init_rfdn_sr(self, model_config):
        # (기존 코드 유지 - Arch0와 동일)
        rfdn_config = getattr(model_config, 'rfdn', {})
        if isinstance(rfdn_config, dict):
            self.nf = rfdn_config.get('nf', 50)
            self.num_modules = rfdn_config.get('num_modules', 4)
        else:
            self.nf = getattr(rfdn_config, 'nf', 50)
            self.num_modules = getattr(rfdn_config, 'num_modules', 4)
        
        weights_config = getattr(model_config, 'weights', {})
        self.sr_weights_path = getattr(weights_config, 'sr_model', None) if not isinstance(weights_config, dict) else weights_config.get('sr_model')
        
        self.sr_model = RFDN(in_channels=3, out_channels=3, nf=self.nf, num_modules=self.num_modules, upscale=self.upscale_factor, input_range='0-255')
        
        if self.sr_weights_path and Path(self.sr_weights_path).exists():
            checkpoint = torch.load(self.sr_weights_path, map_location='cpu')
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            self.sr_model.load_state_dict(state_dict, strict=False)
            print(f"[Arch4] ✓ RFDN weights loaded")

    def _init_mamba_sr(self, model_config):
        from src.models.sr_models.mamba_sr import MambaSR
        self.sr_model = MambaSR(scale_factor=self.upscale_factor)

    def _apply_full_sr(self, lr_image: torch.Tensor) -> torch.Tensor:
        lr_255 = lr_image * 255.0
        hr_255 = self.sr_model(lr_255)
        hr_image = torch.clamp(hr_255 / 255.0, 0.0, 1.0)
        return hr_image

    def _classify_image(self, detections: Dict) -> str:
        scores = detections.get('scores', torch.tensor([]))
        if len(scores) == 0: return 'zero_detection'
        
        # pass1_conf(low)보다 크고 high_conf보다 작은 것이 하나라도 있으면 SR
        has_uncertain = ((scores >= self.pass1_conf_threshold) & (scores < self.high_conf_threshold)).any()
        
        if has_uncertain: return 'need_sr'
        else: return 'confirmed'

    # =========================================================================
    # Forward Method (Arch0 스타일로 수정됨)
    # =========================================================================
    @torch.no_grad()
    def forward(self, lr_image: torch.Tensor, return_intermediate: bool = False) -> Dict[str, Any]:
        """
        [Arch0 스타일 수정]
        1. Pass 1 입력: Upsampled Image (640px)
           - Arch0가 640px SR 이미지를 쓰는 것과 스케일을 맞춤.
        2. 좌표 변환: 삭제 (입력이 640px이므로 출력도 640px 기준)
        """
        self.eval()
        B = lr_image.size(0)
        
        # 1. Upsampling (160 -> 640) [Arch0의 SR 역할 대용]
        lr_upsampled = F.interpolate(
            lr_image,
            scale_factor=self.upscale_factor,
            mode='bilinear',
            align_corners=False
        )

        # 2. Pass 1 Detect (Arch0처럼 큰 이미지 입력)
        pass1_detections = self.detector_lr.predict(
            lr_upsampled, # ★ 중요: 큰 이미지 입력
            conf=self.pass1_conf_threshold,
            iou=self.nms_iou_threshold
        )

        # ★ 좌표 스케일링(* scale) 삭제됨 ★
        # Arch0도 여기서 별도 스케일링을 안 함 (이미지가 크니까)

        self.total_images += B
        final_detections = []
        actions_taken = []
        hr_images = []

        for i in range(B):
            det = pass1_detections[i]
            action = self._classify_image(det)
            actions_taken.append(action)
            
            if action == 'confirmed':
                self.confirmed_count += 1
                scores = det['scores']
                if len(scores) > 0:
                    mask = scores >= self.high_conf_threshold
                    final_detections.append({
                        'boxes': det['boxes'][mask],
                        'scores': det['scores'][mask],
                        'classes': det['classes'][mask]
                    })
                else:
                    final_detections.append(det)
                hr_images.append(None)
            
            elif action == 'need_sr':
                self.full_sr_count += 1
                # SR 수행 (Arch0와 동일 과정)
                hr_image = self._apply_full_sr(lr_image[i:i+1])
                hr_images.append(hr_image)
                # Pass 2 재탐지
                pass2_result = self.detector_hr.predict(
                    hr_image, 
                    conf=self.final_conf_threshold, 
                    iou=self.nms_iou_threshold
                )[0]
                final_detections.append(pass2_result)
            
            else: # zero_detection
                self.zero_det_count += 1
                if self.sr_on_zero_detection:
                    hr_image = self._apply_full_sr(lr_image[i:i+1])
                    hr_images.append(hr_image)
                    pass2_result = self.detector_hr.predict(
                        hr_image, 
                        conf=self.final_conf_threshold, 
                        iou=self.nms_iou_threshold
                    )[0]
                    final_detections.append(pass2_result)
                else:
                    hr_images.append(None)
                    final_detections.append({'boxes': torch.tensor([], device=self.device), 'scores': torch.tensor([], device=self.device), 'classes': torch.tensor([], device=self.device)})

        result = {
            'detections': final_detections,
            'actions': actions_taken,
            'stats': self.get_stats()
        }

        if return_intermediate:
            result['pass1_detections'] = pass1_detections
            result['hr_images'] = hr_images
            result['lr_upsampled'] = lr_upsampled

        return result

    # --- Training Methods (학습 시에도 일관성 유지) ---
    def forward_train(self, lr_image: torch.Tensor, hr_gt: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        self.train()
        lr_255 = lr_image * 255.0
        hr_255 = self.sr_model(lr_255)
        hr_image = torch.clamp(hr_255 / 255.0, 0.0, 1.0)
        
        lr_upsampled = F.interpolate(lr_image, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        
        self.detector_hr.train()
        self.detector_lr.train()
        
        detections_hr = self.detector_hr(hr_image)
        # [수정] 학습 시에도 Pass 1은 Upsampled 이미지 사용 (Arch0 스타일)
        detections_lr = self.detector_lr(lr_upsampled)

        return {'hr_image': hr_image, 'lr_upsampled': lr_upsampled, 'detections_hr': detections_hr, 'detections_lr': detections_lr}

    def compute_loss(self, outputs, targets, hr_gt=None, loss_mode='both') -> Dict[str, torch.Tensor]:
        hr_image = outputs['hr_image']
        detections_hr = outputs['detections_hr']
        detections_lr = outputs['detections_lr']
        lr_upsampled = outputs['lr_upsampled'] # Upsampled 이미지 사용
        device = hr_image.device
        
        det_loss_hr = torch.tensor(0.0, device=device)
        if loss_mode in ['hr_only', 'both'] and targets is not None:
            det_loss_hr = self.det_loss_fn(detections_hr, targets, hr_image)['total']
            
        det_loss_lr = torch.tensor(0.0, device=device)
        if loss_mode in ['lr_only', 'both'] and targets is not None:
            # LR Loss는 Upsampled 이미지를 기준으로 계산
            det_loss_lr = self.det_loss_fn(detections_lr, targets, lr_upsampled)['total']
            
        sr_loss = torch.tensor(0.0, device=device)
        if hr_gt is not None:
            sr_loss = self.sr_loss_fn(hr_image, hr_gt)['total']
            
        total_loss = self._det_weight * det_loss_hr + 0.3 * det_loss_lr + self._sr_weight * sr_loss
        return {'total': total_loss, 'det_loss_hr': det_loss_hr, 'det_loss_lr': det_loss_lr, 'sr_loss': sr_loss}

    def set_thresholds(self, pass1_conf=None, high_conf=None, final_conf=None, nms_iou=None, sr_on_zero=None):
        if pass1_conf is not None: self.pass1_conf_threshold = pass1_conf
        if high_conf is not None: self.high_conf_threshold = high_conf
        if final_conf is not None: self.final_conf_threshold = final_conf
        if nms_iou is not None: self.nms_iou_threshold = nms_iou
        if sr_on_zero is not None: self.sr_on_zero_detection = sr_on_zero
        # print(f"[Arch4] Thresholds updated: Pass1={self.pass1_conf_threshold}, High={self.high_conf_threshold}")

    def get_stats(self) -> Dict[str, Any]:
        total = max(self.total_images.item(), 1)
        return {
            'total_images': self.total_images.item(),
            'confirmed': self.confirmed_count.item(),
            'full_sr': self.full_sr_count.item(),
            'zero_det': self.zero_det_count.item(),
            'sr_saved_ratio': (self.confirmed_count.item() + self.zero_det_count.item()) / total * 100
        }
    
    def reset_stats(self) -> None:
        self.total_images.zero_()
        self.confirmed_count.zero_()
        self.full_sr_count.zero_()
        self.zero_det_count.zero_()
        
    def get_architecture_info(self) -> Dict[str, Any]:
        return {'architecture': 'Arch4_Adaptive_Arch0_Style', 'stats': self.get_stats()}
        
    def freeze_yolo(self): pass
    def freeze_sr(self): pass
    def unfreeze_all(self): pass
    def get_parameter_groups(self, base_lr, sr_lr_scale, yolo_lr_scale): return []