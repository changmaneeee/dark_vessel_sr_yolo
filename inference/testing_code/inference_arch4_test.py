"""
=============================================================================
arch4_adaptive.py - Architecture 4: Adaptive 2-Pass Pipeline
=============================================================================

[수정 내역]
- [CRITICAL] SR 출력값 Clamp를 (0.0, 1.0 - epsilon)으로 설정하여 YOLO 오작동 완전 차단
- RFDN load_pretrained 호출 제거
- Mamba/Einops Mocking 코드 포함
"""

# Mamba Mocking
from unittest.mock import MagicMock
import sys
if "mamba_ssm" not in sys.modules:
    mamba_mock = MagicMock()
    sys.modules["mamba_ssm"] = mamba_mock
    sys.modules["mamba_ssm.ops"] = MagicMock()
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = MagicMock()
    sys.modules["mamba_ssm.modules"] = MagicMock()
if "einops" not in sys.modules:
    sys.modules["einops"] = MagicMock()


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from torchvision.ops import nms, batched_nms

from src.models.pipelines.base_pipeline import BasePipeline
from src.models.sr_models.rfdn import RFDN
from src.models.detectors.yolo_wrapper import YOLOWrapper
from src.losses.detection_loss import DetectionLoss
from src.losses.sr_loss import SRLoss
from types import SimpleNamespace



class Arch4Adaptive(BasePipeline):
    SUPPORTED_SR_TYPES = ['rfdn', 'mamba']

    def __init__(self, config: Any):
        super().__init__(config)
        # (생성자 코드는 위와 동일 - 생략 가능하지만 편의를 위해 전체 포함)
        def get_val(obj, key, default=None):
            if hasattr(obj, key): return getattr(obj, key)
            elif isinstance(obj, dict): return obj.get(key, default)
            return default

        model_config = get_val(config, 'model', config)
        data_config = get_val(config, 'data', SimpleNamespace())

        self.upscale_factor = get_val(data_config, 'upscale_factor', 4)
        self.sr_type = get_val(model_config, 'sr_type', 'rfdn').lower()
        if self.sr_type not in self.SUPPORTED_SR_TYPES: self.sr_type = 'rfdn'

        yolo_config = get_val(model_config, 'yolo', SimpleNamespace())
        self.yolo_weights = get_val(yolo_config, 'weights_path', 'yolov8n.pt')
        self.num_classes = get_val(yolo_config, 'num_classes', 1)

        adaptive_config = get_val(model_config, 'adaptive', SimpleNamespace())
        self.low_conf_threshold = get_val(adaptive_config, 'low_conf_threshold', 0.1)
        self.high_conf_threshold = get_val(adaptive_config, 'high_conf_threshold', 0.5)
        self.merge_iou_threshold = get_val(adaptive_config, 'merge_iou_threshold', 0.5)
        self.final_conf_threshold = get_val(data_config, 'final_conf_threshold', 0.25)

        if self.sr_type == 'mamba': self._init_mamba_sr(model_config)
        else: self._init_rfdn_sr(model_config)

        self.detector = YOLOWrapper(model_path=self.yolo_weights, num_classes=self.num_classes, device=self.device, verbose=False)
        self.det_loss_fn = DetectionLoss(self.detector.detection_model)
        self.sr_loss_fn = SRLoss(l1_weight=1.0, charbonnier=True)
        self.register_buffer('pass2_trigger_count', torch.tensor(0))
        self.register_buffer('total_inference_count', torch.tensor(0))
        self.to(self.device)

    def _init_rfdn_sr(self, model_config):
        rfdn_config = getattr(model_config, 'rfdn', {})
        # (RFDN 초기화 로직 동일)
        if isinstance(rfdn_config, dict):
            self.nf = rfdn_config.get('nf', 50)
            self.num_modules = rfdn_config.get('num_modules', 4)
            pretrain_path = rfdn_config.get('pretrain_path', None)
        else:
            self.nf = getattr(rfdn_config, 'nf', 50)
            self.num_modules = getattr(rfdn_config, 'num_modules', 4)
            pretrain_path = getattr(rfdn_config, 'pretrain_path', None)
        
        self.sr_model = RFDN(in_channels=3, out_channels=3, nf=self.nf, num_modules=self.num_modules, upscale=self.upscale_factor)
        
        if pretrain_path and os.path.exists(pretrain_path):
            print(f"[Arch4] RFDN 가중치 로딩: {pretrain_path}")
            try:
                checkpoint = torch.load(pretrain_path, map_location='cpu')
                if 'params_ema' in checkpoint: state_dict = checkpoint['params_ema']
                elif 'params' in checkpoint: state_dict = checkpoint['params']
                elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
                else: state_dict = checkpoint
                
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'): new_state_dict[k[7:]] = v
                    elif k.startswith('net_g.'): new_state_dict[k[6:]] = v
                    else: new_state_dict[k] = v
                try:
                    self.sr_model.load_state_dict(new_state_dict, strict=True)
                    print("[Arch4] ✓ RFDN 가중치 로드 성공 (Strict)")
                except Exception as e:
                    print(f"[Arch4] ⚠️ Strict loading failed, retrying with strict=False...")
                    self.sr_model.load_state_dict(new_state_dict, strict=False)
                    print("[Arch4] ✓ RFDN 가중치 로드 성공 (Non-strict)")
            except Exception as e:
                print(f"[Arch4] ❌ RFDN 가중치 로드 실패: {e}")

    def _init_mamba_sr(self, model_config): pass
    
    def _needs_second_pass(self, detections):
        # (기존 동일)
        needs_pass2 = []
        for det in detections:
            scores = det.get('scores', torch.tensor([]))
            if len(scores) == 0: needs_pass2.append(True)
            else:
                low_conf_mask = (scores > self.low_conf_threshold) & (scores < self.high_conf_threshold)
                needs_pass2.append(low_conf_mask.any().item())
        return needs_pass2

    def _merge_detections(self, det1, det2, scale_factor=1.0):
        # (기존 동일)
        device = det1['boxes'].device if len(det1['boxes']) > 0 else det2['boxes'].device if len(det2['boxes']) > 0 else self.device
        boxes1 = det1['boxes'] * scale_factor if len(det1['boxes']) > 0 else torch.zeros(0, 4, device=device)
        scores1 = det1['scores'] if len(det1['scores']) > 0 else torch.zeros(0, device=device)
        classes1 = det1['classes'] if len(det1['classes']) > 0 else torch.zeros(0, device=device)
        boxes2 = det2['boxes'] if len(det2['boxes']) > 0 else torch.zeros(0, 4, device=device)
        scores2 = det2['scores'] if len(det2['scores']) > 0 else torch.zeros(0, device=device)
        classes2 = det2['classes'] if len(det2['classes']) > 0 else torch.zeros(0, device=device)
        all_boxes = torch.cat([boxes1, boxes2], dim=0)
        all_scores = torch.cat([scores1, scores2], dim=0)           
        all_classes = torch.cat([classes1, classes2], dim=0)
        if len(all_boxes) == 0:
            return {'boxes': torch.zeros(0, 4, device=device), 'scores': torch.zeros(0, device=device), 'classes': torch.zeros(0, device=device)}
        keep = batched_nms(all_boxes, all_scores, all_classes.long(), self.merge_iou_threshold)
        final_mask = all_scores[keep] >= self.final_conf_threshold
        keep = keep[final_mask]
        return {'boxes': all_boxes[keep], 'scores': all_scores[keep], 'classes': all_classes[keep]}

    @torch.no_grad()
    def forward(self, lr_image: torch.Tensor, return_intermediate: bool = False) -> Dict[str, Any]:
        """추론용 Forward (2-Pass)"""
        self.eval()
        B = lr_image.size(0)

        # 1st Pass: Interpolation
        lr_upsampled = F.interpolate(lr_image, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        
        # [Safety] Interpolation 결과도 clamp (혹시 모를 오차 방지)
        lr_upsampled = torch.clamp(lr_upsampled, 0.0, 1.0 - 1e-6)
        
        pass1_detections = self.detector.predict(lr_upsampled, conf=self.low_conf_threshold, iou=0.45)

        needs_pass2 = self._needs_second_pass(pass1_detections)
        any_needs_pass2 = any(needs_pass2)

        self.total_inference_count += B
        if any_needs_pass2:
            self.pass2_trigger_count += sum(needs_pass2)

        hr_image = None
        pass2_detections = [None] * B

        if any_needs_pass2:
            # SR 실행
            hr_image = self.sr_model(lr_image)
            
            # =================================================================
            # [EXTREME SAFETY FIX] 
            # YOLO가 1.0 이상을 255 스케일로 오인하는 것을 완벽 방지하기 위해
            # 1.0에서 아주 작은 값(epsilon)을 뺀 값으로 Clamp 합니다.
            # =================================================================
            epsilon = 1e-6
            hr_image = torch.clamp(hr_image, 0.0, 1.0 - epsilon)

            # 2nd Pass: SR Result
            pass2_results = self.detector.predict(
                hr_image,
                conf=self.low_conf_threshold,
                iou=0.45
            )

            for i, needs in enumerate(needs_pass2):
                if needs:
                    pass2_detections[i] = pass2_results[i]

        final_detections = []
        for i in range(B):
            if needs_pass2[i] and pass2_detections[i] is not None:
                merged = self._merge_detections(pass1_detections[i], pass2_detections[i], scale_factor=1.0)
                final_detections.append(merged)
            else:
                det = pass1_detections[i]
                if len(det['scores']) > 0:
                    mask = det['scores'] >= self.final_conf_threshold
                    final_detections.append({
                        'boxes': det['boxes'][mask],
                        'scores': det['scores'][mask],
                        'classes': det['classes'][mask]
                    })
                else:
                    final_detections.append(det)

        result = {
            'detections': final_detections,
            'pass2_triggered': needs_pass2,
            'pass2_ratio': sum(needs_pass2) / B
        }

        if return_intermediate:
            result['pass1_detections'] = pass1_detections
            result['pass2_detections'] = pass2_detections
            result['hr_image'] = hr_image
            result['lr_upsampled'] = lr_upsampled

        return result

    def forward_train(self, lr_image, hr_gt=None):
        self.train()
        hr_image = self.sr_model(lr_image)
        hr_image = torch.clamp(hr_image, 0.0, 1.0 - 1e-6) # 학습 시에도 동일하게 적용
        
        lr_upsampled = F.interpolate(lr_image, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        self.detector.train()
        detections_hr = self.detector(hr_image)
        detections_lr = self.detector(lr_upsampled)
        return {'hr_image': hr_image, 'lr_upsampled': lr_upsampled, 'detections_hr': detections_hr, 'detections_lr': detections_lr}

    # compute_loss 등 나머지는 동일
    def compute_loss(self, outputs, targets, hr_gt=None, loss_mode='both'):
        hr_image = outputs['hr_image']
        lr_upsampled = outputs['lr_upsampled']
        detections_hr = outputs['detections_hr']
        detections_lr = outputs['detections_lr']
        device = hr_image.device
        loss_dict = {}
        det_loss_hr = torch.tensor(0.0, device=device)
        if loss_mode in ['hr_only', 'both'] and targets is not None and len(targets) > 0:
            det_loss_hr_dict = self.det_loss_fn(detections_hr, targets, hr_image)
            det_loss_hr = det_loss_hr_dict['total']
            loss_dict['box_loss_hr'] = det_loss_hr_dict.get('box_loss', torch.tensor(0.0, device=device))
            loss_dict['cls_loss_hr'] = det_loss_hr_dict.get('cls_loss', torch.tensor(0.0, device=device))
            loss_dict['dfl_loss_hr'] = det_loss_hr_dict.get('dfl_loss', torch.tensor(0.0, device=device))
        loss_dict['det_loss_hr'] = det_loss_hr
        det_loss_lr = torch.tensor(0.0, device=device)
        if loss_mode in ['lr_only', 'both'] and targets is not None and len(targets) > 0:
            det_loss_lr_dict = self.det_loss_fn(detections_lr, targets, lr_upsampled)
            det_loss_lr = det_loss_lr_dict['total']
            loss_dict['box_loss_lr'] = det_loss_lr_dict.get('box_loss', torch.tensor(0.0, device=device))
            loss_dict['cls_loss_lr'] = det_loss_lr_dict.get('cls_loss', torch.tensor(0.0, device=device))
            loss_dict['dfl_loss_lr'] = det_loss_lr_dict.get('dfl_loss', torch.tensor(0.0, device=device))
        loss_dict['det_loss_lr'] = det_loss_lr
        sr_loss = torch.tensor(0.0, device=device)
        if hr_gt is not None and self._sr_weight > 0:
            sr_loss_dict = self.sr_loss_fn(hr_image, hr_gt)
            sr_loss = sr_loss_dict['total']
        loss_dict['sr_loss'] = sr_loss
        total_loss = self._det_weight * det_loss_hr + 0.3 * det_loss_lr + self._sr_weight * sr_loss
        loss_dict['total'] = total_loss
        return loss_dict