"""
=============================================================================
arch4_adaptive.py - Architecture 4: Adaptive 2-Pass Pipeline
=============================================================================

[Arch 4 개념]

"1차 탐지에서 놓친 객체가 있을 수 있다. SR 후 다시 보자!"

핵심 아이디어:
- 1차 탐지 (LR): 빠르게 탐지
- 결과 분석: 낮은 confidence 객체 있으면 → 2차 진행
- 2차 탐지 (SR→HR): 더 정밀하게 재탐지
- 결과 병합: 1차 고conf + 2차 추가발견

[vs Arch 2 차이점]

| 항목 | Arch 2 (SoftGate) | Arch 4 (Adaptive) |
|------|-------------------|-------------------|
| 판단 근거 | 이미지 자체 | 탐지 결과 (conf) |
| 판단 시점 | SR 적용 전 | 1차 탐지 후 |
| YOLO 횟수 | 1번 | 1~2번 |
| 결과 처리 | 단일 | 병합 (NMS) |
| 목적 | 연산 절약 | FN 감소 |

[아키텍처]

Pass 1: LR → YOLO → 결과 분석
                        │
              ┌─────────┴─────────┐
              │                   │
         낮은 conf 있음      낮은 conf 없음
              │                   │
              ▼                   │
Pass 2: LR → SR → HR → YOLO      │
              │                   │
              ▼                   ▼
         결과 병합 (NMS)    1차 결과 그대로

[장점]
1. FN 감소: 1차에서 놓친 객체를 2차에서 찾음
2. 적응적: 필요할 때만 SR 적용 (연산 효율)
3. 결과 품질: 1차 고conf + 2차 추가발견 병합
"""

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

class Arch4Adaptive(BasePipeline):

    def __init__(self, config: Any):
        super().__init__(config)

        model_config = getattr(config, 'model', config.get('model', {}))
        data_config = getattr(config, 'data', config.get('data', {}))

        rfdn_config = getattr(model_config, 'rfdn', model_config.get('rfdn', {}))
        self.nf = getattr(rfdn_config, 'nf', rfdn_config.get('nf', 50))
        self.num_modules = getattr(rfdn_config, 'num_modules', rfdn_config.get('num_modules', 4))

        yolo_config = getattr(model_config, 'yolo', model_config.get('yolo', {}))
        self.yolo_weights = getattr(yolo_config, 'weights_path', yolo_config.get('weights_path', 'yolov11s.pt'))
        self.num_classes = getattr(yolo_config, 'num_classes', yolo_config.get('num_classes', 1))

        adaptive_config = getattr(model_config, 'adaptive', model_config.get('adaptive', {}))

        self.low_conf_threshold = getattr(adaptive_config, 'low_conf_threshold', adaptive_config.get('low_conf_threshold', 0.1))
        self.high_conf_threshold = getattr(adaptive_config, 'high_conf_threshold', adaptive_config.get('high_conf_threshold', 0.5))

        self.merge_iou_threshold = getattr(adaptive_config, 'merge_iou_threshold', adaptive_config.get('merge_iou_threshold', 0.5))

        self.final_conf_threshold = getattr(data_config, 'final_conf_threshold', data_config.get('final_conf_threshold', 0.25))

        self.upscale_factor = getattr(data_config, 'upscale_factor', data_config.get('upscale_factor', 4))


        #==========================================================================================
        #SR MODEL
        #==========================================================================================

        print(f"\n[Arch4] Initializing RFDN..")
        self.sr_model = RFDN(
            in_channels=3,
            out_channels=3, 
            nf=self.nf, 
            num_modules=self.num_modules,   
            upscale_factor=self.upscale_factor
        )

        #==========================================================================================
        #YOLO MODEL
        #==========================================================================================
        print(f"\n[Arch4] Initializing YOLO")

        self.detector = YOLOWrapper(
            model_path = self.yolo_weights,
            num_classes = self.num_classes,
            device=self.device,
            verbose = False
        )

        #==========================================================================================
        #LOSS FUNCTIONS
        #==========================================================================================

        self.det_loss_fn = DetectionLoss(self.detector.detection_model)
        self.sr_loss_fn = SRLoss(l1_weight=1.0, use_charbonnier=False)

        self.register_buffer('pass2_trigger_count', torch.tensor(0))
        self.register_buffer('total_inference_count', torch.tensor(0))

        #Model Information

        sr_params = sum(p.numel() for p in self.sr_model.parameters())
        yolo_params = sum(p.numel() for p in self.detector.detection_model.parameters())
        total_params = sr_params + yolo_params

        print(f"\n[Arch4] Model initialized:")
        print(f"  - SR params: {sr_params:,}")
        print(f"  - YOLO params: {yolo_params:,}")
        print(f"  - Total params: {total_params:,}")
        print(f"  - Low conf threshold: {self.low_conf_threshold}")
        print(f"  - High conf threshold: {self.high_conf_threshold}")

        #==========================================================================================
        # Core Logic: 2-Pass Detection
        #=========================================================================================

    def _needs_second_paa(self, detections: List[Dict]) -> List[bool]:
        """
        Arg: 
            Detection: 1st detection result list
                
        Returns:
            List of booleans indicating if 2nd pass is needed per image
        """
        needs_pass2 = []

        for det in detections:
            scores = det.get('scores', torch.tensor([]))

            if len(scores) == 0:
                needs_pass2.append(True)

            else:
                low_conf_mask = (scores > self.low_conf_threshold) & (scores < self.high_conf_threshold)
                has_low_conf = low_conf_mask.any().item()
                needs_pass2.append(has_low_conf)

        return needs_pass2
    
    def _merge_detections(
            selfm,
            det1: Dict[str, torch.Tensor],
            det2: Dict[str, torch.Tensor],
            scale_factors: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        
        device = det1['boxes'].device if len(det1['boxes']) > 0 else \
        det2['boxes'].device if len(det2['boxes']) > 0 else 'cpu'

        boxes1 = det1['boxes']*scale_factors if len(det1['boxes']) > 0 else torch.zeros(0,4, device = device)
        scores1 = det1['scores'] if len(det1['scores']) > 0 else torch.zeros(0, device = device)
        classes1 = det1['classes'] if len(det1['classes']) > 0 else torch.zeros(0, device = device)

        boxes2 = det2['boxes'] if len(det2['boxes']) > 0 else torch.zeros(0,4, device = device)
        scores2 = det2['scores'] if len(det2['scores']) > 0 else torch.zeros(0, device = device)
        classes2 = det2['classes'] if len(det2['classes']) > 0 else torch.zeros(0, device = device)

        all_boxes = torch.cat([boxes1, boxes2], dim=0)
        all_scores = torch.cat([scores1, scores2], dim=0)           
        all_classes = torch.cat([classes1, classes2], dim=0)

        if len(all_boxes) == 0:
            return {
                'boxes': torch.zeros(0, 4, device=device),
                'scores': torch.zeros(0, device=device),
                'classes': torch.zeros(0, device=device)
            }
        
        keep = batched_nms(
        all_boxes,
        all_scores,
        all_classes.long(),
        self.merge_iou_threshold
        )


        final_mask = all_scores[keep] >= self.final_conf_threshold
        keep = keep[final_mask]
        
        return {
            'boxes': all_boxes[keep],
            'scores': all_scores[keep],
            'classes': all_classes[keep]
        }
    

    #==========================================================================================
    # Forward Method
    #==========================================================================================

    @torch.no_grad()
    def forward(
        self,
        lr_image: torch.Tensor,
        return_intermediate: bool = False
    ) -> Dict[str, Any]:

        self.eval()
        B = lr_image.size(0)

        lr_upsampled = F.interpolate(
            lr_image,
            scale_factor = self.upscale_factor,
            mode = 'bilinear',
            align_corners = False
        )

        pass1_detections = self.detector.predict(
            lr_upsampled,
            conf = self.low_conf_threshold,
            iou = 0.45
        )

        needs_pass2 = self._needs_second_paa(pass1_detections)
        any_needs_pass2 = any(needs_pass2)

        self.total_inference_count += B
        if any_needs_pass2:
            self.pass2_trigger_count += sum(needs_pass2)

        hr_imgae = None
        pass2_detections = [None]*B

        if any_needs_pass2:
            hr_image = self.sr_model(lr_image)

            pass2_results = self.detector.predict(
                hr_image,
                conf = self.low_conf_threshold,
                iou = 0.45
            )

            for i, needs in enumerate(needs_pass2):
                if needs:
                    pass2_detections[i] = pass2_results[i]

        final_detections = []

        for i in range(B):

            if needs_pass2[i] and pass2_detections[i] is not None:
                merged = self._merge_detections(
                    pass1_detections[i],
                    pass2_detections[i],
                    scale_factors = 1.0
                )
                final_detections.append(merged)
            else:
                det = pass1_detections[i]
                if len(det['scores']) >0:
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
            'pass2_ratio': sum(needs_pass2)/B
        }

        if return_intermediate:
            result['pass1_detections'] = pass1_detections
            result['pass2_detections'] = pass2_detections
            result['hr_image'] = hr_image
            result['lr_upsampled'] = lr_upsampled

        return result
    

    #==========================================================================================
    # Training Forward
    #=========================================================================================

    def forward_train(
        self,
        lr_image: torch.Tensor,
        hr_gt: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        
        self.train()

        hr_image = self.sr_model(lr_image)
        lr_upsampled = F.interpolate(
            lr_image,
            scale_factor = self.upscale_factor,
            mode = 'bilinear',
            align_corners = False
        )


        self.detector.train()
        detection_hr = self.detector(hr_image)
        detection_lr = self.detector(lr_upsampled)

        return {
            'hr_image': hr_image,
            'lr_upsampled': lr_upsampled,
            'detection_hr': detection_hr
        }
    
    #==========================================================================================
    # Loss Calculation
    #==========================================================================================

    def compute_loss(
            self,
            outputs: Dict[str, Any],
            targets: torch.Tensor,
            hr_gt: Optional[torch.Tensor] = None,
            loss_mode: str = 'both'
    )-> Dict[str, torch.Tensor]:

        loss_dict = {}

        hr_image = outputs['hr_image']
        lr_upsampled = outputs['lr_upsampled']
        detections_hr = outputs['detection_hr']
        detection_lr = outputs['detection_lr']

        device = hr_image.device
        loss_dict = {}
        total_loss = torch.tensor(0.0, device = device)

        if loss_mode in ['hr_only', 'both'] and targets is not None and len(targets) > 0:
            det_loss_hr_dict = self.det_loss_fn(detections_hr, targets, hr_image)
            det_loss_hr = det_loss_hr_dict['total']

            loss_dict['box_loss_hr'] = det_loss_hr_dict.get('box_loss', torch.tensor(0.0, device=device))
            loss_dict['cls_loss_hr'] = det_loss_hr_dict.get('cls_loss', torch.tensor(0.0, device=device))
            loss_dict['dfl_loss_hr'] = det_loss_hr_dict.get('dfl_loss', torch.tensor(0.0, device=device))
        loss_dict['det_loss_hr'] = det_loss_hr


        det_loss_lr = torch.tensor(0.0, device=device)

        if loss_mode in ['lr_only', 'both'] and targets is not None and len(targets) > 0:
            det_loss_lr_dict = self.det_loss_fn(detection_lr, targets, lr_upsampled)
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

        total_loss = (
            self._det_weight * det_loss_hr + 0.3*det_loss_lr + self._sr_weight * sr_loss
        )

        loss_dict['total'] = total_loss

        return loss_dict
    
    #==========================================================================================
    # Inference
    #=========================================================================================

    @torch.no_grad()
    def inference(
        self,
        lr_image: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    )->Dict[str, Any]:
        
        original_final_conf = self.final_conf_threshold
        self.final_conf_threshold = conf_threshold
        
        result = self.forward(lr_image, return_intermediate=True)

        self.final_conf_threshold = original_final_conf

        return result
    
    #==========================================================================================
    # Analysis Methods
    #==========================================================================================

    def analyze_pass2_behavior(
        self,
        dataloader,
        device: str = 'cuda',
        num_batches: int = 50
    )->Dict[str, Any]:

        self.eval()
        
        stats = {
            'pass_triggered': 0,
            'total_images': 0,
            'pass1_det_counts': [],
            'pass2_det_counts': [],
            'merged_det_counts': []
        }

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    lr_image = batch[0]
                else:
                    lr_image = batch
                
                lr_image = lr_image.to(device)
                result = self.forward(lr_image, return_intermediate=True)

                B = lr_image.size(0)
                stats['total_images'] += B
                stats['pass2_triggered'] += sum(result['pass2_triggered'])

                for j in range(B):
                    p1_count = len(result['pass1_detections'][j]['boxes'])
                    stats['pass1_det_counts'].append(p1_count)

                    if result['pass2_detections'][j] is not None:
                        p2_count = len(result['pass2_detections'][j]['boxes'])
                    else:
                        p2_count = 0
                    stats['pass2_det_counts'].append(p2_count)

                    merged_count = len(result['detections'][j]['boxes'])
                    stats['merged_det_counts'].append(merged_count)

        return {
            'pass2_ratio': stats['pass2_triggered'] / max(stats['total_images'], 1),
            'avg_pass1_detections': sum(stats['pass1_det_counts']) / max(len(stats['pass1_det_counts']), 1),
            'avg_pass2_detections': sum(stats['pass2_det_counts']) / max(len(stats['pass2_det_counts']), 1),
            'avg_merged_detections': sum(stats['merged_det_counts']) / max(len(stats['merged_det_counts']), 1),
            'total_images': stats['total_images']
        }            

    def compare_single_vs_dual_pass(
            self,
            lr_image: torch.Tensor,
            targets: Optional[torch.Tensor] = None
    )->Dict[str, Any]:

        self.eval()
        with torch.no_grad():
            lr_upsampled = F.interpolate(
                lr_image,
                scale_factor = self.upscale_factor,
                mode = 'bilinear',
                align_corners = False
            )
        
        hr_image = self.sr_model(lr_image)
        single_lr = self.detector.predict(lr_upsampled, conf=self.final_conf_threshold)
        single_hr = self.detector.predict(hr_image, conf=self.final_conf_threshold)
        dual_result  = self.forward(lr_image, return_intermediate=True)

        return {
            'single_pass_lr': single_lr,
            'single_pass_hr': single_hr,
            'dual_pass': dual_result['detections'],
            'pass2_triggered': dual_result['pass2_triggered'],
            'hr_image': hr_image,
            'lr_upsampled': lr_upsampled            
        }
    # =========================================================================
    # Phase Control
    # =========================================================================
    
    def freeze_yolo(self) -> None:
        """YOLO Freeze (SR만 학습)"""
        self.detector.freeze()
        self.detector.set_bn_eval()
        
        for param in self.sr_model.parameters():
            param.requires_grad = True
        
        print("[Arch4] YOLO frozen, SR trainable")
    
    def freeze_sr(self) -> None:
        """SR Freeze (YOLO만 학습)"""
        for param in self.sr_model.parameters():
            param.requires_grad = False
        
        self.detector.unfreeze()
        
        print("[Arch4] SR frozen, YOLO trainable")
    
    def unfreeze_all(self) -> None:
        """전체 Unfreeze"""
        for param in self.sr_model.parameters():
            param.requires_grad = True
        
        self.detector.unfreeze()
        
        print("[Arch4] All trainable")
    
    def get_parameter_groups(
        self,
        base_lr: float = 1e-4,
        sr_lr_scale: float = 1.0,
        yolo_lr_scale: float = 0.1
    ) -> List[Dict]:
        """파라미터 그룹 반환"""
        return [
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
    # Threshold Control
    # =========================================================================
    
    def set_thresholds(
        self,
        low_conf: Optional[float] = None,
        high_conf: Optional[float] = None,
        merge_iou: Optional[float] = None,
        final_conf: Optional[float] = None
    ) -> None:
        """Threshold 조정"""
        if low_conf is not None:
            self.low_conf_threshold = low_conf
        if high_conf is not None:
            self.high_conf_threshold = high_conf
        if merge_iou is not None:
            self.merge_iou_threshold = merge_iou
        if final_conf is not None:
            self.final_conf_threshold = final_conf
        
        print(f"[Arch4] Thresholds updated:")
        print(f"  - Low conf: {self.low_conf_threshold}")
        print(f"  - High conf: {self.high_conf_threshold}")
        print(f"  - Merge IoU: {self.merge_iou_threshold}")
        print(f"  - Final conf: {self.final_conf_threshold}")
    
    # =========================================================================
    # Info
    # =========================================================================
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """아키텍처 정보"""
        info = super().get_architecture_info()
        
        sr_params = sum(p.numel() for p in self.sr_model.parameters())
        yolo_params = sum(p.numel() for p in self.detector.detection_model.parameters())
        
        info.update({
            'architecture': 'Arch4_Adaptive_2Pass',
            'description': 'Adaptive 2-pass detection with result merging',
            'components': {
                'sr_model': f'RFDN ({sr_params:,} params)',
                'detector': f'YOLO ({yolo_params:,} params)'
            },
            'thresholds': {
                'low_conf': self.low_conf_threshold,
                'high_conf': self.high_conf_threshold,
                'merge_iou': self.merge_iou_threshold,
                'final_conf': self.final_conf_threshold
            },
            'pass2_stats': {
                'triggered': self.pass2_trigger_count.item(),
                'total': self.total_inference_count.item(),
                'ratio': self.pass2_trigger_count.item() / max(self.total_inference_count.item(), 1)
            }
        })
        
        return info
    
    def get_pass2_stats(self) -> Dict[str, float]:
        """2차 탐지 통계"""
        total = max(self.total_inference_count.item(), 1)
        return {
            'pass2_triggered': self.pass2_trigger_count.item(),
            'total_inferences': self.total_inference_count.item(),
            'pass2_ratio': self.pass2_trigger_count.item() / total
        }
    
    def reset_stats(self) -> None:
        """통계 리셋"""
        self.pass2_trigger_count.zero_()
        self.total_inference_count.zero_()


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Arch4 Adaptive 2-Pass 테스트")
    print("=" * 70)
    
    from types import SimpleNamespace
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Config
    config = SimpleNamespace(
        model=SimpleNamespace(
            rfdn=SimpleNamespace(nf=50, num_modules=4),
            yolo=SimpleNamespace(weights_path="yolov8n.pt", num_classes=80),
            adaptive=SimpleNamespace(
                low_conf_threshold=0.1,
                high_conf_threshold=0.5,
                merge_iou_threshold=0.5,
                final_conf_threshold=0.25
            )
        ),
        data=SimpleNamespace(upscale_factor=4),
        training=SimpleNamespace(sr_weight=0.3, det_weight=0.7),
        device=device
    )
    
    try:
        # 1. 모델 생성
        print("\n[1. 모델 생성]")
        model = Arch4Adaptive(config)
        print("✓ Arch4Adaptive 생성 성공")
        
        # 2. Forward (추론)
        print("\n[2. Forward 테스트 (추론)]")
        lr_image = torch.randn(2, 3, 160, 160, device=device)
        
        result = model.forward(lr_image, return_intermediate=True)
        
        print(f"  LR input: {lr_image.shape}")
        print(f"  Pass 2 triggered: {result['pass2_triggered']}")
        print(f"  Pass 2 ratio: {result['pass2_ratio']:.2%}")
        
        for i, det in enumerate(result['detections']):
            print(f"  Image {i}: {len(det['boxes'])} detections")
        
        # 3. Forward (학습)
        print("\n[3. Forward 테스트 (학습)]")
        model.train()
        train_result = model.forward_train(lr_image)
        
        print(f"  HR image: {train_result['hr_image'].shape}")
        print(f"  LR upsampled: {train_result['lr_upsampled'].shape}")
        
        # 4. Loss 계산
        print("\n[4. Loss 테스트]")
        
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.2, 0.2],
            [1, 0, 0.3, 0.7, 0.15, 0.25],
        ], device=device)
        
        loss_dict = model.compute_loss(train_result, targets)
        
        print("  Loss 결과:")
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {v.item():.6f}")
        
        # 5. Gradient Flow
        print("\n[5. Gradient Flow 테스트]")
        
        if loss_dict['total'].requires_grad:
            loss_dict['total'].backward()
            
            sr_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                             for p in model.sr_model.parameters())
            yolo_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                               for p in model.detector.detection_model.parameters())
            
            print(f"  ✓ SR gradient: {sr_has_grad}")
            print(f"  ✓ YOLO gradient: {yolo_has_grad}")
        
        # 6. 비교 분석
        print("\n[6. Single vs Dual Pass 비교]")
        
        model.eval()
        comparison = model.compare_single_vs_dual_pass(lr_image)
        
        for i in range(lr_image.size(0)):
            lr_count = len(comparison['single_pass_lr'][i]['boxes'])
            hr_count = len(comparison['single_pass_hr'][i]['boxes'])
            dual_count = len(comparison['dual_pass'][i]['boxes'])
            
            print(f"  Image {i}:")
            print(f"    - LR only: {lr_count} detections")
            print(f"    - HR only: {hr_count} detections")
            print(f"    - Dual pass: {dual_count} detections")
        
        # 7. 통계
        print("\n[7. Pass 2 통계]")
        stats = model.get_pass2_stats()
        print(f"  Pass 2 triggered: {stats['pass2_triggered']}")
        print(f"  Total inferences: {stats['total_inferences']}")
        print(f"  Pass 2 ratio: {stats['pass2_ratio']:.2%}")
        
        # 8. 아키텍처 정보
        print("\n[8. 아키텍처 정보]")
        info = model.get_architecture_info()
        print(f"  Architecture: {info['architecture']}")
        print(f"  Thresholds: {info['thresholds']}")
        
        print("\n" + "=" * 70)
        print("✓ Arch4 Adaptive 테스트 완료!")
        print("=" * 70)
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()
