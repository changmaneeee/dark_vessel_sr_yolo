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

[지원 SR 모델]
- RFDN: 경량, 빠름 (기본)
- MambaSR: 고성능, Mamba 기반

[아키텍처]

Pass 1: LR → YOLO → 결과 분석
                        │
              ┌─────────┴─────────┐
              │                   │
         낮은 conf 있음      낮은 conf 없음
              │                   │
              ▼                   │
Pass 2: LR → SR → HR → YOLO       │
              │                   │
              ▼                   ▼
         결과 병합 (NMS)    1차 결과 그대로
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
from types import SimpleNamespace


class Arch4Adaptive(BasePipeline):
    """
    Architecture 4: Adaptive 2-Pass Pipeline
    
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

        # Data 설정
        self.upscale_factor = get_val(data_config, 'upscale_factor', 4)
        
        # SR 타입 결정
        self.sr_type = get_val(model_config, 'sr_type', 'rfdn').lower()
        
        if self.sr_type not in self.SUPPORTED_SR_TYPES:
            print(f"[Arch4] ⚠️ Unknown SR type '{self.sr_type}', falling back to RFDN")
            self.sr_type = 'rfdn'

        # YOLO 설정
        yolo_config = get_val(model_config, 'yolo', SimpleNamespace())
        self.yolo_weights = get_val(yolo_config, 'weights_path', 'yolov8n.pt')
        self.num_classes = get_val(yolo_config, 'num_classes', 1)

        # Adaptive 설정
        adaptive_config = get_val(model_config, 'adaptive', SimpleNamespace())
        self.low_conf_threshold = get_val(adaptive_config, 'low_conf_threshold', 0.1)
        self.high_conf_threshold = get_val(adaptive_config, 'high_conf_threshold', 0.5)
        self.merge_iou_threshold = get_val(adaptive_config, 'merge_iou_threshold', 0.5)
        self.final_conf_threshold = get_val(data_config, 'final_conf_threshold', 0.25)

        # =====================================================================
        # SR 모델 생성
        # =====================================================================
        print(f"\n[Arch4] 선택된 SR 모델: {self.sr_type.upper()}")
        
        if self.sr_type == 'mamba':
            self._init_mamba_sr(model_config)
        else:
            self._init_rfdn_sr(model_config)

        # =====================================================================
        # YOLO Detector 생성
        # =====================================================================
        print(f"[Arch4] Initializing YOLO...")
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

        # 통계 추적
        self.register_buffer('pass2_trigger_count', torch.tensor(0))
        self.register_buffer('total_inference_count', torch.tensor(0))

        # 모델 정보 출력
        sr_params = sum(p.numel() for p in self.sr_model.parameters())
        yolo_params = sum(p.numel() for p in self.detector.detection_model.parameters())
        total_params = sr_params + yolo_params

        print(f"\n[Arch4] ✓ Model initialized:")
        print(f"  - SR Model: {self.sr_type.upper()}")
        print(f"  - SR params: {sr_params:,}")
        print(f"  - YOLO params: {yolo_params:,}")
        print(f"  - Total params: {total_params:,}")
        print(f"  - Low conf threshold: {self.low_conf_threshold}")
        print(f"  - High conf threshold: {self.high_conf_threshold}")

    # =========================================================================
    # SR 모델 초기화 헬퍼
    # =========================================================================
    
    def _init_rfdn_sr(self, model_config):
        """RFDN 초기화"""
        rfdn_config = getattr(model_config, 'rfdn', {})
        if isinstance(rfdn_config, dict):
            self.nf = rfdn_config.get('nf', 50)
            self.num_modules = rfdn_config.get('num_modules', 4)
            pretrain_path = rfdn_config.get('pretrain_path', None)
        else:
            self.nf = getattr(rfdn_config, 'nf', 50)
            self.num_modules = getattr(rfdn_config, 'num_modules', 4)
            pretrain_path = getattr(rfdn_config, 'pretrain_path', None)
        
        self.sr_model = RFDN(
            in_channels=3,
            out_channels=3,
            nf=self.nf,
            num_modules=self.num_modules,
            upscale=self.upscale_factor
        )
        
        if pretrain_path:
            self.sr_model.load_pretrained(pretrain_path)
            print(f"[Arch4] RFDN pretrained 로드: {pretrain_path}")
    
    def _init_mamba_sr(self, model_config):
        """MambaSR 초기화"""
        from src.models.sr_models.mamba_sr import MambaSR
        
        mamba_config = getattr(model_config, 'mamba', {})
        if isinstance(mamba_config, dict):
            img_size = mamba_config.get('img_size', 64)
            embed_dim = mamba_config.get('embed_dim', 48)
            d_state = mamba_config.get('d_state', 8)
            depths = mamba_config.get('depths', [5, 5, 5, 5])
            num_heads = mamba_config.get('num_heads', [4, 4, 4, 4])
            window_size = mamba_config.get('window_size', 16)
            pretrain_path = mamba_config.get('pretrain_path', None)
        else:
            img_size = getattr(mamba_config, 'img_size', 64)
            embed_dim = getattr(mamba_config, 'embed_dim', 48)
            d_state = getattr(mamba_config, 'd_state', 8)
            depths = getattr(mamba_config, 'depths', [5, 5, 5, 5])
            num_heads = getattr(mamba_config, 'num_heads', [4, 4, 4, 4])
            window_size = getattr(mamba_config, 'window_size', 16)
            pretrain_path = getattr(mamba_config, 'pretrain_path', None)
        
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
        
        print(f"[Arch4] MambaSR 초기화 완료")

    # =========================================================================
    # Core Logic: 2-Pass Detection
    # =========================================================================

    def _needs_second_pass(self, detections: List[Dict]) -> List[bool]:
        """
        2차 탐지 필요 여부 판단
        
        Args: 
            detections: 1차 탐지 결과 리스트
                
        Returns:
            각 이미지별 2차 탐지 필요 여부
        """
        needs_pass2 = []

        for det in detections:
            scores = det.get('scores', torch.tensor([]))

            if len(scores) == 0:
                # 탐지된 것이 없으면 2차 필요
                needs_pass2.append(True)
            else:
                # 낮은 confidence 탐지가 있으면 2차 필요
                low_conf_mask = (scores > self.low_conf_threshold) & (scores < self.high_conf_threshold)
                has_low_conf = low_conf_mask.any().item()
                needs_pass2.append(has_low_conf)

        return needs_pass2
    
    def _merge_detections(
        self,
        det1: Dict[str, torch.Tensor],
        det2: Dict[str, torch.Tensor],
        scale_factor: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        두 탐지 결과 병합 (NMS 적용)
        
        Args:
            det1: 1차 탐지 결과
            det2: 2차 탐지 결과
            scale_factor: 좌표 스케일 (보통 1.0)
        """
        device = det1['boxes'].device if len(det1['boxes']) > 0 else \
                 det2['boxes'].device if len(det2['boxes']) > 0 else 'cpu'

        # 결과 추출
        boxes1 = det1['boxes'] * scale_factor if len(det1['boxes']) > 0 else torch.zeros(0, 4, device=device)
        scores1 = det1['scores'] if len(det1['scores']) > 0 else torch.zeros(0, device=device)
        classes1 = det1['classes'] if len(det1['classes']) > 0 else torch.zeros(0, device=device)

        boxes2 = det2['boxes'] if len(det2['boxes']) > 0 else torch.zeros(0, 4, device=device)
        scores2 = det2['scores'] if len(det2['scores']) > 0 else torch.zeros(0, device=device)
        classes2 = det2['classes'] if len(det2['classes']) > 0 else torch.zeros(0, device=device)

        # 병합
        all_boxes = torch.cat([boxes1, boxes2], dim=0)
        all_scores = torch.cat([scores1, scores2], dim=0)           
        all_classes = torch.cat([classes1, classes2], dim=0)

        if len(all_boxes) == 0:
            return {
                'boxes': torch.zeros(0, 4, device=device),
                'scores': torch.zeros(0, device=device),
                'classes': torch.zeros(0, device=device)
            }
        
        # Batched NMS
        keep = batched_nms(
            all_boxes,
            all_scores,
            all_classes.long(),
            self.merge_iou_threshold
        )

        # Confidence threshold 적용
        final_mask = all_scores[keep] >= self.final_conf_threshold
        keep = keep[final_mask]
        
        return {
            'boxes': all_boxes[keep],
            'scores': all_scores[keep],
            'classes': all_classes[keep]
        }

    # =========================================================================
    # Forward Method
    # =========================================================================

    @torch.no_grad()
    def forward(
        self,
        lr_image: torch.Tensor,
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """
        추론용 Forward (2-Pass)
        
        Args:
            lr_image: LR 입력 이미지
            return_intermediate: 중간 결과 반환 여부
        """
        self.eval()
        B = lr_image.size(0)

        # LR 업샘플 (1차 탐지용)
        lr_upsampled = F.interpolate(
            lr_image,
            scale_factor=self.upscale_factor,
            mode='bilinear',
            align_corners=False
        )

        # 1차 탐지 (LR)
        pass1_detections = self.detector.predict(
            lr_upsampled,
            conf=self.low_conf_threshold,
            iou=0.45
        )

        # 2차 탐지 필요 여부 판단
        needs_pass2 = self._needs_second_pass(pass1_detections)
        any_needs_pass2 = any(needs_pass2)

        # 통계 업데이트
        self.total_inference_count += B
        if any_needs_pass2:
            self.pass2_trigger_count += sum(needs_pass2)

        hr_image = None
        pass2_detections = [None] * B

        # 2차 탐지 (필요한 경우)
        if any_needs_pass2:
            hr_image = self.sr_model(lr_image)

            pass2_results = self.detector.predict(
                hr_image,
                conf=self.low_conf_threshold,
                iou=0.45
            )

            for i, needs in enumerate(needs_pass2):
                if needs:
                    pass2_detections[i] = pass2_results[i]

        # 결과 병합
        final_detections = []

        for i in range(B):
            if needs_pass2[i] and pass2_detections[i] is not None:
                merged = self._merge_detections(
                    pass1_detections[i],
                    pass2_detections[i],
                    scale_factor=1.0
                )
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

    # =========================================================================
    # Training Forward
    # =========================================================================

    def forward_train(
        self,
        lr_image: torch.Tensor,
        hr_gt: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        학습용 Forward
        
        Args:
            lr_image: LR 입력
            hr_gt: HR GT (SR Loss용)
        """
        self.train()

        hr_image = self.sr_model(lr_image)
        lr_upsampled = F.interpolate(
            lr_image,
            scale_factor=self.upscale_factor,
            mode='bilinear',
            align_corners=False
        )

        self.detector.train()
        detections_hr = self.detector(hr_image)
        detections_lr = self.detector(lr_upsampled)

        return {
            'hr_image': hr_image,
            'lr_upsampled': lr_upsampled,
            'detections_hr': detections_hr,
            'detections_lr': detections_lr
        }

    # =========================================================================
    # Loss Calculation
    # =========================================================================

    def compute_loss(
        self,
        outputs: Dict[str, Any],
        targets: torch.Tensor,
        hr_gt: Optional[torch.Tensor] = None,
        loss_mode: str = 'both'
    ) -> Dict[str, torch.Tensor]:
        """
        Loss 계산
        
        Args:
            outputs: forward_train() 결과
            targets: Detection GT
            hr_gt: HR GT (SR Loss용)
            loss_mode: 'hr_only', 'lr_only', 'both'
        """
        hr_image = outputs['hr_image']
        lr_upsampled = outputs['lr_upsampled']
        detections_hr = outputs['detections_hr']
        detections_lr = outputs['detections_lr']

        device = hr_image.device
        loss_dict = {}

        # HR Detection Loss
        det_loss_hr = torch.tensor(0.0, device=device)
        if loss_mode in ['hr_only', 'both'] and targets is not None and len(targets) > 0:
            det_loss_hr_dict = self.det_loss_fn(detections_hr, targets, hr_image)
            det_loss_hr = det_loss_hr_dict['total']
            loss_dict['box_loss_hr'] = det_loss_hr_dict.get('box_loss', torch.tensor(0.0, device=device))
            loss_dict['cls_loss_hr'] = det_loss_hr_dict.get('cls_loss', torch.tensor(0.0, device=device))
            loss_dict['dfl_loss_hr'] = det_loss_hr_dict.get('dfl_loss', torch.tensor(0.0, device=device))
        loss_dict['det_loss_hr'] = det_loss_hr

        # LR Detection Loss
        det_loss_lr = torch.tensor(0.0, device=device)
        if loss_mode in ['lr_only', 'both'] and targets is not None and len(targets) > 0:
            det_loss_lr_dict = self.det_loss_fn(detections_lr, targets, lr_upsampled)
            det_loss_lr = det_loss_lr_dict['total']
            loss_dict['box_loss_lr'] = det_loss_lr_dict.get('box_loss', torch.tensor(0.0, device=device))
            loss_dict['cls_loss_lr'] = det_loss_lr_dict.get('cls_loss', torch.tensor(0.0, device=device))
            loss_dict['dfl_loss_lr'] = det_loss_lr_dict.get('dfl_loss', torch.tensor(0.0, device=device))
        loss_dict['det_loss_lr'] = det_loss_lr

        # SR Loss
        sr_loss = torch.tensor(0.0, device=device)
        if hr_gt is not None and self._sr_weight > 0:
            sr_loss_dict = self.sr_loss_fn(hr_image, hr_gt)
            sr_loss = sr_loss_dict['total']
        loss_dict['sr_loss'] = sr_loss

        # Total Loss
        total_loss = self._det_weight * det_loss_hr + 0.3 * det_loss_lr + self._sr_weight * sr_loss
        loss_dict['total'] = total_loss

        return loss_dict

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
        original_final_conf = self.final_conf_threshold
        self.final_conf_threshold = conf_threshold
        
        result = self.forward(lr_image, return_intermediate=True)

        self.final_conf_threshold = original_final_conf

        return result

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def analyze_pass2_behavior(
        self,
        dataloader,
        device: str = 'cuda',
        num_batches: int = 50
    ) -> Dict[str, Any]:
        """Pass2 동작 분석"""
        self.eval()
        
        stats = {
            'pass2_triggered': 0,
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

    # =========================================================================
    # Phase Control
    # =========================================================================
    
    def freeze_yolo(self) -> None:
        """YOLO Freeze (SR만 학습)"""
        self.detector.freeze()
        self.detector.set_bn_eval()
        
        for param in self.sr_model.parameters():
            param.requires_grad = True
        
        print(f"[Arch4] YOLO frozen, SR ({self.sr_type}) trainable")
    
    def freeze_sr(self) -> None:
        """SR Freeze (YOLO만 학습)"""
        for param in self.sr_model.parameters():
            param.requires_grad = False
        
        self.detector.unfreeze()
        
        print(f"[Arch4] SR ({self.sr_type}) frozen, YOLO trainable")
    
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
            'sr_type': self.sr_type,
            'description': 'Adaptive 2-pass detection with result merging',
            'components': {
                'sr_model': f'{self.sr_type.upper()} ({sr_params:,} params)',
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
    print("Arch4 Adaptive 2-Pass 테스트 (RFDN + MambaSR)")
    print("=" * 70)
    
    from types import SimpleNamespace
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test 1: RFDN
    print("\n" + "=" * 50)
    print("[TEST 1] Arch4 + RFDN")
    print("=" * 50)
    
    config_rfdn = SimpleNamespace(
        model=SimpleNamespace(
            sr_type='rfdn',
            rfdn=SimpleNamespace(nf=50, num_modules=4),
            yolo=SimpleNamespace(weights_path="yolov8n.pt", num_classes=80),
            adaptive=SimpleNamespace(
                low_conf_threshold=0.1,
                high_conf_threshold=0.5,
                merge_iou_threshold=0.5
            )
        ),
        data=SimpleNamespace(upscale_factor=4, final_conf_threshold=0.25),
        training=SimpleNamespace(sr_weight=0.3, det_weight=0.7),
        device=device
    )
    
    try:
        model_rfdn = Arch4Adaptive(config_rfdn)
        
        lr_image = torch.randn(2, 3, 160, 160, device=device)
        result = model_rfdn.forward(lr_image, return_intermediate=True)
        
        print(f"  ✓ Forward 성공!")
        print(f"    - Pass2 triggered: {result['pass2_triggered']}")
        print(f"    - Pass2 ratio: {result['pass2_ratio']:.2%}")
        
    except Exception as e:
        print(f"  ✗ RFDN 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: MambaSR
    print("\n" + "=" * 50)
    print("[TEST 2] Arch4 + MambaSR")
    print("=" * 50)
    
    config_mamba = SimpleNamespace(
        model=SimpleNamespace(
            sr_type='mamba',
            mamba=SimpleNamespace(
                img_size=64,
                embed_dim=48,
                d_state=8,
                depths=[5, 5, 5, 5],
                num_heads=[4, 4, 4, 4],
                window_size=16
            ),
            yolo=SimpleNamespace(weights_path="yolov8n.pt", num_classes=80),
            adaptive=SimpleNamespace(
                low_conf_threshold=0.1,
                high_conf_threshold=0.5,
                merge_iou_threshold=0.5
            )
        ),
        data=SimpleNamespace(upscale_factor=4, final_conf_threshold=0.25),
        training=SimpleNamespace(sr_weight=0.3, det_weight=0.7),
        device=device
    )
    
    try:
        model_mamba = Arch4Adaptive(config_mamba)
        
        lr_image = torch.randn(2, 3, 160, 160, device=device)
        result = model_mamba.forward(lr_image, return_intermediate=True)
        
        print(f"  ✓ Forward 성공!")
        print(f"    - Pass2 triggered: {result['pass2_triggered']}")
        print(f"    - Pass2 ratio: {result['pass2_ratio']:.2%}")
        
    except Exception as e:
        print(f"  ✗ MambaSR 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✓ Arch4 Adaptive 테스트 완료!")
    print("=" * 70)