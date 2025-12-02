"""파이프라인 추상 베이스 클래스

이 모듈은 SR-Detection 통합 파이프라인의 추상 베이스 클래스를 정의합니다.
Arch0(Sequential), Arch2(SoftGate), Arch4(Adaptive), Arch5B(Fusion) 등
모든 아키텍처는 이 인터페이스를 구현해야 합니다.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List


class BasePipeline(ABC, nn.Module):
    """모든 아키텍처 파이프라인이 상속해야 하는 베이스 클래스

    이 클래스는 SR-Detection 파이프라인의 표준 인터페이스를 정의합니다:
    - forward: LR 이미지 → Detection 결과
    - get_loss: 학습용 loss 계산
    - inference: 추론 모드 (후처리 포함)

    Attributes:
        sr_model: Super-Resolution 모델
        detector: Object Detection 모델 (YOLO)
        scale_factor: SR 배율
    """

    def __init__(
        self,
        sr_model: nn.Module,
        detector: nn.Module,
        scale_factor: int = 4,
        device: str = 'cuda'
    ):
        """
        Args:
            sr_model: SR 모델 인스턴스
            detector: Detection 모델 인스턴스
            scale_factor: Super-resolution 배율
            device: 실행 장치 ('cuda' or 'cpu')
        """
        super().__init__()
        self.sr_model = sr_model
        self.detector = detector
        self.scale_factor = scale_factor
        self.device = device

    @abstractmethod
    def forward(self, lr_image: torch.Tensor) -> Dict[str, Any]:
        """LR 이미지 → Detection 결과

        Args:
            lr_image: 저해상도 입력 이미지 [B, 3, H, W]

        Returns:
            outputs: Dictionary containing:
                - 'detections': Detection 결과 (boxes, scores, classes)
                - 'hr_image': SR로 복원된 HR 이미지 (선택적)
                - 'features': 중간 feature maps (선택적, 분석용)
        """
        pass

    @abstractmethod
    def get_loss(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        hr_image: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Loss 계산 (SR + Detection)

        Args:
            outputs: forward()의 출력
            targets: Ground truth targets
                - 'boxes': Detection boxes
                - 'labels': Class labels
            hr_image: Ground truth HR 이미지 (SR loss 계산용)

        Returns:
            loss_dict: Dictionary containing:
                - 'total_loss': 전체 loss
                - 'sr_loss': SR reconstruction loss
                - 'det_loss': Detection loss
                - (선택적) 'aux_loss': 보조 loss
        """
        pass

    def inference(
        self,
        lr_image: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Dict[str, Any]:
        """추론 모드 실행 (후처리 포함)

        Args:
            lr_image: 저해상도 입력 이미지 [B, 3, H, W]
            conf_threshold: Confidence threshold for NMS
            iou_threshold: IoU threshold for NMS

        Returns:
            results: Dictionary containing:
                - 'boxes': NMS 후 최종 boxes [N, 4]
                - 'scores': Confidence scores [N]
                - 'classes': Class labels [N]
                - 'hr_image': SR 복원 이미지 (선택적)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(lr_image)

            # TODO: NMS 후처리 구현
            # 현재는 raw outputs 반환
            return outputs

    def get_architecture_info(self) -> Dict[str, Any]:
        """아키텍처 정보 반환 (디버깅/분석용)

        Returns:
            info: Dictionary with architecture details
        """
        return {
            'pipeline_type': self.__class__.__name__,
            'sr_model': self.sr_model.__class__.__name__,
            'detector': self.detector.__class__.__name__,
            'scale_factor': self.scale_factor,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

    def freeze_sr(self):
        """SR 모델 파라미터 고정 (Detection만 학습)"""
        for param in self.sr_model.parameters():
            param.requires_grad = False
        print("✓ SR model frozen")

    def freeze_detector(self):
        """Detector 파라미터 고정 (SR만 학습)"""
        for param in self.detector.parameters():
            param.requires_grad = False
        print("✓ Detector frozen")

    def unfreeze_all(self):
        """모든 파라미터 학습 가능하도록 설정"""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ All parameters unfrozen")

    def load_checkpoint(self, checkpoint_path: str, strict: bool = True):
        """체크포인트 로드

        Args:
            checkpoint_path: 체크포인트 파일 경로
            strict: 엄격한 키 매칭 여부
        """
        # TODO: 체크포인트 로딩 로직 구현
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        return checkpoint

    def save_checkpoint(self, checkpoint_path: str, **kwargs):
        """체크포인트 저장

        Args:
            checkpoint_path: 저장 경로
            **kwargs: 추가 정보 (epoch, optimizer_state 등)
        """
        # TODO: 체크포인트 저장 로직 구현
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'architecture': self.get_architecture_info(),
            **kwargs
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Saved checkpoint to {checkpoint_path}")
