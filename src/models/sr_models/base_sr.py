"""SR 모델의 추상 베이스 클래스

이 모듈은 모든 Super-Resolution 모델이 상속해야 하는 추상 베이스 클래스를 정의합니다.
RFDN, Mamba-SR, TTST 등 모든 SR 모델은 이 인터페이스를 구현해야 합니다.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class BaseSRModel(ABC, nn.Module):
    """모든 SR 모델이 상속해야 하는 베이스 클래스

    이 클래스는 SR 모델의 표준 인터페이스를 정의합니다:
    - encode: LR 이미지에서 feature 추출
    - decode: feature를 HR 이미지로 복원
    - forward: 전체 SR 파이프라인 (encode + decode)

    Attributes:
        scale_factor (int): SR 배율 (기본값: 4)
        in_channels (int): 입력 채널 수 (RGB=3)
        out_channels (int): 출력 채널 수 (RGB=3)
    """

    def __init__(self, scale_factor: int = 4, in_channels: int = 3, out_channels: int = 3):
        """
        Args:
            scale_factor: Super-resolution 배율 (2, 4, 8 등)
            in_channels: 입력 이미지 채널 수
            out_channels: 출력 이미지 채널 수
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """LR 이미지 → Feature 추출

        Args:
            x: LR 입력 이미지 [B, C, H, W]

        Returns:
            features: 추출된 feature map [B, C', H, W] 또는 [B, C', H*scale, W*scale]
        """
        pass

    @abstractmethod
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Features → HR 이미지 복원

        Args:
            features: encode()로부터 추출된 feature map

        Returns:
            hr_image: 복원된 HR 이미지 [B, C, H*scale, W*scale]
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """전체 SR 수행 (encode + decode)

        Args:
            x: LR 입력 이미지 [B, C, H, W]

        Returns:
            hr_image: 복원된 HR 이미지 [B, C, H*scale, W*scale]
        """
        features = self.encode(x)
        return self.decode(features)

    def get_feature_shapes(self, input_shape: Tuple[int, int, int, int]) -> Dict[str, Tuple]:
        """각 stage의 feature shape 반환 (디버깅/설계용)

        Args:
            input_shape: 입력 shape (B, C, H, W)

        Returns:
            Dictionary with feature shapes at each stage
        """
        # TODO: 각 모델에서 override하여 구현
        return {
            'input': input_shape,
            'encoded': None,  # 모델마다 다름
            'output': (input_shape[0], self.out_channels,
                      input_shape[2] * self.scale_factor,
                      input_shape[3] * self.scale_factor)
        }

    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """사전 학습된 가중치 로드

        Args:
            checkpoint_path: 체크포인트 파일 경로 (.pth)
            strict: 엄격한 키 매칭 여부
        """
        # TODO: 체크포인트 로딩 로직 구현
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=strict)
        print(f"✓ Loaded pretrained weights from {checkpoint_path}")
