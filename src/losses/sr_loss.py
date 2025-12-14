"""
=============================================================================
sr_loss.py - Super-Resolution Loss Functions
=============================================================================

[제공 Loss]
1. CharbonnierLoss: L1보다 이상치에 강건한 loss
2. PerceptualLoss: VGG19 기반 perceptual loss
3. SSIMLoss: 구조적 유사성 loss
4. SRLoss: 위 loss들을 결합한 통합 loss

[사용 예시]
from src.losses.sr_loss import SRLoss, CharbonnierLoss

# 단일 loss
loss_fn = CharbonnierLoss()
loss = loss_fn(pred, target)

# 통합 loss
loss_fn = SRLoss(l1_weight=1.0, perceptual_weight=0.1, use_charbonnier=True)
loss_dict = loss_fn(pred, target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from torchvision import models


# =============================================================================
# Charbonnier Loss
# =============================================================================

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (Smooth L1의 일반화)
    
    [수식]
    L = sqrt((pred - target)² + ε²)
    
    [특징]
    - L1과 L2의 장점 결합
    - 작은 오차: L2처럼 smooth
    - 큰 오차: L1처럼 robust
    - 이상치에 강건함
    
    [vs L1/L2]
    - L1: |x|, 미분 불연속점 존재 (x=0)
    - L2: x², 이상치에 민감
    - Charbonnier: smooth하면서 robust
    
    [파라미터]
    - eps: 작을수록 L1에 가까움, 클수록 L2에 가까움
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.eps_sq = eps * eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 예측 이미지 [B, C, H, W]
            target: GT 이미지 [B, C, H, W]
        
        Returns:
            loss: 스칼라 텐서
        """
        diff = pred - target
        # sqrt(diff² + ε²)
        loss = torch.sqrt(diff * diff + self.eps_sq)
        return loss.mean()


# =============================================================================
# Perceptual Loss (VGG19)
# =============================================================================

class VGGFeatureExtractor(nn.Module):
    """
    VGG19 Feature Extractor
    
    [추출 레이어]
    - relu1_2: 저수준 특징 (엣지, 텍스처)
    - relu2_2: 중수준 특징
    - relu3_4: 고수준 특징 (패턴)
    - relu4_4: 의미적 특징
    - relu5_4: 가장 추상적
    
    [일반적 선택]
    - SR: relu2_2, relu3_4 주로 사용
    - Style Transfer: relu1_1 ~ relu5_1 모두 사용
    """
    
    def __init__(
        self,
        layer_names: List[str] = ['relu2_2', 'relu3_4'],
        use_input_norm: bool = True
    ):
        super().__init__()
        
        self.layer_names = layer_names
        self.use_input_norm = use_input_norm
        
        # ImageNet normalization
        if use_input_norm:
            self.register_buffer(
                'mean',
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                'std',
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )
        
        # VGG19 로드
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # 레이어 이름 → 인덱스 매핑
        self.layer_name_to_idx = {
            'relu1_1': 1,  'relu1_2': 3,
            'relu2_1': 6,  'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
            'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
            'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35,
        }
        
        # 필요한 마지막 레이어까지만 저장
        max_idx = max(self.layer_name_to_idx[name] for name in layer_names)
        self.features = nn.Sequential(*list(vgg19.features.children())[:max_idx + 1])
        
        # Freeze
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 입력 이미지 [B, 3, H, W], 값 범위 [0, 1]
        
        Returns:
            features: {layer_name: feature_tensor}
        """
        # Normalize
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        
        # Extract features
        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # 현재 인덱스가 원하는 레이어인지 확인
            for name, idx in self.layer_name_to_idx.items():
                if i == idx and name in self.layer_names:
                    features[name] = x
        
        return features


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss (VGG Feature Loss)
    
    [수식]
    L_perceptual = Σ w_i × ||φ_i(pred) - φ_i(target)||₁
    
    where φ_i is VGG feature at layer i
    
    [장점]
    - 픽셀 단위가 아닌 "지각적" 유사성 측정
    - 더 자연스러운 SR 결과
    - 텍스처 보존 우수
    
    [단점]
    - 추가 메모리 (VGG forward)
    - ImageNet 도메인에 편향될 수 있음
    """
    
    def __init__(
        self,
        layer_names: List[str] = ['relu2_2', 'relu3_4'],
        layer_weights: Optional[List[float]] = None,
        use_input_norm: bool = True,
        criterion: str = 'l1'
    ):
        super().__init__()
        
        self.layer_names = layer_names
        self.layer_weights = layer_weights or [1.0] * len(layer_names)
        
        assert len(self.layer_names) == len(self.layer_weights)
        
        # Feature extractor
        self.feature_extractor = VGGFeatureExtractor(
            layer_names=layer_names,
            use_input_norm=use_input_norm
        )
        
        # Loss criterion
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        elif criterion == 'charbonnier':
            self.criterion = CharbonnierLoss()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 예측 이미지 [B, 3, H, W]
            target: GT 이미지 [B, 3, H, W]
        
        Returns:
            loss: 스칼라 텐서
        """
        # Feature 추출
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        # 각 레이어별 loss 계산
        total_loss = 0.0
        for name, weight in zip(self.layer_names, self.layer_weights):
            layer_loss = self.criterion(pred_features[name], target_features[name])
            total_loss = total_loss + weight * layer_loss
        
        return total_loss


# =============================================================================
# SSIM Loss
# =============================================================================

def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    """1D Gaussian 커널 생성"""
    x = torch.arange(window_size).float() - window_size // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    """2D Gaussian window 생성"""
    _1D_window = _gaussian_kernel(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True
) -> torch.Tensor:
    """SSIM 계산"""
    
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    """
    SSIM Loss
    
    [수식]
    SSIM(x, y) = (2μxμy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
    
    [Loss]
    L = 1 - SSIM (최대화를 최소화로 변환)
    
    [특징]
    - 구조적 유사성 측정
    - 휘도, 대비, 구조 비교
    - 인간 시각 시스템에 맞춤
    """
    
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = _create_window(window_size, self.channel)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 예측 이미지 [B, C, H, W]
            target: GT 이미지 [B, C, H, W]
        
        Returns:
            loss: 1 - SSIM
        """
        channel = pred.size(1)
        
        if channel != self.channel:
            self.window = _create_window(self.window_size, channel)
            self.channel = channel
        
        window = self.window.to(pred.device).type(pred.dtype)
        
        ssim_val = _ssim(pred, target, window, self.window_size, channel, self.size_average)
        
        return 1 - ssim_val


# =============================================================================
# 통합 SR Loss
# =============================================================================

class SRLoss(nn.Module):
    """
    통합 SR Loss
    
    [구성]
    L_total = w1 × L_pixel + w2 × L_perceptual + w3 × L_ssim
    
    [기본 설정]
    - Pixel loss: Charbonnier (robust)
    - Perceptual: relu2_2, relu3_4
    - SSIM: window_size=11
    
    [사용 예시]
    loss_fn = SRLoss(
        l1_weight=1.0,
        perceptual_weight=0.1,
        ssim_weight=0.0,
        use_charbonnier=True
    )
    
    loss_dict = loss_fn(sr_image, hr_gt)
    total_loss = loss_dict['total']
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.0,
        ssim_weight: float = 0.0,
        use_charbonnier: bool = True,
        perceptual_layers: List[str] = ['relu2_2', 'relu3_4']
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        
        # Pixel loss
        if use_charbonnier:
            self.pixel_loss = CharbonnierLoss()
        else:
            self.pixel_loss = nn.L1Loss()
        
        # Perceptual loss
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss(layer_names=perceptual_layers)
        else:
            self.perceptual_loss = None
        
        # SSIM loss
        if ssim_weight > 0:
            self.ssim_loss = SSIMLoss()
        else:
            self.ssim_loss = None
        
        print(f"[SRLoss] Initialized:")
        print(f"  - Pixel weight: {l1_weight} ({'Charbonnier' if use_charbonnier else 'L1'})")
        print(f"  - Perceptual weight: {perceptual_weight}")
        print(f"  - SSIM weight: {ssim_weight}")
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: SR 이미지 [B, 3, H, W]
            target: HR GT [B, 3, H, W]
        
        Returns:
            loss_dict: {
                'total': 전체 loss,
                'pixel_loss': 픽셀 loss,
                'perceptual_loss': (optional),
                'ssim_loss': (optional)
            }
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Pixel loss
        pixel_loss = self.pixel_loss(pred, target)
        loss_dict['pixel_loss'] = pixel_loss
        total_loss = total_loss + self.l1_weight * pixel_loss
        
        # Perceptual loss
        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(pred, target)
            loss_dict['perceptual_loss'] = perceptual_loss
            total_loss = total_loss + self.perceptual_weight * perceptual_loss
        
        # SSIM loss
        if self.ssim_loss is not None and self.ssim_weight > 0:
            ssim_loss = self.ssim_loss(pred, target)
            loss_dict['ssim_loss'] = ssim_loss
            total_loss = total_loss + self.ssim_weight * ssim_loss
        
        loss_dict['total'] = total_loss
        
        return loss_dict


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SR Loss 테스트")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 더미 데이터
    pred = torch.rand(2, 3, 256, 256, device=device)
    target = torch.rand(2, 3, 256, 256, device=device)
    
    # 1. Charbonnier Loss
    print("\n[1. Charbonnier Loss]")
    loss_fn = CharbonnierLoss()
    loss = loss_fn(pred, target)
    print(f"  Loss: {loss.item():.6f}")
    
    # 2. Perceptual Loss
    print("\n[2. Perceptual Loss]")
    loss_fn = PerceptualLoss(layer_names=['relu2_2', 'relu3_4'])
    loss = loss_fn(pred, target)
    print(f"  Loss: {loss.item():.6f}")
    
    # 3. SSIM Loss
    print("\n[3. SSIM Loss]")
    loss_fn = SSIMLoss()
    loss = loss_fn(pred, target)
    print(f"  Loss: {loss.item():.6f}")
    
    # 4. 통합 SRLoss
    print("\n[4. SRLoss (통합)]")
    loss_fn = SRLoss(
        l1_weight=1.0,
        perceptual_weight=0.1,
        ssim_weight=0.1,
        use_charbonnier=True
    )
    
    loss_dict = loss_fn(pred, target)
    print("  Loss dict:")
    for k, v in loss_dict.items():
        print(f"    {k}: {v.item():.6f}")
    
    # 5. Gradient 테스트
    print("\n[5. Gradient 테스트]")
    pred.requires_grad = True
    loss_dict = loss_fn(pred, target)
    loss_dict['total'].backward()
    print(f"  ✓ Gradient computed: {pred.grad is not None}")
    
    print("\n" + "=" * 70)
    print("✓ SR Loss 테스트 완료!")
    print("=" * 70)
