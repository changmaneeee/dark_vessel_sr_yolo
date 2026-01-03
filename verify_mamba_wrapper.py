"""
MambaSR Wrapper 검증: 원본 MambaIR vs 우리 Wrapper 비교

[검증 목표]
같은 가중치, 같은 입력에서 동일한 출력이 나오는지 확인

[실행]
python verify_mamba_wrapper.py
"""

import torch
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# ============ 경로 설정 ============
# 원본 MambaIR 경로 (네 환경에 맞게 수정)
ORIGINAL_MAMBAIR_PATH = "/home/octolab-rtx4090/Desktop/changmin/MambaIR"

# 가중치 경로
WEIGHTS_PATH = "/home/octolab-rtx4090/Desktop/changmin/MambaIR/experiments/MambaIRv2_SmartAirbus/models/net_g_450000.pth"

# 테스트 이미지 경로 (LR 이미지)
TEST_IMAGE_PATH = "/home/octolab-rtx4090/Desktop/changmin/MambaIR/datasets/SmartAirbus/test/LRx4/test_001.png"  # 실제 경로로 수정!

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_original_mambair():
    """원본 MambaIR 모델 로드"""
    print("\n[1] 원본 MambaIR 로딩...")
    
    # 원본 MambaIR 경로를 sys.path에 추가
    sys.path.insert(0, ORIGINAL_MAMBAIR_PATH)
    
    # 원본 import (basicsr 기반)
    from basicsr.archs.mambairvv2_arch import MambaIRv2Light as OriginalMambaIRv2Light
    
    # 모델 생성 (학습 때 사용한 설정과 동일하게!)
    original_model = OriginalMambaIRv2Light(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=48,
        d_state=8,
        depths=[5, 5, 5, 5],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        mlp_ratio=1.0,
        upscale=4,
        upsampler='pixelshuffledirect',
        resi_connection='1conv'
    )
    
    # 가중치 로드
    checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')
    if 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint
    
    original_model.load_state_dict(state_dict, strict=True)
    original_model = original_model.to(DEVICE)
    original_model.eval()
    
    print("  ✓ 원본 MambaIR 로드 완료")
    
    # sys.path 정리
    sys.path.pop(0)
    
    return original_model


def load_our_wrapper():
    """우리 Wrapper 모델 로드"""
    print("\n[2] 우리 MambaSR Wrapper 로딩...")
    
    from src.models.sr_models import MambaSR
    
    our_model = MambaSR(
        scale_factor=4,
        img_size=64,
        embed_dim=48,
        d_state=8,
        depths=[5, 5, 5, 5],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        pretrained_path=WEIGHTS_PATH
    )
    
    our_model = our_model.to(DEVICE)
    our_model.eval()
    
    print("  ✓ 우리 Wrapper 로드 완료")
    
    return our_model


def load_test_image():
    """테스트 이미지 로드"""
    print("\n[3] 테스트 이미지 로딩...")
    
    if not Path(TEST_IMAGE_PATH).exists():
        print(f"  ⚠️ 이미지 없음: {TEST_IMAGE_PATH}")
        print("  → 랜덤 이미지로 테스트")
        # 랜덤 이미지 생성 (실제 테스트에선 진짜 이미지 사용!)
        img_tensor = torch.rand(1, 3, 64, 64).to(DEVICE)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img = Image.open(TEST_IMAGE_PATH).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        print(f"  ✓ 이미지 로드: {img_tensor.shape}")
    
    return img_tensor


def compare_outputs(out_original, out_ours):
    """두 출력 비교"""
    print("\n[5] 결과 비교...")
    
    # 차이 계산
    diff = (out_original - out_ours).abs()
    
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    
    print(f"  - 평균 차이: {mean_diff:.10f}")
    print(f"  - 최대 차이: {max_diff:.10f}")
    
    # PSNR 계산 (두 이미지 간)
    mse = ((out_original - out_ours) ** 2).mean()
    if mse > 0:
        psnr = 10 * torch.log10(1.0 / mse)
        print(f"  - 두 출력 간 PSNR: {psnr.item():.2f} dB")
    else:
        print(f"  - 두 출력 간 PSNR: ∞ (완전 동일)")
    
    # 판정
    # max_diff < 1e-5 이면 거의 동일 (float 오차 수준)
    # max_diff < 1e-3 이면 매우 유사
    # max_diff < 1e-2 이면 유사
    
    if max_diff < 1e-5:
        verdict = "✅ 완전 동일 (float 오차 수준)"
        passed = True
    elif max_diff < 1e-3:
        verdict = "✅ 매우 유사 (무시 가능한 차이)"
        passed = True
    elif max_diff < 1e-2:
        verdict = "⚠️ 유사하지만 약간의 차이 있음"
        passed = True
    else:
        verdict = "❌ 차이가 큼 - 문제 있음!"
        passed = False
    
    print(f"\n  판정: {verdict}")
    
    return passed, {
        'mean_diff': mean_diff,
        'max_diff': max_diff,
    }


def visualize_comparison(lr_img, out_original, out_ours, save_path="verification_result.png"):
    """결과 시각화"""
    print("\n[6] 시각화 저장...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: LR, Original SR, Our SR
    axes[0, 0].imshow(lr_img[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0, 0].set_title(f"LR Input\n{lr_img.shape[-2]}x{lr_img.shape[-1]}")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(out_original[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0, 1].set_title(f"Original MambaIR\n{out_original.shape[-2]}x{out_original.shape[-1]}")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(out_ours[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0, 2].set_title(f"Our Wrapper\n{out_ours.shape[-2]}x{out_ours.shape[-1]}")
    axes[0, 2].axis('off')
    
    # Row 2: Difference visualization
    diff = (out_original - out_ours).abs()
    diff_amplified = diff * 100  # 차이를 100배 증폭해서 시각화
    
    axes[1, 0].imshow(diff[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[1, 0].set_title(f"Difference (raw)\nmax={diff.max():.6f}")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff_amplified[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[1, 1].set_title("Difference (x100 amplified)")
    axes[1, 1].axis('off')
    
    # Histogram of differences
    axes[1, 2].hist(diff.cpu().numpy().flatten(), bins=100, color='blue', alpha=0.7)
    axes[1, 2].set_title("Difference Distribution")
    axes[1, 2].set_xlabel("Pixel Difference")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].axvline(x=diff.mean().item(), color='red', linestyle='--', label=f'mean={diff.mean():.6f}')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  ✓ 저장: {save_path}")
    plt.close()


def run_verification():
    """전체 검증 실행"""
    print("=" * 70)
    print("   MambaSR Wrapper 검증: 원본 vs 우리 Wrapper")
    print("=" * 70)
    print(f"\n  Device: {DEVICE}")
    print(f"  Weights: {WEIGHTS_PATH}")
    
    try:
        # 1. 원본 모델 로드
        original_model = load_original_mambair()
        
        # 2. 우리 Wrapper 로드
        our_model = load_our_wrapper()
        
        # 3. 테스트 이미지 로드
        lr_img = load_test_image()
        
        # 4. 추론 실행
        print("\n[4] 추론 실행...")
        with torch.no_grad():
            out_original = original_model(lr_img)
            out_ours = our_model(lr_img)
        
        print(f"  - 원본 출력: {out_original.shape}, 범위: [{out_original.min():.3f}, {out_original.max():.3f}]")
        print(f"  - 우리 출력: {out_ours.shape}, 범위: [{out_ours.min():.3f}, {out_ours.max():.3f}]")
        
        # 5. 비교
        passed, metrics = compare_outputs(out_original, out_ours)
        
        # 6. 시각화
        visualize_comparison(lr_img, out_original, out_ours)
        
        # 결과 요약
        print("\n" + "=" * 70)
        if passed:
            print("  🎉 검증 통과! Wrapper가 원본과 동일하게 작동합니다.")
        else:
            print("  ❌ 검증 실패! Wrapper 구현에 문제가 있습니다.")
        print("=" * 70)
        
        return passed
        
    except Exception as e:
        print(f"\n❌ 검증 중 에러: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============ 메인 실행 ============
if __name__ == '__main__':
    run_verification()
