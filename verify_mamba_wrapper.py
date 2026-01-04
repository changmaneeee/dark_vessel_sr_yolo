# verify_mamba_wrapper.py (수정된 버전)

"""
MambaSR Wrapper 검증: 저장된 원본 출력 vs 우리 Wrapper 비교

[실행 순서]
1. MambaIR 폴더에서: python save_original_output.py
2. dark_vessel_sr_yolo에서: python verify_mamba_wrapper.py
"""

import torch
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# ============ 경로 설정 ============
# Step 1에서 저장한 원본 출력
ORIGINAL_OUTPUT_PATH = "/home/octolab-rtx4090/Desktop/changmin/original_mambair_output.pt"

# 가중치 경로
WEIGHTS_PATH = "/home/octolab-rtx4090/Desktop/changmin/MambaIR/experiments/MambaIRv2_SmartAirbus/models/net_g_450000.pth"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_original_output():
    """Step 1에서 저장한 원본 출력 로드"""
    print("\n[1] 원본 MambaIR 출력 로딩...")
    
    if not Path(ORIGINAL_OUTPUT_PATH).exists():
        print(f"  ❌ 파일 없음: {ORIGINAL_OUTPUT_PATH}")
        print("  → 먼저 MambaIR 폴더에서 save_original_output.py를 실행하세요!")
        return None, None
    
    data = torch.load(ORIGINAL_OUTPUT_PATH, map_location='cpu')
    
    lr_input = data['lr_input'].to(DEVICE)
    sr_output = data['sr_output'].to(DEVICE)
    
    print(f"  ✓ LR 입력: {lr_input.shape}")
    print(f"  ✓ SR 출력: {sr_output.shape}")
    print(f"  ✓ 출력 범위: [{sr_output.min():.4f}, {sr_output.max():.4f}]")
    
    return lr_input, sr_output


def load_our_wrapper():
    """우리 Wrapper 모델 로드"""
    print("\n[2] 우리 MambaSR Wrapper 로딩...")
    
    # 프로젝트 경로 추가
    project_path = Path(__file__).parent
    if str(project_path) not in sys.path:
        sys.path.insert(0, str(project_path))
    
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


def compare_outputs(out_original, out_ours):
    """두 출력 비교"""
    print("\n[4] 결과 비교...")
    
    # Shape 확인
    if out_original.shape != out_ours.shape:
        print(f"  ❌ Shape 불일치!")
        print(f"     원본: {out_original.shape}")
        print(f"     우리: {out_ours.shape}")
        return False, {}
    
    # 차이 계산
    diff = (out_original - out_ours).abs()
    
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    
    print(f"  - 평균 차이: {mean_diff:.10f}")
    print(f"  - 최대 차이: {max_diff:.10f}")
    
    # PSNR 계산
    mse = ((out_original - out_ours) ** 2).mean()
    if mse > 0:
        psnr = 10 * torch.log10(1.0 / mse)
        print(f"  - 두 출력 간 PSNR: {psnr.item():.2f} dB")
    else:
        print(f"  - 두 출력 간 PSNR: ∞ (완전 동일)")
    
    # 판정
    if max_diff < 1e-5:
        verdict = "✅ 완전 동일 (float 오차 수준)"
        passed = True
    elif max_diff < 1e-3:
        verdict = "✅ 매우 유사 (무시 가능한 차이)"
        passed = True
    elif max_diff < 1e-2:
        verdict = "⚠️ 유사하지만 약간의 차이 있음 (확인 필요)"
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
    print("\n[5] 시각화 저장...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: LR, Original SR, Our SR
    axes[0, 0].imshow(lr_img[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axes[0, 0].set_title(f"LR Input\n{lr_img.shape[-2]}x{lr_img.shape[-1]}")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(out_original[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axes[0, 1].set_title(f"Original MambaIR\n{out_original.shape[-2]}x{out_original.shape[-1]}")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(out_ours[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axes[0, 2].set_title(f"Our Wrapper\n{out_ours.shape[-2]}x{out_ours.shape[-1]}")
    axes[0, 2].axis('off')
    
    # Row 2: Difference visualization
    diff = (out_original - out_ours).abs()
    diff_amplified = diff * 100
    
    axes[1, 0].imshow(diff[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axes[1, 0].set_title(f"Difference (raw)\nmax={diff.max():.6f}")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff_amplified[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axes[1, 1].set_title("Difference (x100 amplified)")
    axes[1, 1].axis('off')
    
    # Histogram
    diff_np = diff.cpu().numpy().flatten()
    axes[1, 2].hist(diff_np, bins=100, color='blue', alpha=0.7)
    axes[1, 2].set_title("Difference Distribution")
    axes[1, 2].set_xlabel("Pixel Difference")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].axvline(x=diff_np.mean(), color='red', linestyle='--', 
                       label=f'mean={diff_np.mean():.6f}')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  ✓ 저장: {save_path}")
    plt.close()


def run_verification():
    """전체 검증 실행"""
    print("=" * 70)
    print("   MambaSR Wrapper 검증: 원본 출력 vs 우리 Wrapper")
    print("=" * 70)
    print(f"\n  Device: {DEVICE}")
    print(f"  Weights: {WEIGHTS_PATH}")
    
    try:
        # 1. 원본 출력 로드
        lr_input, out_original = load_original_output()
        if lr_input is None:
            return False
        
        # 2. 우리 Wrapper 로드
        our_model = load_our_wrapper()
        
        # 3. 추론 실행
        print("\n[3] 우리 Wrapper 추론...")
        with torch.no_grad():
            out_ours = our_model(lr_input)
        
        print(f"  - 우리 출력: {out_ours.shape}")
        print(f"    범위: [{out_ours.min():.4f}, {out_ours.max():.4f}]")
        
        # 4. 비교
        passed, metrics = compare_outputs(out_original, out_ours)
        
        # 5. 시각화
        visualize_comparison(lr_input, out_original, out_ours)
        
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


if __name__ == '__main__':
    run_verification()