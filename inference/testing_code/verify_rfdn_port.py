#!/usr/bin/env python
"""
=============================================================================
test_rfdn_standalone.py - RFDN 단독 테스트
=============================================================================

[목적]
1. RFDN 모델 자체가 정상 동작하는지 확인
2. PSNR 측정으로 SR 품질 검증
3. 출력 범위 확인 (버그 있으면 비정상 범위)

[정상 기준]
- PSNR: 28~35 dB (Bicubic보다 높아야 함)
- 출력 범위: 입력과 비슷 (0~1 또는 0~255)

[비정상 징후]
- PSNR: < 10 dB
- 출력 범위: [-1000, +1000] 이상

사용법:
    python test_rfdn_standalone.py \
        --lr_root /home/changmin/smart_airbus_data_lr \
        --hr_root /home/changmin/smart_airbus_data \
        --weights /path/to/rfdn_weights.pt \
        --num_samples 10
"""

import sys
from unittest.mock import MagicMock

print("[System] Mamba 라이브러리 우회(Mocking) 설정 중...")

# 1. mamba_ssm 가짜 모듈 생성
mamba_mock = MagicMock()
sys.modules["mamba_ssm"] = mamba_mock
sys.modules["mamba_ssm.ops"] = MagicMock()
sys.modules["mamba_ssm.ops.selective_scan_interface"] = MagicMock()
sys.modules["mamba_ssm.modules"] = MagicMock()
sys.modules["mamba_ssm.modules.mamba_simple"] = MagicMock()

# 2. causal_conv1d 가짜 모듈 생성 (설치 실패 원인 제거)
sys.modules["causal_conv1d"] = MagicMock()
sys.modules["causal_conv1d.causal_conv1d_fn"] = MagicMock()

# 3. einops 가짜 모듈 (없을 경우 대비)
sys.modules["einops"] = MagicMock()

print("[System] 우회 설정 완료. RFDN 모드로 실행합니다.")

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# PSNR 계산
try:
    from skimage.metrics import peak_signal_noise_ratio as calc_psnr
    from skimage.metrics import structural_similarity as calc_ssim
except ImportError:
    def calc_psnr(img1, img2, data_range=1.0):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(data_range ** 2 / mse)
    
    def calc_ssim(img1, img2, data_range=1.0, channel_axis=None):
        return 0.0  # placeholder


def load_image_tensor(path: Path, normalize: bool = True) -> torch.Tensor:
    """
    이미지 로드
    
    Args:
        path: 이미지 경로
        normalize: True면 0-1, False면 0-255
    
    Returns:
        [1, 3, H, W] 텐서
    """
    img = Image.open(path).convert('RGB')
    img_np = np.array(img).astype(np.float32)
    
    if normalize:
        img_np = img_np / 255.0
    
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def bicubic_upsample(lr: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """Bicubic 업샘플링 (baseline)"""
    return F.interpolate(lr, scale_factor=scale, mode='bicubic', align_corners=False)


def main():
    parser = argparse.ArgumentParser(description='RFDN Standalone Test')
    
    parser.add_argument('--lr_root', type=str, required=True,
                        help='LR dataset root')
    parser.add_argument('--hr_root', type=str, required=True,
                        help='HR dataset root')
    parser.add_argument('--weights', type=str, default=None,
                        help='RFDN weights path (optional)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of test samples')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"🔍 RFDN 단독 테스트")
    print(f"{'='*70}")
    print(f"[Device] {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
    
    # =========================================================================
    # 1. RFDN 모델 로드
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"📦 RFDN 모델 로드")
    print(f"{'='*70}")
    
    # 프로젝트 경로 추가
    project_root = Path(args.lr_root).parent.parent / 'dark_vessel_sr_yolo'
    if project_root.exists():
        sys.path.insert(0, str(project_root))
        print(f"  Project path: {project_root}")
    
    from src.models.sr_models.rfdn import RFDN
    
    model = RFDN(
        in_channels=3,
        out_channels=3,
        nf=50,
        num_modules=4,
        upscale=4
    )
    
    # 가중치 로드 (있으면)
    if args.weights and Path(args.weights).exists():
        print(f"  Loading weights: {args.weights}")
        state_dict = torch.load(args.weights, map_location='cpu')
        
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # 키 확인
        print(f"  Weight keys (first 5): {list(state_dict.keys())[:5]}")
        
        model.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Weights loaded")
    else:
        print(f"  ⚠️ No weights loaded (using random init)")
    
    model = model.to(device)
    model.eval()
    
    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")
    
    # =========================================================================
    # 2. 모델 구조 확인
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"🔧 모델 구조 확인")
    print(f"{'='*70}")
    
    # c 레이어 확인 (LeakyReLU 있어야 함)
    print(f"  model.c: {model.c}")
    
    # RFDB forward 확인 (residual 여부)
    print(f"  model.B1 type: {type(model.B1)}")
    
    # =========================================================================
    # 3. 테스트 이미지 로드
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"📂 테스트 이미지 로드")
    print(f"{'='*70}")
    
    lr_root = Path(args.lr_root)
    hr_root = Path(args.hr_root)
    
    lr_img_dir = lr_root / 'images' / 'val'
    hr_img_dir = hr_root / 'images' / 'val'
    
    img_paths = sorted(list(lr_img_dir.glob('*.jpg')) + list(lr_img_dir.glob('*.png')))
    img_paths = img_paths[:args.num_samples]
    
    print(f"  LR dir: {lr_img_dir}")
    print(f"  HR dir: {hr_img_dir}")
    print(f"  Test samples: {len(img_paths)}")
    
    # =========================================================================
    # 4. PSNR 테스트
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"📊 PSNR 테스트 시작")
    print(f"{'='*70}")
    
    results = []
    
    print(f"\n{'이미지':<20} {'SR Range':<25} {'SR PSNR':<12} {'Bicubic PSNR':<12} {'차이':<10}")
    print("-" * 80)
    
    for img_path in img_paths:
        # Load images
        lr_image = load_image_tensor(img_path, normalize=True).to(device)
        
        hr_path = hr_img_dir / img_path.name
        if not hr_path.exists():
            print(f"  ⚠️ HR not found: {hr_path.name}")
            continue
        
        hr_image = load_image_tensor(hr_path, normalize=True).to(device)
        
        # SR
        with torch.no_grad():
            sr_image = model(lr_image)
        
        # Bicubic baseline
        bicubic_image = bicubic_upsample(lr_image, scale=4)
        
        # 크기 맞추기
        if sr_image.shape[-2:] != hr_image.shape[-2:]:
            sr_image = F.interpolate(sr_image, size=hr_image.shape[-2:], 
                                     mode='bilinear', align_corners=False)
        if bicubic_image.shape[-2:] != hr_image.shape[-2:]:
            bicubic_image = F.interpolate(bicubic_image, size=hr_image.shape[-2:],
                                          mode='bilinear', align_corners=False)
        
        # numpy 변환
        sr_np = sr_image.squeeze().cpu().numpy().transpose(1, 2, 0)
        hr_np = hr_image.squeeze().cpu().numpy().transpose(1, 2, 0)
        bicubic_np = bicubic_image.squeeze().cpu().numpy().transpose(1, 2, 0)
        
        # 출력 범위 확인
        sr_min, sr_max = sr_np.min(), sr_np.max()
        sr_range = f"[{sr_min:.2f}, {sr_max:.2f}]"
        
        # Clamp for PSNR (0-1 범위로)
        sr_np_clamped = np.clip(sr_np, 0, 1)
        bicubic_np_clamped = np.clip(bicubic_np, 0, 1)
        
        # PSNR 계산
        psnr_sr = calc_psnr(hr_np, sr_np_clamped, data_range=1.0)
        psnr_bicubic = calc_psnr(hr_np, bicubic_np_clamped, data_range=1.0)
        diff = psnr_sr - psnr_bicubic
        
        results.append({
            'name': img_path.stem[:15],
            'sr_min': sr_min,
            'sr_max': sr_max,
            'psnr_sr': psnr_sr,
            'psnr_bicubic': psnr_bicubic,
            'diff': diff
        })
        
        # 출력
        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
        print(f"{img_path.stem[:18]:<20} {sr_range:<25} {psnr_sr:<12.2f} {psnr_bicubic:<12.2f} {diff_str:<10}")
    
    # =========================================================================
    # 5. 결과 요약
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"📈 결과 요약")
    print(f"{'='*70}")
    
    if results:
        avg_psnr_sr = np.mean([r['psnr_sr'] for r in results])
        avg_psnr_bicubic = np.mean([r['psnr_bicubic'] for r in results])
        avg_diff = np.mean([r['diff'] for r in results])
        
        avg_sr_min = np.mean([r['sr_min'] for r in results])
        avg_sr_max = np.mean([r['sr_max'] for r in results])
        
        print(f"\n  평균 SR PSNR:      {avg_psnr_sr:.2f} dB")
        print(f"  평균 Bicubic PSNR: {avg_psnr_bicubic:.2f} dB")
        print(f"  평균 차이:         {avg_diff:+.2f} dB")
        print(f"  평균 출력 범위:    [{avg_sr_min:.2f}, {avg_sr_max:.2f}]")
        
        # 진단
        print(f"\n{'='*70}")
        print(f"🔍 진단 결과")
        print(f"{'='*70}")
        
        issues = []
        
        # 출력 범위 체크
        if avg_sr_min < -10 or avg_sr_max > 10:
            issues.append(f"❌ 출력 범위 비정상! [{avg_sr_min:.1f}, {avg_sr_max:.1f}]")
            issues.append(f"   → RFDB.forward()에서 이중 residual 버그 가능성")
        
        # PSNR 체크
        if avg_psnr_sr < 15:
            issues.append(f"❌ PSNR 매우 낮음! ({avg_psnr_sr:.1f} dB)")
            issues.append(f"   → 모델 구조 또는 가중치 문제")
        
        # SR vs Bicubic 비교
        if avg_diff < -1:
            issues.append(f"❌ SR이 Bicubic보다 {-avg_diff:.1f} dB 낮음!")
            issues.append(f"   → 모델이 오히려 이미지를 악화시킴")
        
        if issues:
            print("\n  🔴 문제 발견:")
            for issue in issues:
                print(f"    {issue}")
            
            print("\n  📝 수정 필요 사항:")
            print("    1. rfdn.py의 RFDB.forward() 마지막 줄 확인")
            print("       - 'return out_fused + input' → 'return out_fused'")
            print("    2. rfdn.py의 self.c에 LeakyReLU 추가 확인")
        else:
            print("\n  ✅ RFDN 정상 동작!")
            print(f"    - PSNR: {avg_psnr_sr:.2f} dB (Bicubic 대비 +{avg_diff:.2f} dB)")
            print(f"    - 출력 범위: 정상")
    
    print(f"\n{'='*70}")
    print(f"✅ 테스트 완료!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()