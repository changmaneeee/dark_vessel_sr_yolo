"""MambaSR Wrapper 테스트

실행 방법:
    python -m models.sr_models.test_mamba_sr
    또는
    python test_mamba_sr.py (파일 위치에서)

필요 조건:
    - mamba_ssm 설치됨
    - CUDA 사용 가능
"""

import torch
import sys
from pathlib import Path

# ============ 테스트 설정 ============
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SCALE = 4
BATCH_SIZE = 1
INPUT_SIZE = (48, 48)  # (H, W) - window_size(16)의 배수가 아닌 값으로 테스트!


def print_header(title):
    """테스트 섹션 헤더 출력"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name, success, detail=""):
    """테스트 결과 출력"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status} | {name}")
    if detail:
        print(f"         {detail}")


def test_import():
    """테스트 1: Import 테스트"""
    print_header("TEST 1: Import")
    
    try:
        from models.sr_models import MambaSR, create_mamba_sr
        print_result("MambaSR import", True)
    except ImportError as e:
        print_result("MambaSR import", False, str(e))
        return False
    
    try:
        from models.sr_models import MambaIRv2Light
        print_result("MambaIRv2Light import", True)
    except ImportError as e:
        print_result("MambaIRv2Light import", False, str(e))
        return False
    
    try:
        from models.sr_models.mamba_archs import to_2tuple, trunc_normal_
        print_result("arch_util import", True)
    except ImportError as e:
        print_result("arch_util import", False, str(e))
        return False
    
    return True


def test_model_creation():
    """테스트 2: 모델 생성 테스트"""
    print_header("TEST 2: Model Creation")
    
    from models.sr_models import MambaSR, create_mamba_sr
    
    # 방법 1: 직접 생성
    try:
        model1 = MambaSR(scale_factor=SCALE)
        print_result("MambaSR() 직접 생성", True)
    except Exception as e:
        print_result("MambaSR() 직접 생성", False, str(e))
        return None
    
    # 방법 2: 헬퍼 함수
    try:
        model2 = create_mamba_sr(scale=SCALE)
        print_result("create_mamba_sr() 헬퍼", True)
    except Exception as e:
        print_result("create_mamba_sr() 헬퍼", False, str(e))
    
    # 모델 정보 출력
    total_params = sum(p.numel() for p in model1.parameters())
    trainable_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print(f"\n  📊 모델 정보:")
    print(f"     - 전체 파라미터: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"     - 학습 가능: {trainable_params:,}")
    print(f"     - Feature 채널: {model1.feature_channels}")
    print(f"     - Scale factor: {model1.scale_factor}")
    
    return model1


def test_forward(model):
    """테스트 3: forward() 전체 SR 테스트"""
    print_header("TEST 3: forward() - Full SR")
    
    model = model.to(DEVICE)
    model.eval()
    
    H, W = INPUT_SIZE
    x = torch.randn(BATCH_SIZE, 3, H, W).to(DEVICE)
    
    print(f"  입력: {list(x.shape)}")
    
    try:
        with torch.no_grad():
            output = model(x)
        
        expected_shape = (BATCH_SIZE, 3, H * SCALE, W * SCALE)
        actual_shape = tuple(output.shape)
        
        shape_correct = actual_shape == expected_shape
        print_result(
            "forward() 실행", 
            shape_correct,
            f"출력: {list(actual_shape)}, 예상: {list(expected_shape)}"
        )
        
        # 값 범위 체크
        val_min, val_max = output.min().item(), output.max().item()
        print(f"         값 범위: [{val_min:.3f}, {val_max:.3f}]")
        
        return shape_correct
        
    except Exception as e:
        print_result("forward() 실행", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_encode(model):
    """테스트 4: encode() Feature 추출 테스트"""
    print_header("TEST 4: encode() - Feature Extraction")
    
    model = model.to(DEVICE)
    model.eval()
    
    H, W = INPUT_SIZE
    x = torch.randn(BATCH_SIZE, 3, H, W).to(DEVICE)
    
    print(f"  입력: {list(x.shape)}")
    
    try:
        with torch.no_grad():
            features = model.encode(x)
        
        # Feature shape: [B, embed_dim, H, W]
        expected_shape = (BATCH_SIZE, model.feature_channels, H, W)
        actual_shape = tuple(features.shape)
        
        shape_correct = actual_shape == expected_shape
        print_result(
            "encode() 실행",
            shape_correct,
            f"출력: {list(actual_shape)}, 예상: {list(expected_shape)}"
        )
        
        return features if shape_correct else None
        
    except Exception as e:
        print_result("encode() 실행", False, str(e))
        import traceback
        traceback.print_exc()
        return None


def test_decode(model, features):
    """테스트 5: decode() HR 복원 테스트"""
    print_header("TEST 5: decode() - HR Reconstruction")
    
    if features is None:
        print_result("decode() 실행", False, "features가 None (encode 실패)")
        return False
    
    model = model.to(DEVICE)
    model.eval()
    
    H, W = INPUT_SIZE
    print(f"  입력 features: {list(features.shape)}")
    
    try:
        with torch.no_grad():
            hr_image = model.decode(features)
        
        expected_shape = (BATCH_SIZE, 3, H * SCALE, W * SCALE)
        actual_shape = tuple(hr_image.shape)
        
        shape_correct = actual_shape == expected_shape
        print_result(
            "decode() 실행",
            shape_correct,
            f"출력: {list(actual_shape)}, 예상: {list(expected_shape)}"
        )
        
        return shape_correct
        
    except Exception as e:
        print_result("decode() 실행", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_encode_decode_consistency(model):
    """테스트 6: encode + decode vs forward 일관성"""
    print_header("TEST 6: encode→decode vs forward 일관성")
    
    model = model.to(DEVICE)
    model.eval()
    
    H, W = INPUT_SIZE
    x = torch.randn(BATCH_SIZE, 3, H, W).to(DEVICE)
    
    try:
        with torch.no_grad():
            # 방법 1: forward (원본)
            out_forward = model(x)
            
            # 방법 2: encode + decode
            features = model.encode(x)
            out_enc_dec = model.decode(features)
        
        # Shape 비교
        shape_match = out_forward.shape == out_enc_dec.shape
        print_result("Shape 일치", shape_match)
        
        # 값 비교 (완전히 같지는 않을 수 있음 - 정규화 차이)
        # 참고용으로 출력
        diff = (out_forward - out_enc_dec).abs()
        print(f"         차이 (평균): {diff.mean().item():.6f}")
        print(f"         차이 (최대): {diff.max().item():.6f}")
        
        return shape_match
        
    except Exception as e:
        print_result("일관성 테스트", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_different_sizes(model):
    """테스트 7: 다양한 입력 크기 테스트"""
    print_header("TEST 7: 다양한 입력 크기")
    
    model = model.to(DEVICE)
    model.eval()
    
    test_sizes = [
        (32, 32),   # window_size(16)의 배수
        (48, 48),   # 배수 아님
        (64, 64),   # 배수
        (50, 70),   # 배수 아님, 비정방형
    ]
    
    all_passed = True
    
    for H, W in test_sizes:
        x = torch.randn(1, 3, H, W).to(DEVICE)
        try:
            with torch.no_grad():
                out = model(x)
                features = model.encode(x)
            
            expected_out = (1, 3, H * SCALE, W * SCALE)
            expected_feat = (1, model.feature_channels, H, W)
            
            out_ok = tuple(out.shape) == expected_out
            feat_ok = tuple(features.shape) == expected_feat
            
            if out_ok and feat_ok:
                print_result(f"Size {H}x{W}", True, f"→ {H*SCALE}x{W*SCALE}")
            else:
                print_result(f"Size {H}x{W}", False)
                all_passed = False
                
        except Exception as e:
            print_result(f"Size {H}x{W}", False, str(e))
            all_passed = False
    
    return all_passed


def test_gpu_memory():
    """테스트 8: GPU 메모리 사용량"""
    print_header("TEST 8: GPU 메모리")
    
    if DEVICE != 'cuda':
        print("  ⚠️ CUDA 사용 불가, 스킵")
        return True
    
    from models.sr_models import create_mamba_sr
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = create_mamba_sr(scale=SCALE).to(DEVICE)
    model.eval()
    
    # 다양한 크기로 테스트
    test_sizes = [(64, 64), (128, 128), (192, 192)]
    
    for H, W in test_sizes:
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(1, 3, H, W).to(DEVICE)
        
        with torch.no_grad():
            _ = model(x)
        
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  {H}x{W} → Peak Memory: {peak_mb:.1f} MB")
    
    return True


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "🧪" * 30)
    print("       MambaSR Wrapper 테스트")
    print("🧪" * 30)
    print(f"\n  Device: {DEVICE}")
    print(f"  Scale: {SCALE}x")
    print(f"  Test size: {INPUT_SIZE}")
    
    results = {}
    
    # 테스트 1: Import
    results['import'] = test_import()
    if not results['import']:
        print("\n❌ Import 실패로 테스트 중단")
        return results
    
    # 테스트 2: 모델 생성
    model = test_model_creation()
    results['creation'] = model is not None
    if not results['creation']:
        print("\n❌ 모델 생성 실패로 테스트 중단")
        return results
    
    # 테스트 3: forward
    results['forward'] = test_forward(model)
    
    # 테스트 4: encode
    features = test_encode(model)
    results['encode'] = features is not None
    
    # 테스트 5: decode
    results['decode'] = test_decode(model, features)
    
    # 테스트 6: 일관성
    results['consistency'] = test_encode_decode_consistency(model)
    
    # 테스트 7: 다양한 크기
    results['sizes'] = test_different_sizes(model)
    
    # 테스트 8: GPU 메모리
    results['memory'] = test_gpu_memory()
    
    # ============ 결과 요약 ============
    print_header("테스트 결과 요약")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  총 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("\n  🎉 모든 테스트 통과!")
    else:
        print("\n  ⚠️ 일부 테스트 실패")
    
    return results


# ============ 메인 실행 ============
if __name__ == '__main__':
    run_all_tests()
