"""
=============================================================================
test_all_architectures.py - 전체 Architecture 통합 테스트
=============================================================================

[수정 내역]
- Gradient 테스트 로직 개선 (trainable 파라미터만 체크)
- 더 나은 에러 처리
- 테스트 결과 상세 출력
"""

import torch
import torch.nn as nn
import argparse
import time
import traceback
from types import SimpleNamespace
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


# =============================================================================
# 테스트 결과 클래스
# =============================================================================

@dataclass
class TestResult:
    """테스트 결과 저장"""
    name: str
    arch: str
    sr_type: str
    creation: bool = False
    forward: bool = False
    loss: bool = False
    gradient: bool = False
    inference_time_ms: float = 0.0
    error_msg: str = ""
    
    @property
    def all_passed(self) -> bool:
        return self.creation and self.forward and self.loss and self.gradient
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.all_passed else "❌ FAIL"
        return f"{status} | {self.name} | Create:{self.creation} Forward:{self.forward} Loss:{self.loss} Grad:{self.gradient}"


# =============================================================================
# Config 생성 함수들
# =============================================================================

def get_base_config(sr_type: str, device: str) -> SimpleNamespace:
    """기본 config 생성"""
    
    if sr_type == 'rfdn':
        sr_config = SimpleNamespace(
            nf=50,
            num_modules=4,
            pretrain_path=None
        )
        model_config = SimpleNamespace(
            sr_type='rfdn',
            rfdn=sr_config,
            yolo=SimpleNamespace(weights_path="yolov8n.pt", num_classes=80)
        )
    else:  # mamba
        sr_config = SimpleNamespace(
            img_size=64,
            embed_dim=48,
            d_state=8,
            depths=[5, 5, 5, 5],
            num_heads=[4, 4, 4, 4],
            window_size=16,
            pretrain_path=None
        )
        model_config = SimpleNamespace(
            sr_type='mamba',
            mamba=sr_config,
            yolo=SimpleNamespace(weights_path="yolov8n.pt", num_classes=80)
        )
    
    return SimpleNamespace(
        model=model_config,
        data=SimpleNamespace(upscale_factor=4),
        training=SimpleNamespace(sr_weight=0.3, det_weight=0.7),
        device=device
    )


def get_arch2_config(sr_type: str, device: str) -> SimpleNamespace:
    """Arch2용 config (gate 추가)"""
    config = get_base_config(sr_type, device)
    config.model.gate = SimpleNamespace(base_channels=32, num_layers=4)
    return config


def get_arch4_config(sr_type: str, device: str) -> SimpleNamespace:
    """Arch4용 config (adaptive 추가)"""
    config = get_base_config(sr_type, device)
    config.model.adaptive = SimpleNamespace(
        low_conf_threshold=0.1,
        high_conf_threshold=0.5,
        merge_iou_threshold=0.5
    )
    config.data.final_conf_threshold = 0.25
    return config


def get_arch5b_config(sr_type: str, device: str) -> SimpleNamespace:
    """Arch5B용 config (fusion 추가)"""
    config = get_base_config(sr_type, device)
    config.model.fusion = SimpleNamespace(
        use_cross_attention=True,
        use_cbam=True,
        num_heads=4
    )
    return config


# =============================================================================
# Gradient 체크 헬퍼
# =============================================================================

def check_gradient_flow(model: nn.Module, component_name: str = None) -> Tuple[bool, str]:
    """
    Gradient flow 체크
    
    Returns:
        (has_gradient, details_string)
    """
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    
    if not trainable_params:
        return False, "No trainable parameters"
    
    params_with_grad = []
    for name, param in trainable_params:
        if param.grad is not None and param.grad.abs().sum() > 0:
            params_with_grad.append(name)
    
    has_grad = len(params_with_grad) > 0
    
    if has_grad:
        # 어느 컴포넌트에 gradient가 있는지 확인
        components = set()
        for name in params_with_grad:
            if 'sr_model' in name:
                components.add('sr')
            elif 'gate' in name:
                components.add('gate')
            elif 'fusion' in name:
                components.add('fusion')
            elif 'detector' in name:
                components.add('yolo')
        details = f"grad in: {', '.join(components)}"
    else:
        details = "no gradient"
    
    return has_grad, details


# =============================================================================
# 개별 아키텍처 테스트 함수들
# =============================================================================

def test_arch0(sr_type: str, device: str) -> TestResult:
    """Arch0 Sequential 테스트"""
    result = TestResult(
        name=f"Arch0_{sr_type.upper()}",
        arch="arch0",
        sr_type=sr_type
    )
    
    try:
        from src.models.pipelines.arch0_sequential import Arch0Sequential
        
        config = get_base_config(sr_type, device)
        
        # 1. Creation
        print(f"    [1/4] Creating model...", end=" ")
        model = Arch0Sequential(config)
        model.to(device)
        result.creation = True
        print("✓")
        
        # 2. Forward
        print(f"    [2/4] Testing forward...", end=" ")
        lr_image = torch.randn(2, 3, 160, 160, device=device)
        model.eval()
        
        start_time = time.time()
        with torch.no_grad():
            sr_image, detections = model(lr_image)
        result.inference_time_ms = (time.time() - start_time) * 1000
        
        assert sr_image.shape == (2, 3, 640, 640), f"Expected (2,3,640,640), got {sr_image.shape}"
        result.forward = True
        print(f"✓ ({result.inference_time_ms:.1f}ms)")
        
        # 3. Loss
        print(f"    [3/4] Testing loss...", end=" ")
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.2, 0.2],
            [1, 0, 0.3, 0.7, 0.15, 0.25],
        ], device=device)
        
        model.train()
        model.zero_grad()
        outputs = model(lr_image)
        loss_dict = model.compute_loss(outputs, targets)
        
        assert 'total' in loss_dict
        assert loss_dict['total'].requires_grad or loss_dict['total'].item() >= 0
        result.loss = True
        print(f"✓ (total={loss_dict['total'].item():.4f})")
        
        # 4. Gradient
        print(f"    [4/4] Testing gradient...", end=" ")
        if loss_dict['total'].requires_grad:
            loss_dict['total'].backward()
        
        has_grad, grad_details = check_gradient_flow(model)
        result.gradient = has_grad
        print(f"✓ ({grad_details})" if has_grad else f"✗ ({grad_details})")
        
    except Exception as e:
        result.error_msg = str(e)
        print(f"✗ Error: {e}")
        traceback.print_exc()
    
    return result


def test_arch2(sr_type: str, device: str) -> TestResult:
    """Arch2 SoftGate 테스트"""
    result = TestResult(
        name=f"Arch2_{sr_type.upper()}",
        arch="arch2",
        sr_type=sr_type
    )
    
    try:
        from src.models.pipelines.arch2_softgate import Arch2SoftGate
        
        config = get_arch2_config(sr_type, device)
        
        # 1. Creation
        print(f"    [1/4] Creating model...", end=" ")
        model = Arch2SoftGate(config)
        model.to(device)
        result.creation = True
        print("✓")
        
        # 2. Forward
        print(f"    [2/4] Testing forward...", end=" ")
        lr_image = torch.randn(2, 3, 160, 160, device=device)
        model.eval()
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(lr_image, return_intermediates=True)
        result.inference_time_ms = (time.time() - start_time) * 1000
        
        assert outputs['hr_image'].shape == (2, 3, 640, 640)
        assert outputs['gate'].shape == (2, 1)
        result.forward = True
        gate_vals = outputs['gate'].squeeze().tolist()
        print(f"✓ ({result.inference_time_ms:.1f}ms, gate={[f'{g:.3f}' for g in gate_vals]})")
        
        # 3. Loss
        print(f"    [3/4] Testing loss...", end=" ")
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.2, 0.2],
            [1, 0, 0.3, 0.7, 0.15, 0.25],
        ], device=device)
        
        model.train()
        model.zero_grad()
        outputs = model(lr_image, return_intermediates=True)
        loss_dict = model.compute_loss(outputs, targets)
        
        assert 'total' in loss_dict
        result.loss = True
        print(f"✓ (total={loss_dict['total'].item():.4f})")
        
        # 4. Gradient
        print(f"    [4/4] Testing gradient...", end=" ")
        if loss_dict['total'].requires_grad:
            loss_dict['total'].backward()
        
        has_grad, grad_details = check_gradient_flow(model)
        result.gradient = has_grad
        print(f"✓ ({grad_details})" if has_grad else f"✗ ({grad_details})")
        
    except Exception as e:
        result.error_msg = str(e)
        print(f"✗ Error: {e}")
        traceback.print_exc()
    
    return result


def test_arch4(sr_type: str, device: str) -> TestResult:
    """Arch4 Adaptive 테스트"""
    result = TestResult(
        name=f"Arch4_{sr_type.upper()}",
        arch="arch4",
        sr_type=sr_type
    )
    
    try:
        from src.models.pipelines.arch4_adaptive import Arch4Adaptive
        
        config = get_arch4_config(sr_type, device)
        
        # 1. Creation
        print(f"    [1/4] Creating model...", end=" ")
        model = Arch4Adaptive(config)
        # model.to(device)  # 이미 __init__에서 처리됨
        result.creation = True
        print("✓")
        
        # 2. Forward (추론)
        print(f"    [2/4] Testing forward (inference)...", end=" ")
        lr_image = torch.randn(2, 3, 160, 160, device=device)
        
        start_time = time.time()
        outputs = model.forward(lr_image, return_intermediate=True)
        result.inference_time_ms = (time.time() - start_time) * 1000
        
        assert 'detections' in outputs
        assert 'pass2_triggered' in outputs
        result.forward = True
        print(f"✓ ({result.inference_time_ms:.1f}ms, pass2={outputs['pass2_triggered']})")
        
        # 3. Loss (학습)
        print(f"    [3/4] Testing loss (train)...", end=" ")
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.2, 0.2],
            [1, 0, 0.3, 0.7, 0.15, 0.25],
        ], device=device)
        
        model.zero_grad()
        train_outputs = model.forward_train(lr_image)
        loss_dict = model.compute_loss(train_outputs, targets)
        
        assert 'total' in loss_dict
        result.loss = True
        print(f"✓ (total={loss_dict['total'].item():.4f})")
        
        # 4. Gradient
        print(f"    [4/4] Testing gradient...", end=" ")
        if loss_dict['total'].requires_grad:
            loss_dict['total'].backward()
        
        has_grad, grad_details = check_gradient_flow(model)
        result.gradient = has_grad
        print(f"✓ ({grad_details})" if has_grad else f"✗ ({grad_details})")
        
    except Exception as e:
        result.error_msg = str(e)
        print(f"✗ Error: {e}")
        traceback.print_exc()
    
    return result


def test_arch5b(sr_type: str, device: str) -> TestResult:
    """Arch5B Fusion 테스트"""
    result = TestResult(
        name=f"Arch5B_{sr_type.upper()}",
        arch="arch5b",
        sr_type=sr_type
    )
    
    try:
        from src.models.pipelines.arch5b_fusion import Arch5BFusion
        
        config = get_arch5b_config(sr_type, device)
        
        # 1. Creation
        print(f"    [1/4] Creating model...", end=" ")
        model = Arch5BFusion(config)
        model.to(device)
        result.creation = True
        print("✓")
        
        # 2. Forward
        print(f"    [2/4] Testing forward...", end=" ")
        lr_image = torch.randn(2, 3, 160, 160, device=device)
        model.eval()
        
        start_time = time.time()
        with torch.no_grad():
            detections, features = model(lr_image, return_features=True)
        result.inference_time_ms = (time.time() - start_time) * 1000
        
        assert features is not None
        assert 'sr_features' in features
        assert 'fused_features' in features
        result.forward = True
        print(f"✓ ({result.inference_time_ms:.1f}ms)")
        
        # 3. Loss
        print(f"    [3/4] Testing loss...", end=" ")
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.2, 0.2],
            [1, 0, 0.3, 0.7, 0.15, 0.25],
        ], device=device)
        
        model.train()
        model.zero_grad()
        outputs = model(lr_image, return_features=True)
        loss_dict = model.compute_loss(outputs, targets, lr_image=lr_image)
        
        assert 'total' in loss_dict
        result.loss = True
        print(f"✓ (total={loss_dict['total'].item():.4f})")
        
        # 4. Gradient
        print(f"    [4/4] Testing gradient...", end=" ")
        if loss_dict['total'].requires_grad:
            loss_dict['total'].backward()
        
        has_grad, grad_details = check_gradient_flow(model)
        result.gradient = has_grad
        print(f"✓ ({grad_details})" if has_grad else f"✗ ({grad_details})")
        
    except Exception as e:
        result.error_msg = str(e)
        print(f"✗ Error: {e}")
        traceback.print_exc()
    
    return result


# =============================================================================
# 메인 테스트 함수
# =============================================================================

def run_all_tests(
    target_arch: str = None,
    target_sr: str = None,
    device: str = 'cuda'
) -> List[TestResult]:
    """전체 테스트 실행"""
    print("=" * 70)
    print("🧪 Architecture Integration Test")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    
    # 테스트 조합 정의
    test_matrix = [
        ('arch0', 'rfdn', test_arch0),
        ('arch0', 'mamba', test_arch0),
        ('arch2', 'rfdn', test_arch2),
        ('arch2', 'mamba', test_arch2),
        ('arch4', 'rfdn', test_arch4),
        ('arch4', 'mamba', test_arch4),
        ('arch5b', 'rfdn', test_arch5b),
        ('arch5b', 'mamba', test_arch5b),
    ]
    
    # 필터링
    if target_arch:
        test_matrix = [(a, s, f) for a, s, f in test_matrix if a == target_arch]
    if target_sr:
        test_matrix = [(a, s, f) for a, s, f in test_matrix if s == target_sr]
    
    results = []
    
    for arch, sr_type, test_func in test_matrix:
        print(f"\n{'='*50}")
        print(f"  Testing: {arch.upper()} + {sr_type.upper()}")
        print(f"{'='*50}")
        
        # GPU 메모리 정리
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        result = test_func(sr_type, device)
        results.append(result)
        
        print(f"  Result: {result}")
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    
    print(f"\n{'Architecture':<20} {'Creation':^10} {'Forward':^10} {'Loss':^10} {'Gradient':^10} {'Time(ms)':^12} {'Status':^8}")
    print("-" * 80)
    
    passed = 0
    total = len(results)
    
    for r in results:
        status = "✅" if r.all_passed else "❌"
        print(f"{r.name:<20} {'✓' if r.creation else '✗':^10} {'✓' if r.forward else '✗':^10} {'✓' if r.loss else '✗':^10} {'✓' if r.gradient else '✗':^10} {r.inference_time_ms:^12.1f} {status:^8}")
        if r.all_passed:
            passed += 1
    
    print("-" * 80)
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed!")
        print("\nFailed tests:")
        for r in results:
            if not r.all_passed:
                print(f"  - {r.name}: {r.error_msg if r.error_msg else 'Check details above'}")
    
    return results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Architecture Integration Test")
    parser.add_argument('--arch', type=str, default=None,
                       choices=['arch0', 'arch2', 'arch4', 'arch5b'],
                       help='Test specific architecture')
    parser.add_argument('--sr', type=str, default=None,
                       choices=['rfdn', 'mamba'],
                       help='Test specific SR model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU")
        device = 'cpu'
    
    results = run_all_tests(
        target_arch=args.arch,
        target_sr=args.sr,
        device=device
    )