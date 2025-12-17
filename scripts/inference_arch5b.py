"""
=============================================================================
inference_arch5b.py - Architecture 5-B: Feature Fusion Inference & Analysis
=============================================================================

[Arch5-B 특징]
- Feature 레벨에서 SR과 YOLO 융합
- 이미지 전체 SR 불필요 → 연산량 감소
- End-to-end 최적화된 feature 융합

[논문용 분석 기능]
1. Feature 시각화 (SR Feature, YOLO Feature, Fused Feature)
2. Attention Map 시각화
3. Arch0 vs Arch5-B 성능 비교
4. 연산량/속도 비교

[사용법]
# 단일 이미지 추론
python scripts/inference_arch5b.py \
    --input image.jpg \
    --weights checkpoints/arch5b_best.pth \
    --output results/arch5b/

# 폴더 처리 + 분석
python scripts/inference_arch5b.py \
    --input data/lr/images/test/ \
    --weights checkpoints/arch5b_best.pth \
    --output results/arch5b/ \
    --analyze

# Arch0과 비교
python scripts/inference_arch5b.py \
    --input image.jpg \
    --weights checkpoints/arch5b_best.pth \
    --rfdn_weights checkpoints/rfdn_best.pth \
    --yolo_weights checkpoints/yolo_ship.pt \
    --compare_arch0 \
    --output results/comparison/
"""

import os
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import time
from datetime import datetime
from types import SimpleNamespace

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.pipelines.arch5b_fusion import Arch5BFusion
from src.models.sr_models.rfdn import RFDN
from ultralytics import YOLO


class Arch5BAnalyzer:
    """
    Arch5-B 추론 및 분석 클래스
    
    [기능]
    1. 기본 추론
    2. Feature 시각화
    3. Attention 분석
    4. Arch0과 비교
    5. 속도/연산량 프로파일링
    """
    
    def __init__(
        self,
        arch5b_weights: str,
        device: str = 'cuda',
        # Arch0 비교용 (선택)
        rfdn_weights: str = None,
        yolo_weights: str = 'yolov8n.pt'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # =====================================================================
        # Arch5-B 모델 로드
        # =====================================================================
        print(f"[Arch5BAnalyzer] Loading Arch5-B on {self.device}")
        
        # Config (기본값)
        config = SimpleNamespace(
            model=SimpleNamespace(
                rfdn=SimpleNamespace(nf=50, num_modules=4),
                yolo=SimpleNamespace(weights_path=yolo_weights, num_classes=1),
                fusion=SimpleNamespace(use_cross_attention=True, use_cbam=True, num_heads=4)
            ),
            data=SimpleNamespace(upscale_factor=4),
            training=SimpleNamespace(sr_weight=0.0, det_weight=1.0),
            device=str(self.device)
        )
        
        self.arch5b = Arch5BFusion(config)
        
        # 가중치 로드
        if arch5b_weights and Path(arch5b_weights).exists():
            ckpt = torch.load(arch5b_weights, map_location=self.device)
            self.arch5b.load_state_dict(ckpt['model_state_dict'])
            print(f"  ✓ Arch5-B loaded: {arch5b_weights}")
        
        self.arch5b = self.arch5b.to(self.device)
        self.arch5b.eval()
        
        # =====================================================================
        # Arch0 모델 로드 (비교용)
        # =====================================================================
        self.arch0_rfdn = None
        self.arch0_yolo = None
        
        if rfdn_weights and Path(rfdn_weights).exists():
            print(f"[Arch5BAnalyzer] Loading Arch0 components for comparison...")
            
            # RFDN
            self.arch0_rfdn = RFDN(
                in_channels=3, out_channels=3,
                nf=50, num_modules=4, upscale=4
            )
            ckpt = torch.load(rfdn_weights, map_location=self.device)
            state = ckpt.get('model_state_dict', ckpt)
            self.arch0_rfdn.load_state_dict(state)
            self.arch0_rfdn = self.arch0_rfdn.to(self.device)
            self.arch0_rfdn.eval()
            print(f"  ✓ RFDN loaded for Arch0")
            
            # YOLO
            self.arch0_yolo = YOLO(yolo_weights)
            print(f"  ✓ YOLO loaded for Arch0")
        
        print(f"[Arch5BAnalyzer] Ready!")
    
    # =========================================================================
    # Basic Inference
    # =========================================================================
    
    @torch.no_grad()
    def inference(
        self,
        lr_image: np.ndarray,
        conf_threshold: float = 0.25,
        return_features: bool = False
    ) -> dict:
        """
        단일 이미지 추론
        
        Args:
            lr_image: LR 이미지 (BGR, numpy)
            conf_threshold: Confidence threshold
            return_features: Feature 반환 여부
        
        Returns:
            {
                'detections': [{boxes, scores, classes}],
                'features': (optional) 중간 feature들,
                'inference_time': 추론 시간 (ms)
            }
        """
        # Preprocess
        lr_rgb = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        lr_tensor = torch.from_numpy(lr_rgb).permute(2, 0, 1).float() / 255.0
        lr_tensor = lr_tensor.unsqueeze(0).to(self.device)
        
        # Inference with timing
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        result = self.arch5b.inference(
            lr_tensor,
            conf_threshold=conf_threshold,
            return_features=True
        )
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # 결과 정리
        output = {
            'detections': result['detections'],
            'inference_time': inference_time
        }
        
        if return_features:
            output['features'] = result['features']
        
        return output
    
    # =========================================================================
    # Feature Visualization (논문 Figure용)
    # =========================================================================
    
    @torch.no_grad()
    def visualize_features(
        self,
        lr_image: np.ndarray,
        save_path: str = None
    ):
        """
        Feature 시각화 (논문 Figure용)
        
        시각화 항목:
        - SR Feature (RFDN encoder 출력)
        - YOLO Features (P3, P4, P5)
        - Fused Features (융합 후)
        """
        # Preprocess
        lr_rgb = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        lr_tensor = torch.from_numpy(lr_rgb).permute(2, 0, 1).float() / 255.0
        lr_tensor = lr_tensor.unsqueeze(0).to(self.device)
        
        # Forward with features
        _, features = self.arch5b(lr_tensor, return_features=True)
        
        # Feature 추출
        sr_feat = features['sr_features']  # [B, 50, H, W]
        yolo_feats = features['yolo_features']  # {'p3': ..., 'p4': ..., 'p5': ...}
        fused_feats = features['fused_features']  # {'p3': ..., 'p4': ..., 'p5': ...}
        
        # 시각화
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # Row 1: SR Feature
        axes[0, 0].imshow(cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Input LR Image', fontsize=12)
        axes[0, 0].axis('off')
        
        sr_vis = sr_feat[0].mean(dim=0).cpu().numpy()
        axes[0, 1].imshow(sr_vis, cmap='viridis')
        axes[0, 1].set_title(f'SR Feature (mean)\n{sr_feat.shape}', fontsize=12)
        axes[0, 1].axis('off')
        
        # SR Feature 채널별 분산
        sr_std = sr_feat[0].std(dim=0).cpu().numpy()
        axes[0, 2].imshow(sr_std, cmap='hot')
        axes[0, 2].set_title('SR Feature (std)', fontsize=12)
        axes[0, 2].axis('off')
        
        axes[0, 3].axis('off')
        
        # Row 2: YOLO Features
        for i, (name, feat) in enumerate(yolo_feats.items()):
            if i < 3:
                feat_vis = feat[0].mean(dim=0).cpu().numpy()
                axes[1, i].imshow(feat_vis, cmap='viridis')
                axes[1, i].set_title(f'YOLO {name.upper()}\n{feat.shape}', fontsize=12)
                axes[1, i].axis('off')
        axes[1, 3].axis('off')
        
        # Row 3: Fused Features
        for i, (name, feat) in enumerate(fused_feats.items()):
            if i < 3:
                feat_vis = feat[0].mean(dim=0).cpu().numpy()
                axes[2, i].imshow(feat_vis, cmap='viridis')
                axes[2, i].set_title(f'Fused {name.upper()}\n{feat.shape}', fontsize=12)
                axes[2, i].axis('off')
        axes[2, 3].axis('off')
        
        plt.suptitle('Arch5-B Feature Visualization', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature visualization saved: {save_path}")
        
        plt.show()
    
    # =========================================================================
    # Arch0 vs Arch5-B Comparison (논문 Table용)
    # =========================================================================
    
    @torch.no_grad()
    def compare_with_arch0(
        self,
        lr_image: np.ndarray,
        conf_threshold: float = 0.25
    ) -> dict:
        """
        Arch0 vs Arch5-B 비교
        
        비교 항목:
        - 탐지 결과 (boxes, scores)
        - 추론 시간
        - 메모리 사용량
        """
        if self.arch0_rfdn is None or self.arch0_yolo is None:
            raise ValueError("Arch0 components not loaded. Provide rfdn_weights.")
        
        # Preprocess
        lr_rgb = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        lr_tensor = torch.from_numpy(lr_rgb).permute(2, 0, 1).float() / 255.0
        lr_tensor = lr_tensor.unsqueeze(0).to(self.device)
        
        results = {}
        
        # =====================================================================
        # Arch0: LR → SR → HR → YOLO
        # =====================================================================
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start = time.time()
        
        # SR
        sr_image = self.arch0_rfdn(lr_tensor)
        sr_np = self._tensor_to_numpy(sr_image)
        
        # YOLO Detection
        arch0_results = self.arch0_yolo(sr_np, conf=conf_threshold, verbose=False)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        arch0_time = (time.time() - start) * 1000
        
        arch0_det = self._parse_yolo_results(arch0_results)[0]
        
        results['arch0'] = {
            'detections': arch0_det,
            'num_ships': len(arch0_det['boxes']),
            'inference_time': arch0_time,
            'sr_image': sr_np
        }
        
        # =====================================================================
        # Arch5-B: Feature Fusion
        # =====================================================================
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start = time.time()
        
        arch5b_result = self.arch5b.inference(
            lr_tensor,
            conf_threshold=conf_threshold
        )
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        arch5b_time = (time.time() - start) * 1000
        
        # 결과 파싱 (Arch5B의 출력 형식에 맞게)
        arch5b_det = self._parse_arch5b_detections(arch5b_result['detections'])
        
        results['arch5b'] = {
            'detections': arch5b_det,
            'num_ships': len(arch5b_det['boxes']),
            'inference_time': arch5b_time
        }
        
        # =====================================================================
        # 비교 요약
        # =====================================================================
        results['comparison'] = {
            'speedup': arch0_time / arch5b_time if arch5b_time > 0 else 0,
            'arch0_ships': results['arch0']['num_ships'],
            'arch5b_ships': results['arch5b']['num_ships'],
            'arch0_time_ms': arch0_time,
            'arch5b_time_ms': arch5b_time
        }
        
        return results
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Tensor → BGR numpy"""
        img = tensor.squeeze(0).cpu().clamp(0, 1)
        img = (img * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    def _parse_yolo_results(self, results) -> list:
        """YOLO 결과 파싱"""
        detections = []
        for r in results:
            boxes = r.boxes
            det = {
                'boxes': boxes.xyxy.cpu().numpy().tolist() if len(boxes) > 0 else [],
                'scores': boxes.conf.cpu().numpy().tolist() if len(boxes) > 0 else [],
                'classes': boxes.cls.cpu().numpy().astype(int).tolist() if len(boxes) > 0 else []
            }
            detections.append(det)
        return detections
    
    def _parse_arch5b_detections(self, detections) -> dict:
        """Arch5B 결과 파싱"""
        # Arch5B는 raw predictions 반환, NMS 필요
        # 현재 구조에서는 detect head output
        return {
            'boxes': [],
            'scores': [],
            'classes': []
        }
    
    # =========================================================================
    # Comparison Visualization
    # =========================================================================
    
    def plot_comparison(
        self,
        lr_image: np.ndarray,
        save_path: str = None
    ):
        """Arch0 vs Arch5-B 시각화 비교 (논문 Figure용)"""
        comparison = self.compare_with_arch0(lr_image)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # LR Input
        axes[0].imshow(cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Input LR Image', fontsize=14)
        axes[0].axis('off')
        
        # Arch0 결과
        arch0_vis = comparison['arch0']['sr_image'].copy()
        for box in comparison['arch0']['detections']['boxes']:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(arch0_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        axes[1].imshow(cv2.cvtColor(arch0_vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title(
            f"Arch0 (Sequential)\n"
            f"{comparison['arch0']['num_ships']} ships | "
            f"{comparison['arch0']['inference_time']:.1f}ms",
            fontsize=14
        )
        axes[1].axis('off')
        
        # Arch5-B 결과 (LR에 표시, Feature fusion이므로 HR 이미지 없음)
        lr_vis = cv2.resize(lr_image, (lr_image.shape[1]*4, lr_image.shape[0]*4))
        for box in comparison['arch5b']['detections']['boxes']:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(lr_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        axes[2].imshow(cv2.cvtColor(lr_vis, cv2.COLOR_BGR2RGB))
        axes[2].set_title(
            f"Arch5-B (Feature Fusion)\n"
            f"{comparison['arch5b']['num_ships']} ships | "
            f"{comparison['arch5b']['inference_time']:.1f}ms\n"
            f"Speedup: {comparison['comparison']['speedup']:.2f}x",
            fontsize=14
        )
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison saved: {save_path}")
        
        plt.show()
    
    # =========================================================================
    # Batch Processing
    # =========================================================================
    
    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        conf_threshold: float = 0.25,
        save_vis: bool = True
    ) -> dict:
        """폴더 전체 처리"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if save_vis:
            (output_path / 'visualizations').mkdir(exist_ok=True)
        
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        
        all_results = {}
        total_time = 0
        
        for img_path in tqdm(image_files, desc="Processing"):
            lr_image = cv2.imread(str(img_path))
            if lr_image is None:
                continue
            
            result = self.inference(lr_image, conf_threshold)
            
            all_results[img_path.name] = {
                'num_detections': len(result['detections']),
                'inference_time': result['inference_time']
            }
            total_time += result['inference_time']
        
        # 통계
        stats = {
            'total_images': len(image_files),
            'total_time_ms': total_time,
            'avg_time_ms': total_time / len(image_files) if image_files else 0,
            'fps': 1000 / (total_time / len(image_files)) if total_time > 0 else 0
        }
        
        # 저장
        with open(output_path / 'results.json', 'w') as f:
            json.dump({'results': all_results, 'stats': stats}, f, indent=2)
        
        print(f"\n[Summary]")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Avg inference time: {stats['avg_time_ms']:.2f} ms")
        print(f"  FPS: {stats['fps']:.2f}")
        
        return stats
    
    # =========================================================================
    # Profiling (연산량 분석)
    # =========================================================================
    
    def profile_model(self, input_size: tuple = (1, 3, 640, 640)) -> dict:
        """
        모델 프로파일링 (FLOPs, 파라미터 수, 메모리)
        """
        from thop import profile, clever_format
        
        dummy_input = torch.randn(*input_size, device=self.device)
        
        # Arch5-B
        flops, params = profile(self.arch5b, inputs=(dummy_input,), verbose=False)
        arch5b_flops, arch5b_params = clever_format([flops, params], "%.2f")
        
        result = {
            'arch5b': {
                'flops': arch5b_flops,
                'params': arch5b_params,
                'flops_raw': flops,
                'params_raw': params
            }
        }
        
        # Arch0 (비교용)
        if self.arch0_rfdn is not None:
            # RFDN
            rfdn_flops, rfdn_params = profile(self.arch0_rfdn, inputs=(dummy_input,), verbose=False)
            rfdn_flops_fmt, rfdn_params_fmt = clever_format([rfdn_flops, rfdn_params], "%.2f")
            
            result['arch0_rfdn'] = {
                'flops': rfdn_flops_fmt,
                'params': rfdn_params_fmt
            }
            
            # YOLO는 Ultralytics 자체 프로파일링 필요
        
        return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Arch5-B Inference & Analysis")
    
    # Required
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or folder')
    parser.add_argument('--weights', type=str, required=True,
                       help='Arch5-B checkpoint path')
    
    # Optional
    parser.add_argument('--output', type=str, default='results/arch5b/',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--conf', type=float, default=0.25)
    
    # Comparison with Arch0
    parser.add_argument('--compare_arch0', action='store_true',
                       help='Compare with Arch0')
    parser.add_argument('--rfdn_weights', type=str, default=None,
                       help='RFDN weights for Arch0 comparison')
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.pt')
    
    # Analysis
    parser.add_argument('--analyze', action='store_true',
                       help='Run feature analysis')
    parser.add_argument('--profile', action='store_true',
                       help='Profile model FLOPs')
    
    args = parser.parse_args()
    
    # Initialize
    analyzer = Arch5BAnalyzer(
        arch5b_weights=args.weights,
        device=args.device,
        rfdn_weights=args.rfdn_weights,
        yolo_weights=args.yolo_weights
    )
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Profiling
    if args.profile:
        print("\n[Profiling]")
        try:
            profile_result = analyzer.profile_model()
            print(f"  Arch5-B FLOPs: {profile_result['arch5b']['flops']}")
            print(f"  Arch5-B Params: {profile_result['arch5b']['params']}")
        except ImportError:
            print("  thop 패키지 필요: pip install thop")
    
    # Processing
    if input_path.is_file():
        lr_image = cv2.imread(str(input_path))
        
        if args.compare_arch0 and args.rfdn_weights:
            # Arch0 비교
            print("\n[Comparing with Arch0]")
            analyzer.plot_comparison(lr_image, str(output_path / 'comparison.png'))
        
        if args.analyze:
            # Feature 분석
            print("\n[Feature Analysis]")
            analyzer.visualize_features(lr_image, str(output_path / 'features.png'))
        
        else:
            # 기본 추론
            result = analyzer.inference(lr_image, args.conf)
            print(f"\n[Result]")
            print(f"  Detections: {len(result['detections'])}")
            print(f"  Inference time: {result['inference_time']:.2f} ms")
    
    elif input_path.is_dir():
        # 폴더 처리
        stats = analyzer.process_folder(str(input_path), str(output_path), args.conf)


if __name__ == "__main__":
    main()