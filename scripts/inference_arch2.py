"""
=============================================================================
inference_arch2.py - Arch2 추론 + Gate 동작 분석
=============================================================================

[논문용 분석 기능]
1. Gate 분포 히스토그램
2. SR 적용 비율 통계
3. Gate 값별 Detection 성능 비교
4. 이미지별 Gate 시각화
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from types import SimpleNamespace

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.pipelines.arch2_softgate import Arch2SoftGate


class Arch2Analyzer:
    """Arch2 추론 및 분석 클래스"""
    
    def __init__(
        self,
        rfdn_weights: str,
        yolo_weights: str,
        gate_weights: str = None,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Config
        config = SimpleNamespace(
            model=SimpleNamespace(
                rfdn=SimpleNamespace(nf=50, num_modules=4),
                yolo=SimpleNamespace(weights_path=yolo_weights, num_classes=1),
                gate=SimpleNamespace(base_channels=32, num_layers=4)
            ),
            data=SimpleNamespace(upscale_factor=4),
            training=SimpleNamespace(sr_weight=0.0, det_weight=1.0),
            device=str(self.device)
        )
        
        # Model
        self.model = Arch2SoftGate(config)
        self.model = self.model.to(self.device)
        
        # Load weights
        if rfdn_weights:
            ckpt = torch.load(rfdn_weights, map_location=self.device)
            state = ckpt.get('model_state_dict', ckpt)
            self.model.sr_model.load_state_dict(state)
            print(f"✓ RFDN loaded: {rfdn_weights}")
        
        if gate_weights:
            ckpt = torch.load(gate_weights, map_location=self.device)
            state = ckpt.get('gate_state_dict', ckpt)
            self.model.gate_network.load_state_dict(state)
            print(f"✓ Gate loaded: {gate_weights}")
        
        self.model.eval()
    
    # =========================================================================
    # Basic Inference
    # =========================================================================
    
    @torch.no_grad()
    def inference(self, lr_image: np.ndarray) -> dict:
        """단일 이미지 추론"""
        # Preprocess
        lr_img = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).float() / 255.0
        lr_tensor = lr_tensor.unsqueeze(0).to(self.device)
        
        # Forward
        outputs = self.model(lr_tensor, return_intermediates=True)
        
        return {
            'gate': outputs['gate'].item(),
            'hr_image': self._to_numpy(outputs['hr_image']),
            'sr_image': self._to_numpy(outputs['sr_image']),
            'upsampled': self._to_numpy(outputs['upsampled']),
            'detections': outputs['detections']
        }
    
    def _to_numpy(self, tensor):
        """Tensor → numpy BGR"""
        img = tensor.squeeze(0).cpu().clamp(0, 1)
        img = (img * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # =========================================================================
    # Gate Analysis (논문용)
    # =========================================================================
    
    @torch.no_grad()
    def analyze_gate_distribution(self, image_dir: str, num_images: int = 500):
        """
        Gate 분포 분석 (논문 Figure용)
        
        Returns:
            - Gate 히스토그램
            - SR 적용 비율
            - 통계 (mean, std, min, max)
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg"))[:num_images]
        
        gate_values = []
        
        for img_path in tqdm(image_files, desc="Analyzing gates"):
            lr_img = cv2.imread(str(img_path))
            if lr_img is None:
                continue
            
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
            lr_img = cv2.resize(lr_img, (160, 160))
            
            lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).float() / 255.0
            lr_tensor = lr_tensor.unsqueeze(0).to(self.device)
            
            gate = self.model.gate_network(lr_tensor)
            gate_values.append(gate.item())
        
        gate_values = np.array(gate_values)
        
        return {
            'values': gate_values,
            'mean': gate_values.mean(),
            'std': gate_values.std(),
            'min': gate_values.min(),
            'max': gate_values.max(),
            'sr_ratio': (gate_values > 0.5).mean(),  # SR 적용 비율
            'bypass_ratio': (gate_values < 0.5).mean()  # SR 스킵 비율
        }
    
    def plot_gate_histogram(self, gate_stats: dict, save_path: str = None):
        """Gate 히스토그램 (논문 Figure)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(gate_stats['values'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0.5, color='r', linestyle='--', label='Threshold (0.5)')
        ax.axvline(x=gate_stats['mean'], color='g', linestyle='-', label=f'Mean ({gate_stats["mean"]:.3f})')
        
        ax.set_xlabel('Gate Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Gate Network Output Distribution', fontsize=14)
        ax.legend()
        
        # 통계 텍스트
        stats_text = f"SR Apply: {gate_stats['sr_ratio']:.1%}\nSR Skip: {gate_stats['bypass_ratio']:.1%}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.show()
    
    # =========================================================================
    # Comparison Analysis (논문용)
    # =========================================================================
    
    @torch.no_grad()
    def compare_sr_vs_bypass(self, lr_image: np.ndarray):
        """
        SR 적용 vs 미적용 비교 (논문 Figure용)
        
        같은 이미지에 대해:
        - gate=1.0 (SR 강제 적용)
        - gate=0.0 (SR 강제 스킵)
        - gate=자동 (학습된 값)
        """
        lr_img = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        lr_img = cv2.resize(lr_img, (160, 160))
        lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).float() / 255.0
        lr_tensor = lr_tensor.unsqueeze(0).to(self.device)
        
        # SR 강제 적용
        result_sr = self.model.forward_with_gate_control(lr_tensor, force_gate=1.0)
        
        # SR 강제 스킵
        result_bypass = self.model.forward_with_gate_control(lr_tensor, force_gate=0.0)
        
        # 자동 (학습된 Gate)
        result_auto = self.model.forward_with_gate_control(lr_tensor, force_gate=None)
        
        return {
            'with_sr': {
                'hr_image': self._to_numpy(result_sr['hr_image']),
                'detections': result_sr['detections'],
                'gate': 1.0
            },
            'without_sr': {
                'hr_image': self._to_numpy(result_bypass['hr_image']),
                'detections': result_bypass['detections'],
                'gate': 0.0
            },
            'auto': {
                'hr_image': self._to_numpy(result_auto['hr_image']),
                'detections': result_auto['detections'],
                'gate': result_auto['gate'].item()
            }
        }
    
    def visualize_comparison(self, lr_image: np.ndarray, save_path: str = None):
        """비교 시각화 (논문 Figure)"""
        comparison = self.compare_sr_vs_bypass(lr_image)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # LR (업스케일)
        lr_up = cv2.resize(lr_image, (640, 640))
        axes[0].imshow(cv2.cvtColor(lr_up, cv2.COLOR_BGR2RGB))
        axes[0].set_title('LR (Upscaled)', fontsize=12)
        axes[0].axis('off')
        
        # SR 강제 적용
        axes[1].imshow(cv2.cvtColor(comparison['with_sr']['hr_image'], cv2.COLOR_BGR2RGB))
        n_det = len(comparison['with_sr']['detections'][0].get('boxes', []))
        axes[1].set_title(f'With SR (gate=1.0)\n{n_det} detections', fontsize=12)
        axes[1].axis('off')
        
        # SR 스킵
        axes[2].imshow(cv2.cvtColor(comparison['without_sr']['hr_image'], cv2.COLOR_BGR2RGB))
        n_det = len(comparison['without_sr']['detections'][0].get('boxes', []))
        axes[2].set_title(f'Without SR (gate=0.0)\n{n_det} detections', fontsize=12)
        axes[2].axis('off')
        
        # 자동
        axes[3].imshow(cv2.cvtColor(comparison['auto']['hr_image'], cv2.COLOR_BGR2RGB))
        n_det = len(comparison['auto']['detections'][0].get('boxes', []))
        gate_val = comparison['auto']['gate']
        axes[3].set_title(f'Auto (gate={gate_val:.3f})\n{n_det} detections', fontsize=12)
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.show()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--rfdn_weights', type=str, required=True)
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.pt')
    parser.add_argument('--gate_weights', type=str, default=None)
    parser.add_argument('--output', type=str, default='results/')
    parser.add_argument('--analyze', action='store_true', help='Run gate analysis')
    
    args = parser.parse_args()
    
    analyzer = Arch2Analyzer(
        rfdn_weights=args.rfdn_weights,
        yolo_weights=args.yolo_weights,
        gate_weights=args.gate_weights
    )
    
    if args.analyze:
        # Gate 분포 분석
        stats = analyzer.analyze_gate_distribution(args.input)
        print(f"\n[Gate Statistics]")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std: {stats['std']:.3f}")
        print(f"  SR ratio: {stats['sr_ratio']:.1%}")
        
        Path(args.output).mkdir(parents=True, exist_ok=True)
        analyzer.plot_gate_histogram(stats, f"{args.output}/gate_histogram.png")
    else:
        # 단일 이미지 추론
        lr_image = cv2.imread(args.input)
        result = analyzer.inference(lr_image)
        print(f"Gate: {result['gate']:.3f}")
