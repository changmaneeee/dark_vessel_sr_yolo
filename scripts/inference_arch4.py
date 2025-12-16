"""
=============================================================================
inference_arch4.py - Arch4 Adaptive 2-Pass 추론 + 분석
=============================================================================

[Arch4 핵심]
1차 탐지 (LR 업샘플) → 낮은 conf 있으면 → 2차 탐지 (SR) → 결과 병합

[논문용 분석 기능]
1. Pass2 트리거 비율 통계
2. Single-pass vs Dual-pass 성능 비교
3. 1차/2차 탐지 결과 시각화
4. 연산량 절약 분석

[사용법]

# 1️⃣ 단일 이미지 추론
python scripts/inference_arch4.py \
    --input test_image.jpg \
    --rfdn_weights checkpoints/rfdn_best.pth \
    --yolo_weights checkpoints/yolo_ship.pt \
    --output results/arch4/

# 2️⃣ 폴더 전체 처리
python scripts/inference_arch4.py \
    --input data/lr/images/test/ \
    --rfdn_weights checkpoints/rfdn_best.pth \
    --yolo_weights checkpoints/yolo_ship.pt \
    --output results/arch4/

# 3️⃣ Pass2 통계 분석 (논문용)
python scripts/inference_arch4.py \
    --input data/lr/images/test/ \
    --rfdn_weights checkpoints/rfdn_best.pth \
    --yolo_weights checkpoints/yolo_ship.pt \
    --output results/arch4/ \
    --analyze

# 4️⃣ Single vs Dual 비교 (논문 Figure용)
python scripts/inference_arch4.py \
    --input test_image.jpg \
    --rfdn_weights checkpoints/rfdn_best.pth \
    --yolo_weights checkpoints/yolo_ship.pt \
    --output results/arch4/ \
    --compare
```

---

## 📊 출력 예시
```
[Pass2 Statistics]
  total_images: 500
  pass2_triggered: 127
  pass2_ratio: 0.254  (25.4%)
  avg_pass1_detections: 2.3
  avg_final_detections: 2.8
  computational_savings: 74.6% SR skipped  ← 논문 핵심!
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
from datetime import datetime
from torchvision.ops import batched_nms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sr_models.rfdn import RFDN
from ultralytics import YOLO


# =============================================================================
# Arch4 Inference Pipeline
# =============================================================================

class Arch4Inference:
    """
    Arch4 Adaptive 2-Pass 추론 파이프라인
    
    [동작 원리]
    1. LR → Bilinear Upsample → YOLO (1차 탐지)
    2. 결과 분석: 낮은 confidence 객체 있으면?
       - Yes → LR → RFDN(SR) → YOLO (2차 탐지) → 결과 병합
       - No → 1차 결과 그대로 반환 (SR 스킵, 연산 절약!)
    """
    
    def __init__(
        self,
        rfdn_weights: str,
        yolo_weights: str,
        device: str = 'cuda',
        # Threshold 설정
        low_conf_threshold: float = 0.1,   # 이 이상이면 "뭔가 있을 수도"
        high_conf_threshold: float = 0.5,  # 이 이상이면 "확실히 있음"
        final_conf_threshold: float = 0.25,  # 최종 출력 threshold
        merge_iou_threshold: float = 0.5,  # 결과 병합 NMS
        upscale_factor: int = 4
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Thresholds
        self.low_conf = low_conf_threshold
        self.high_conf = high_conf_threshold
        self.final_conf = final_conf_threshold
        self.merge_iou = merge_iou_threshold
        self.upscale = upscale_factor
        
        print(f"[Arch4Inference] Initializing on {self.device}")
        
        # =====================================================================
        # SR Model (RFDN)
        # =====================================================================
        print("[1/2] Loading RFDN...")
        self.sr_model = RFDN(
            in_channels=3,
            out_channels=3,
            nf=50,
            num_modules=4,
            upscale=upscale_factor
        )
        
        if rfdn_weights and Path(rfdn_weights).exists():
            ckpt = torch.load(rfdn_weights, map_location=self.device)
            state = ckpt.get('model_state_dict', ckpt)
            self.sr_model.load_state_dict(state)
            print(f"  ✓ RFDN loaded: {rfdn_weights}")
        else:
            print(f"  ⚠️ RFDN weights not found, using random init")
        
        self.sr_model = self.sr_model.to(self.device)
        self.sr_model.eval()
        
        # =====================================================================
        # YOLO Detector
        # =====================================================================
        print("[2/2] Loading YOLO...")
        self.yolo = YOLO(yolo_weights)
        print(f"  ✓ YOLO loaded: {yolo_weights}")
        
        # 통계
        self.stats = {
            'total_images': 0,
            'pass2_triggered': 0
        }
        
        print(f"\n[Arch4Inference] Ready!")
        print(f"  Thresholds:")
        print(f"    - Low conf: {self.low_conf}")
        print(f"    - High conf: {self.high_conf}")
        print(f"    - Final conf: {self.final_conf}")
    
    # =========================================================================
    # Core Logic
    # =========================================================================
    
    def _needs_pass2(self, detections: list) -> bool:
        """
        2차 탐지 필요 여부 판단
        
        조건: low_conf < score < high_conf 인 객체가 있으면 True
        의미: "뭔가 있는 것 같은데 확실하지 않음" → SR로 확인!
        """
        if len(detections) == 0:
            return True  # 아무것도 없으면 SR로 다시 확인
        
        for det in detections:
            scores = det.get('scores', [])
            for score in scores:
                if self.low_conf < score < self.high_conf:
                    return True  # 애매한 객체 발견!
        
        return False
    
    def _merge_detections(self, det1: dict, det2: dict) -> dict:
        """
        1차 + 2차 탐지 결과 병합 (NMS)
        
        Args:
            det1: 1차 탐지 결과 (LR 업샘플 기반)
            det2: 2차 탐지 결과 (SR 기반)
        
        Returns:
            병합된 탐지 결과
        """
        device = self.device
        
        # 빈 결과 처리
        boxes1 = torch.tensor(det1.get('boxes', []), device=device)
        scores1 = torch.tensor(det1.get('scores', []), device=device)
        classes1 = torch.tensor(det1.get('classes', []), device=device)
        
        boxes2 = torch.tensor(det2.get('boxes', []), device=device)
        scores2 = torch.tensor(det2.get('scores', []), device=device)
        classes2 = torch.tensor(det2.get('classes', []), device=device)
        
        # 결합
        if len(boxes1) == 0 and len(boxes2) == 0:
            return {'boxes': [], 'scores': [], 'classes': []}
        
        if len(boxes1) == 0:
            all_boxes, all_scores, all_classes = boxes2, scores2, classes2
        elif len(boxes2) == 0:
            all_boxes, all_scores, all_classes = boxes1, scores1, classes1
        else:
            all_boxes = torch.cat([boxes1, boxes2], dim=0)
            all_scores = torch.cat([scores1, scores2], dim=0)
            all_classes = torch.cat([classes1, classes2], dim=0)
        
        # NMS
        keep = batched_nms(all_boxes, all_scores, all_classes.long(), self.merge_iou)
        
        # Final threshold
        final_mask = all_scores[keep] >= self.final_conf
        keep = keep[final_mask]
        
        return {
            'boxes': all_boxes[keep].cpu().numpy().tolist(),
            'scores': all_scores[keep].cpu().numpy().tolist(),
            'classes': all_classes[keep].cpu().numpy().astype(int).tolist()
        }
    
    def _parse_yolo_results(self, results) -> list:
        """YOLO 결과를 딕셔너리로 변환"""
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
    
    # =========================================================================
    # Inference
    # =========================================================================
    
    @torch.no_grad()
    def inference(
        self,
        lr_image: np.ndarray,
        return_intermediate: bool = False
    ) -> dict:
        """
        단일 이미지 추론
        
        Args:
            lr_image: LR 이미지 (BGR, numpy)
            return_intermediate: 중간 결과 반환 여부
        
        Returns:
            {
                'detections': 최종 탐지 결과,
                'pass2_triggered': 2차 탐지 실행 여부,
                'num_ships': 탐지된 선박 수,
                (optional) 'pass1_detections', 'pass2_detections', ...
            }
        """
        self.stats['total_images'] += 1
        
        # Preprocess
        lr_rgb = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        lr_tensor = torch.from_numpy(lr_rgb).permute(2, 0, 1).float() / 255.0
        lr_tensor = lr_tensor.unsqueeze(0).to(self.device)
        
        H, W = lr_image.shape[:2]
        hr_size = (W * self.upscale, H * self.upscale)
        
        # =================================================================
        # Pass 1: LR Upsampled → YOLO
        # =================================================================
        lr_upsampled = F.interpolate(
            lr_tensor,
            scale_factor=self.upscale,
            mode='bilinear',
            align_corners=False
        )
        lr_up_np = self._tensor_to_numpy(lr_upsampled)
        
        pass1_results = self.yolo(lr_up_np, conf=self.low_conf, verbose=False)
        pass1_det = self._parse_yolo_results(pass1_results)[0]
        
        # =================================================================
        # 2차 탐지 필요 여부 판단
        # =================================================================
        needs_pass2 = self._needs_pass2([pass1_det])
        
        hr_image = None
        pass2_det = None
        
        if needs_pass2:
            self.stats['pass2_triggered'] += 1
            
            # =============================================================
            # Pass 2: LR → SR → YOLO
            # =============================================================
            sr_tensor = self.sr_model(lr_tensor)
            hr_image = self._tensor_to_numpy(sr_tensor)
            
            pass2_results = self.yolo(hr_image, conf=self.low_conf, verbose=False)
            pass2_det = self._parse_yolo_results(pass2_results)[0]
            
            # 결과 병합
            final_det = self._merge_detections(pass1_det, pass2_det)
        else:
            # 1차 결과만 사용 (final threshold 적용)
            final_det = {
                'boxes': [],
                'scores': [],
                'classes': []
            }
            for box, score, cls in zip(pass1_det['boxes'], pass1_det['scores'], pass1_det['classes']):
                if score >= self.final_conf:
                    final_det['boxes'].append(box)
                    final_det['scores'].append(score)
                    final_det['classes'].append(cls)
        
        result = {
            'detections': final_det,
            'pass2_triggered': needs_pass2,
            'num_ships': len(final_det['boxes'])
        }
        
        if return_intermediate:
            result['pass1_detections'] = pass1_det
            result['pass2_detections'] = pass2_det
            result['lr_upsampled'] = lr_up_np
            result['hr_image'] = hr_image if hr_image is not None else lr_up_np
        
        return result
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Tensor → BGR numpy"""
        img = tensor.squeeze(0).cpu().clamp(0, 1)
        img = (img * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # =========================================================================
    # Batch Processing
    # =========================================================================
    
    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        save_vis: bool = True,
        save_json: bool = True
    ) -> dict:
        """폴더 전체 처리"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if save_vis:
            (output_path / 'visualizations').mkdir(exist_ok=True)
        
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        
        all_results = {}
        
        for img_path in tqdm(image_files, desc="Processing"):
            lr_image = cv2.imread(str(img_path))
            if lr_image is None:
                continue
            
            result = self.inference(lr_image, return_intermediate=True)
            
            # 시각화 저장
            if save_vis:
                vis_img = self.visualize_result(
                    lr_image,
                    result['hr_image'],
                    result['detections'],
                    result['pass2_triggered']
                )
                vis_path = output_path / 'visualizations' / f"vis_{img_path.name}"
                cv2.imwrite(str(vis_path), vis_img)
            
            all_results[img_path.name] = {
                'num_ships': result['num_ships'],
                'pass2_triggered': result['pass2_triggered'],
                'detections': result['detections']
            }
        
        # JSON 저장
        if save_json:
            json_path = output_path / 'results.json'
            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=2)
        
        # 통계 저장
        stats = self.get_stats()
        stats_path = output_path / 'stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n[Summary]")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Pass2 triggered: {stats['pass2_triggered']} ({stats['pass2_ratio']:.1%})")
        print(f"  Computational savings: ~{(1 - stats['pass2_ratio']) * 100:.1f}% SR skipped")
        
        return stats
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def visualize_result(
        self,
        lr_image: np.ndarray,
        hr_image: np.ndarray,
        detections: dict,
        pass2_triggered: bool
    ) -> np.ndarray:
        """결과 시각화"""
        # HR 이미지에 박스 그리기
        vis = hr_image.copy()
        
        for box, score in zip(detections['boxes'], detections['scores']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"Ship {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1-20), (x1+w+4, y1), (0, 255, 0), -1)
            cv2.putText(vis, label, (x1+2, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 상태 표시
        status = "2-Pass (SR Applied)" if pass2_triggered else "1-Pass (SR Skipped)"
        color = (0, 165, 255) if pass2_triggered else (0, 255, 0)
        cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(vis, f"{len(detections['boxes'])} ships", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # LR + HR 나란히
        lr_up = cv2.resize(lr_image, (hr_image.shape[1], hr_image.shape[0]))
        cv2.putText(lr_up, "LR (Upscaled)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return np.hstack([lr_up, vis])
    
    # =========================================================================
    # Analysis (논문용)
    # =========================================================================
    
    def compare_single_vs_dual(self, lr_image: np.ndarray) -> dict:
        """
        Single-pass vs Dual-pass 비교 (논문 Table용)
        
        Returns:
            - single_lr: LR 업샘플만으로 탐지
            - single_hr: SR 적용 후 탐지
            - adaptive: Arch4 (적응적)
        """
        lr_rgb = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        lr_tensor = torch.from_numpy(lr_rgb).permute(2, 0, 1).float() / 255.0
        lr_tensor = lr_tensor.unsqueeze(0).to(self.device)
        
        # LR Upsampled
        lr_up = F.interpolate(lr_tensor, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        lr_up_np = self._tensor_to_numpy(lr_up)
        
        # SR
        with torch.no_grad():
            sr = self.sr_model(lr_tensor)
        sr_np = self._tensor_to_numpy(sr)
        
        # Single-pass LR
        res_lr = self.yolo(lr_up_np, conf=self.final_conf, verbose=False)
        det_lr = self._parse_yolo_results(res_lr)[0]
        
        # Single-pass HR (항상 SR)
        res_hr = self.yolo(sr_np, conf=self.final_conf, verbose=False)
        det_hr = self._parse_yolo_results(res_hr)[0]
        
        # Adaptive (Arch4)
        adaptive = self.inference(lr_image, return_intermediate=True)
        
        return {
            'single_lr': {
                'detections': det_lr,
                'num_ships': len(det_lr['boxes']),
                'sr_applied': False
            },
            'single_hr': {
                'detections': det_hr,
                'num_ships': len(det_hr['boxes']),
                'sr_applied': True
            },
            'adaptive': {
                'detections': adaptive['detections'],
                'num_ships': adaptive['num_ships'],
                'sr_applied': adaptive['pass2_triggered']
            },
            'lr_upsampled': lr_up_np,
            'sr_image': sr_np
        }
    
    @torch.no_grad()
    def analyze_pass2_statistics(
        self,
        image_dir: str,
        num_images: int = 500
    ) -> dict:
        """
        Pass2 트리거 통계 분석 (논문 Table용)
        
        Returns:
            - pass2_ratio: 2차 탐지 비율
            - avg_detections_pass1: 1차 평균 탐지 수
            - avg_detections_final: 최종 평균 탐지 수
            - computational_savings: 연산 절약 비율
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg"))[:num_images]
        
        stats = {
            'pass2_triggered': 0,
            'total': 0,
            'pass1_det_counts': [],
            'final_det_counts': [],
            'pass2_det_counts': []
        }
        
        for img_path in tqdm(image_files, desc="Analyzing"):
            lr_image = cv2.imread(str(img_path))
            if lr_image is None:
                continue
            
            result = self.inference(lr_image, return_intermediate=True)
            
            stats['total'] += 1
            if result['pass2_triggered']:
                stats['pass2_triggered'] += 1
                stats['pass2_det_counts'].append(len(result['pass2_detections']['boxes']))
            
            stats['pass1_det_counts'].append(len(result['pass1_detections']['boxes']))
            stats['final_det_counts'].append(result['num_ships'])
        
        pass2_ratio = stats['pass2_triggered'] / max(stats['total'], 1)
        
        return {
            'total_images': stats['total'],
            'pass2_triggered': stats['pass2_triggered'],
            'pass2_ratio': pass2_ratio,
            'avg_pass1_detections': np.mean(stats['pass1_det_counts']),
            'avg_final_detections': np.mean(stats['final_det_counts']),
            'avg_pass2_detections': np.mean(stats['pass2_det_counts']) if stats['pass2_det_counts'] else 0,
            'computational_savings': f"{(1 - pass2_ratio) * 100:.1f}% SR skipped"
        }
    
    def plot_comparison(self, lr_image: np.ndarray, save_path: str = None):
        """Single vs Dual vs Adaptive 비교 시각화 (논문 Figure)"""
        comparison = self.compare_single_vs_dual(lr_image)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # LR Upsampled
        axes[0].imshow(cv2.cvtColor(comparison['lr_upsampled'], cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"LR Upsampled\n{comparison['single_lr']['num_ships']} ships", fontsize=12)
        axes[0].axis('off')
        
        # Single-pass LR
        vis_lr = self._draw_boxes(comparison['lr_upsampled'], comparison['single_lr']['detections'])
        axes[1].imshow(cv2.cvtColor(vis_lr, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Single-pass (LR)\nNo SR", fontsize=12)
        axes[1].axis('off')
        
        # Single-pass HR
        vis_hr = self._draw_boxes(comparison['sr_image'], comparison['single_hr']['detections'])
        axes[2].imshow(cv2.cvtColor(vis_hr, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Single-pass (HR)\nAlways SR\n{comparison['single_hr']['num_ships']} ships", fontsize=12)
        axes[2].axis('off')
        
        # Adaptive
        sr_status = "SR Applied" if comparison['adaptive']['sr_applied'] else "SR Skipped"
        hr_img = comparison['sr_image'] if comparison['adaptive']['sr_applied'] else comparison['lr_upsampled']
        vis_adapt = self._draw_boxes(hr_img, comparison['adaptive']['detections'])
        axes[3].imshow(cv2.cvtColor(vis_adapt, cv2.COLOR_BGR2RGB))
        axes[3].set_title(f"Adaptive (Arch4)\n{sr_status}\n{comparison['adaptive']['num_ships']} ships", fontsize=12)
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.show()
    
    def _draw_boxes(self, image: np.ndarray, detections: dict) -> np.ndarray:
        """이미지에 박스 그리기"""
        vis = image.copy()
        for box, score in zip(detections['boxes'], detections['scores']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{score:.2f}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return vis
    
    # =========================================================================
    # Stats
    # =========================================================================
    
    def get_stats(self) -> dict:
        """현재 통계 반환"""
        total = max(self.stats['total_images'], 1)
        return {
            'total_images': self.stats['total_images'],
            'pass2_triggered': self.stats['pass2_triggered'],
            'pass2_ratio': self.stats['pass2_triggered'] / total
        }
    
    def reset_stats(self):
        """통계 리셋"""
        self.stats = {'total_images': 0, 'pass2_triggered': 0}


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Arch4 Adaptive 2-Pass Inference")
    
    # Required
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or folder')
    parser.add_argument('--rfdn_weights', type=str, required=True,
                       help='RFDN checkpoint path')
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.pt',
                       help='YOLO checkpoint path')
    
    # Optional
    parser.add_argument('--output', type=str, default='results/arch4/',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    
    # Thresholds
    parser.add_argument('--low_conf', type=float, default=0.1)
    parser.add_argument('--high_conf', type=float, default=0.5)
    parser.add_argument('--final_conf', type=float, default=0.25)
    
    # Analysis
    parser.add_argument('--analyze', action='store_true',
                       help='Run Pass2 statistics analysis')
    parser.add_argument('--compare', action='store_true',
                       help='Compare single vs dual pass')
    
    args = parser.parse_args()
    
    # Initialize
    pipeline = Arch4Inference(
        rfdn_weights=args.rfdn_weights,
        yolo_weights=args.yolo_weights,
        device=args.device,
        low_conf_threshold=args.low_conf,
        high_conf_threshold=args.high_conf,
        final_conf_threshold=args.final_conf
    )
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.analyze and input_path.is_dir():
        # 통계 분석
        print("\n[Analysis Mode]")
        stats = pipeline.analyze_pass2_statistics(str(input_path))
        
        print("\n[Pass2 Statistics]")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
        # JSON 저장
        with open(output_path / 'analysis_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    elif args.compare and input_path.is_file():
        # 비교 시각화
        print("\n[Comparison Mode]")
        lr_image = cv2.imread(str(input_path))
        pipeline.plot_comparison(lr_image, str(output_path / 'comparison.png'))
    
    elif input_path.is_file():
        # 단일 이미지
        print("\n[Single Image Mode]")
        lr_image = cv2.imread(str(input_path))
        result = pipeline.inference(lr_image, return_intermediate=True)
        
        print(f"  Ships detected: {result['num_ships']}")
        print(f"  Pass2 triggered: {result['pass2_triggered']}")
        
        # 시각화 저장
        vis = pipeline.visualize_result(
            lr_image,
            result['hr_image'],
            result['detections'],
            result['pass2_triggered']
        )
        cv2.imwrite(str(output_path / f"result_{input_path.name}"), vis)
        print(f"✓ Saved to {output_path}")
    
    elif input_path.is_dir():
        # 폴더 처리
        print("\n[Batch Mode]")
        pipeline.process_folder(str(input_path), str(output_path))
    
    else:
        print(f"❌ Invalid input: {args.input}")


if __name__ == "__main__":
    main()


