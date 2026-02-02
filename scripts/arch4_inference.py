"""
=============================================================================
Arch4 Adaptive 2-Pass Pipeline - 추론 및 평가 스크립트
=============================================================================

[실행 방법]
python arch4_inference.py --mode baseline  # Arch0 모드 (항상 SR)
python arch4_inference.py --mode adaptive  # Arch4 모드 (adaptive)
python arch4_inference.py --mode compare   # 둘 다 비교

[경로 설정]
아래 CONFIG 딕셔너리에서 경로를 수정하세요.
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import argparse
from tqdm import tqdm
import numpy as np
from torchvision.ops import batched_nms, box_iou
import json
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # 가중치 경로
    "yolo_lr_weights": "/home/jovyan/changmin/yolov8s+airbus_smartdata/weights/best.pt",
    "yolo_hr_weights": "/home/jovyan/changmin/yolov8s+HR_airbus_smartdata/weights/best.pt",
    "rfdn_weights": "/home/jovyan/changmin/rfdn_model/experiment/rfdn_smart_airbus_final_fix/model/model_best.pt",  # 필요시 수정
    
    # 데이터 경로
    "hr_images": "/home/jovyan/changmin/cv_ship_detact/datas/smart_airbus_dataset/images/val",
    "lr_images": "/home/jovyan/changmin/cv_ship_detact/datas/smart_airbus_dataset_lr/images/val",
    "labels": "/home/jovyan/changmin/cv_ship_detact/datas/smart_airbus_dataset/labels/val",
    
    # 모델 설정
    "num_classes": 1,
    "upscale_factor": 4,
    "img_size": 640,  # YOLO 입력 크기
    
    # Threshold 설정 (나중에 탐색할 값들)
    "low_conf_threshold": 0.1,
    "high_conf_threshold": 0.5,
    "merge_iou_threshold": 0.5,
    "final_conf_threshold": 0.25,
    
    # 평가 설정
    "iou_threshold_eval": 0.5,  # mAP 계산용
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 1,  # 메모리 절약을 위해 1로 시작
}


# ============================================================================
# Data Loading
# ============================================================================

def load_image(path: str, target_size: Optional[int] = None) -> torch.Tensor:
    """이미지 로드 및 전처리"""
    from PIL import Image
    import torchvision.transforms as T
    
    img = Image.open(path).convert('RGB')
    
    if target_size:
        # Letterbox 또는 Resize
        img = img.resize((target_size, target_size), Image.BILINEAR)
    
    transform = T.Compose([
        T.ToTensor(),  # [0, 1] 범위
    ])
    
    return transform(img)


def load_labels(label_path: str, img_w: int, img_h: int) -> torch.Tensor:
    """
    YOLO 형식 라벨 로드
    
    Returns:
        boxes: [N, 4] (x1, y1, x2, y2) pixel coordinates
        classes: [N]
    """
    boxes = []
    classes = []
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_center = float(parts[1]) * img_w
                    y_center = float(parts[2]) * img_h
                    w = float(parts[3]) * img_w
                    h = float(parts[4]) * img_h
                    
                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    x2 = x_center + w / 2
                    y2 = y_center + h / 2
                    
                    boxes.append([x1, y1, x2, y2])
                    classes.append(cls)
    
    if boxes:
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(classes, dtype=torch.long)
    else:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)


def get_image_label_pairs(img_dir: str, label_dir: str) -> List[Tuple[str, str]]:
    """이미지-라벨 쌍 찾기"""
    pairs = []
    
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for img_file in sorted(os.listdir(img_dir)):
        if any(img_file.lower().endswith(ext) for ext in img_extensions):
            img_path = os.path.join(img_dir, img_file)
            
            # 라벨 파일 찾기
            base_name = os.path.splitext(img_file)[0]
            label_path = os.path.join(label_dir, base_name + '.txt')
            
            pairs.append((img_path, label_path))
    
    return pairs


# ============================================================================
# Model Loading
# ============================================================================

def load_yolo(weights_path: str, device: str) -> Any:
    """Ultralytics YOLO 로드"""
    from ultralytics import YOLO
    
    print(f"[YOLO] Loading: {weights_path}")
    model = YOLO(weights_path)
    model.to(device)
    return model


def load_rfdn(weights_path: str, device: str, nf: int = 50, upscale: int = 4) -> torch.nn.Module:
    """RFDN 로드"""
    # 현재 디렉토리에서 import 시도
    try:
        from src.models.sr_models.rfdn import RFDN
    except ImportError:
        print("[RFDN] src.models.sr_models.rfdn import 실패, 직접 정의 사용")
        # 간단한 RFDN 대체 (실제로는 전체 코드 필요)
        raise ImportError("RFDN 모듈을 찾을 수 없습니다. src/models/sr_models/rfdn.py 경로를 확인하세요.")
    
    model = RFDN(
        in_channels=3,
        out_channels=3,
        nf=nf,
        num_modules=4,
        upscale=upscale,
        input_range='0-255'
    )
    
    if os.path.exists(weights_path):
        print(f"[RFDN] Loading weights: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        print("[RFDN] ✓ Weights loaded")
    else:
        print(f"[RFDN] ⚠️ Weights not found: {weights_path}")
    
    model.to(device)
    model.eval()
    return model


# ============================================================================
# Arch4 Adaptive Pipeline
# ============================================================================

class Arch4Pipeline:
    """
    Arch4 Adaptive 2-Pass Pipeline
    
    Pass 1: LR → Bilinear Upscale → YOLO_LR
    Pass 2: LR → SR (RFDN) → YOLO_HR (조건부)
    """
    
    def __init__(
        self,
        yolo_lr: Any,
        yolo_hr: Any,
        sr_model: torch.nn.Module,
        config: Dict
    ):
        self.yolo_lr = yolo_lr
        self.yolo_hr = yolo_hr
        self.sr_model = sr_model
        self.config = config
        self.device = config['device']
        
        # Thresholds
        self.low_conf = config['low_conf_threshold']
        self.high_conf = config['high_conf_threshold']
        self.merge_iou = config['merge_iou_threshold']
        self.final_conf = config['final_conf_threshold']
        self.upscale = config['upscale_factor']
        
        # 통계
        self.stats = {
            'total': 0,
            'pass2_triggered': 0,
        }
    
    def set_thresholds(self, low_conf: float, high_conf: float):
        """Threshold 변경"""
        self.low_conf = low_conf
        self.high_conf = high_conf
        print(f"[Arch4] Thresholds: low={low_conf}, high={high_conf}")
    
    def _needs_pass2(self, detections: List[Dict]) -> List[bool]:
        """Pass 2 필요 여부 판단"""
        needs = []
        for det in detections:
            scores = det.get('scores', torch.tensor([]))
            if len(scores) == 0:
                needs.append(True)  # 탐지 없음 → Pass2
            else:
                # low < score < high 인 객체가 있으면 Pass2
                uncertain = (scores > self.low_conf) & (scores < self.high_conf)
                needs.append(uncertain.any().item())
        return needs
    
    def _yolo_predict(self, model, images: torch.Tensor, conf: float = 0.001) -> List[Dict]:
        """YOLO 추론"""
        results = model.predict(
            source=images,
            conf=conf,
            iou=0.45,
            verbose=False,
            device=self.device
        )
        
        outputs = []
        for r in results:
            boxes = r.boxes
            outputs.append({
                'boxes': boxes.xyxy.cpu() if boxes.xyxy.numel() > 0 else torch.zeros(0, 4),
                'scores': boxes.conf.cpu() if boxes.conf.numel() > 0 else torch.zeros(0),
                'classes': boxes.cls.cpu() if boxes.cls.numel() > 0 else torch.zeros(0)
            })
        return outputs
    
    def _merge_detections(self, det1: Dict, det2: Dict) -> Dict:
        """두 탐지 결과 병합"""
        device = det1['boxes'].device
        
        all_boxes = torch.cat([det1['boxes'], det2['boxes']], dim=0)
        all_scores = torch.cat([det1['scores'], det2['scores']], dim=0)
        all_classes = torch.cat([det1['classes'], det2['classes']], dim=0)
        
        if len(all_boxes) == 0:
            return {'boxes': torch.zeros(0, 4), 'scores': torch.zeros(0), 'classes': torch.zeros(0)}
        
        # Batched NMS
        keep = batched_nms(all_boxes, all_scores, all_classes.long(), self.merge_iou)
        
        # Final confidence filter
        final_mask = all_scores[keep] >= self.final_conf
        keep = keep[final_mask]
        
        return {
            'boxes': all_boxes[keep],
            'scores': all_scores[keep],
            'classes': all_classes[keep]
        }
    
    @torch.no_grad()
    def inference_baseline(self, lr_image: torch.Tensor) -> Dict:
        """
        Baseline (Arch0): 항상 SR → YOLO_HR
        """
        # SR 적용
        lr_255 = lr_image * 255.0
        hr_255 = self.sr_model(lr_255.to(self.device))
        hr_image = torch.clamp(hr_255 / 255.0, 0, 1)
        
        # YOLO_HR 추론
        detections = self._yolo_predict(self.yolo_hr, hr_image, conf=self.final_conf)
        
        return {
            'detections': detections,
            'hr_image': hr_image,
            'mode': 'baseline'
        }
    
    @torch.no_grad()
    def inference_adaptive(self, lr_image: torch.Tensor) -> Dict:
        """
        Arch4 Adaptive: 조건부 SR
        """
        self.stats['total'] += 1
        
        # Pass 1: Bilinear upscale → YOLO_LR
        lr_up = F.interpolate(
            lr_image.to(self.device),
            scale_factor=self.upscale,
            mode='bilinear',
            align_corners=False
        )
        
        pass1_det = self._yolo_predict(self.yolo_lr, lr_up, conf=self.low_conf)
        
        # Pass 2 필요 여부
        needs_pass2 = self._needs_pass2(pass1_det)
        
        if needs_pass2[0]:  # batch_size=1 가정
            self.stats['pass2_triggered'] += 1
            
            # SR 적용
            lr_255 = lr_image * 255.0
            hr_255 = self.sr_model(lr_255.to(self.device))
            hr_image = torch.clamp(hr_255 / 255.0, 0, 1)
            
            # YOLO_HR 추론
            pass2_det = self._yolo_predict(self.yolo_hr, hr_image, conf=self.low_conf)
            
            # 병합
            final_det = self._merge_detections(pass1_det[0], pass2_det[0])
        else:
            hr_image = lr_up
            # Pass1 결과에서 final_conf 필터링
            det = pass1_det[0]
            mask = det['scores'] >= self.final_conf
            final_det = {
                'boxes': det['boxes'][mask],
                'scores': det['scores'][mask],
                'classes': det['classes'][mask]
            }
        
        return {
            'detections': [final_det],
            'hr_image': hr_image,
            'pass2_triggered': needs_pass2[0],
            'mode': 'adaptive'
        }
    
    def get_stats(self) -> Dict:
        """통계 반환"""
        total = max(self.stats['total'], 1)
        return {
            'total': self.stats['total'],
            'pass2_triggered': self.stats['pass2_triggered'],
            'pass2_ratio': self.stats['pass2_triggered'] / total
        }
    
    def reset_stats(self):
        """통계 리셋"""
        self.stats = {'total': 0, 'pass2_triggered': 0}


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Average Precision 계산 (VOC 방식)"""
    # Append sentinel values
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])
    
    # Ensure precision is decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Find points where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    
    # Calculate AP
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return float(ap)


def evaluate_detections(
    all_predictions: List[Dict],
    all_gt: List[Tuple[torch.Tensor, torch.Tensor]],
    iou_threshold: float = 0.5,
    conf_thresholds: Optional[List[float]] = None
) -> Dict:
    """
    Detection 성능 평가
    
    Args:
        all_predictions: 각 이미지의 예측 결과 [{'boxes', 'scores', 'classes'}, ...]
        all_gt: 각 이미지의 GT [(boxes, classes), ...]
        iou_threshold: IoU threshold for TP
        
    Returns:
        metrics: {mAP, recall, precision, f1, ...}
    """
    if conf_thresholds is None:
        conf_thresholds = np.arange(0.0, 1.0, 0.01)
    
    # 모든 예측과 GT 수집
    all_scores = []
    all_tp = []
    total_gt = 0
    
    for pred, (gt_boxes, gt_classes) in zip(all_predictions, all_gt):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        
        num_gt = len(gt_boxes)
        total_gt += num_gt
        
        if len(pred_boxes) == 0:
            continue
        
        if num_gt == 0:
            # FP만 있음
            all_scores.extend(pred_scores.numpy())
            all_tp.extend([0] * len(pred_scores))
            continue
        
        # IoU 계산
        ious = box_iou(pred_boxes, gt_boxes)
        
        # 각 예측에 대해 TP/FP 판정
        gt_matched = [False] * num_gt
        
        # Score 순으로 정렬
        sorted_indices = torch.argsort(pred_scores, descending=True)
        
        for idx in sorted_indices:
            score = pred_scores[idx].item()
            iou_row = ious[idx]
            
            # 가장 높은 IoU의 GT 찾기
            max_iou, max_gt_idx = iou_row.max(dim=0)
            max_gt_idx = max_gt_idx.item()
            
            if max_iou >= iou_threshold and not gt_matched[max_gt_idx]:
                # TP
                all_scores.append(score)
                all_tp.append(1)
                gt_matched[max_gt_idx] = True
            else:
                # FP
                all_scores.append(score)
                all_tp.append(0)
    
    if len(all_scores) == 0:
        return {
            'mAP': 0.0,
            'recall': 0.0,
            'precision': 0.0,
            'f1': 0.0,
            'total_gt': total_gt,
            'total_pred': 0
        }
    
    # Score로 정렬
    sorted_indices = np.argsort(all_scores)[::-1]
    all_scores = np.array(all_scores)[sorted_indices]
    all_tp = np.array(all_tp)[sorted_indices]
    
    # Cumulative TP, FP
    cum_tp = np.cumsum(all_tp)
    cum_fp = np.cumsum(1 - all_tp)
    
    # Precision, Recall
    precisions = cum_tp / (cum_tp + cum_fp + 1e-10)
    recalls = cum_tp / (total_gt + 1e-10)
    
    # AP
    ap = compute_ap(recalls, precisions)
    
    # 특정 conf threshold에서의 메트릭 (default: 0.25)
    conf_idx = np.searchsorted(-all_scores, -0.25)
    if conf_idx < len(precisions):
        precision_at_conf = precisions[conf_idx]
        recall_at_conf = recalls[conf_idx]
    else:
        precision_at_conf = precisions[-1] if len(precisions) > 0 else 0
        recall_at_conf = recalls[-1] if len(recalls) > 0 else 0
    
    f1 = 2 * precision_at_conf * recall_at_conf / (precision_at_conf + recall_at_conf + 1e-10)
    
    return {
        'mAP': ap,
        'recall': recall_at_conf,
        'precision': precision_at_conf,
        'f1': f1,
        'total_gt': total_gt,
        'total_pred': len(all_scores),
        'all_recalls': recalls.tolist(),
        'all_precisions': precisions.tolist()
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Arch4 Inference & Evaluation')
    parser.add_argument('--mode', type=str, default='compare',
                        choices=['baseline', 'adaptive', 'compare'],
                        help='Inference mode')
    parser.add_argument('--low_conf', type=float, default=None,
                        help='Low confidence threshold (adaptive mode)')
    parser.add_argument('--high_conf', type=float, default=None,
                        help='High confidence threshold (adaptive mode)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (None=all)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save detailed results to JSON')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Arch4 Adaptive 2-Pass Pipeline - Inference & Evaluation")
    print("=" * 70)
    
    # Config 업데이트
    config = CONFIG.copy()
    if args.low_conf is not None:
        config['low_conf_threshold'] = args.low_conf
    if args.high_conf is not None:
        config['high_conf_threshold'] = args.high_conf
    
    device = config['device']
    print(f"\n[Config]")
    print(f"  Device: {device}")
    print(f"  Mode: {args.mode}")
    print(f"  Low conf: {config['low_conf_threshold']}")
    print(f"  High conf: {config['high_conf_threshold']}")
    
    # =========================================================================
    # 경로 확인
    # =========================================================================
    print(f"\n[Paths]")
    for key in ['yolo_lr_weights', 'yolo_hr_weights', 'rfdn_weights', 
                'hr_images', 'lr_images', 'labels']:
        path = config[key]
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {key}: {path} [{status}]")
    
    # =========================================================================
    # 모델 로드
    # =========================================================================
    print(f"\n[Loading Models]")
    
    yolo_lr = load_yolo(config['yolo_lr_weights'], device)
    yolo_hr = load_yolo(config['yolo_hr_weights'], device)
    sr_model = load_rfdn(config['rfdn_weights'], device)
    
    # Pipeline 생성
    pipeline = Arch4Pipeline(yolo_lr, yolo_hr, sr_model, config)
    
    # =========================================================================
    # 데이터 로드
    # =========================================================================
    print(f"\n[Loading Data]")
    
    pairs = get_image_label_pairs(config['lr_images'], config['labels'])
    if args.num_samples:
        pairs = pairs[:args.num_samples]
    
    print(f"  Found {len(pairs)} image-label pairs")
    
    # =========================================================================
    # 추론 및 평가
    # =========================================================================
    
    results = {}
    
    if args.mode in ['baseline', 'compare']:
        print(f"\n[Baseline Mode (Arch0 - Always SR)]")
        
        all_preds = []
        all_gts = []
        
        for img_path, label_path in tqdm(pairs, desc="Baseline"):
            # LR 이미지 로드
            lr_img = load_image(img_path).unsqueeze(0)  # [1, 3, H, W]
            _, _, h, w = lr_img.shape
            
            # GT 로드 (HR 크기 기준)
            gt_boxes, gt_classes = load_labels(
                label_path, 
                w * config['upscale_factor'], 
                h * config['upscale_factor']
            )
            
            # 추론
            output = pipeline.inference_baseline(lr_img)
            
            all_preds.append(output['detections'][0])
            all_gts.append((gt_boxes, gt_classes))
        
        # 평가
        metrics = evaluate_detections(all_preds, all_gts, config['iou_threshold_eval'])
        results['baseline'] = metrics
        
        print(f"\n  [Baseline Results]")
        print(f"    mAP@0.5: {metrics['mAP']:.4f}")
        print(f"    Recall:  {metrics['recall']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
        print(f"    Total GT: {metrics['total_gt']}, Total Pred: {metrics['total_pred']}")
    
    if args.mode in ['adaptive', 'compare']:
        print(f"\n[Adaptive Mode (Arch4)]")
        print(f"  low_conf={config['low_conf_threshold']}, high_conf={config['high_conf_threshold']}")
        
        pipeline.reset_stats()
        all_preds = []
        all_gts = []
        
        for img_path, label_path in tqdm(pairs, desc="Adaptive"):
            # LR 이미지 로드
            lr_img = load_image(img_path).unsqueeze(0)
            _, _, h, w = lr_img.shape
            
            # GT 로드
            gt_boxes, gt_classes = load_labels(
                label_path,
                w * config['upscale_factor'],
                h * config['upscale_factor']
            )
            
            # 추론
            output = pipeline.inference_adaptive(lr_img)
            
            all_preds.append(output['detections'][0])
            all_gts.append((gt_boxes, gt_classes))
        
        # 평가
        metrics = evaluate_detections(all_preds, all_gts, config['iou_threshold_eval'])
        stats = pipeline.get_stats()
        
        results['adaptive'] = {
            **metrics,
            'pass2_ratio': stats['pass2_ratio'],
            'pass2_triggered': stats['pass2_triggered'],
            'total_images': stats['total']
        }
        
        print(f"\n  [Adaptive Results]")
        print(f"    mAP@0.5: {metrics['mAP']:.4f}")
        print(f"    Recall:  {metrics['recall']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
        print(f"    Pass2 Ratio: {stats['pass2_ratio']:.2%} ({stats['pass2_triggered']}/{stats['total']})")
    
    # =========================================================================
    # 비교 요약
    # =========================================================================
    
    if args.mode == 'compare':
        print(f"\n{'=' * 70}")
        print("COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Metric':<15} {'Baseline':<15} {'Adaptive':<15} {'Diff':<15}")
        print(f"{'-' * 60}")
        
        for metric in ['mAP', 'recall', 'precision', 'f1']:
            base_val = results['baseline'][metric]
            adap_val = results['adaptive'][metric]
            diff = adap_val - base_val
            diff_str = f"{diff:+.4f}" if diff != 0 else "0.0000"
            print(f"{metric:<15} {base_val:<15.4f} {adap_val:<15.4f} {diff_str:<15}")
        
        print(f"\nPass2 Ratio: {results['adaptive']['pass2_ratio']:.2%}")
        print(f"  → SR 적용을 {(1 - results['adaptive']['pass2_ratio']):.2%} 줄임")
    
    # =========================================================================
    # 결과 저장
    # =========================================================================
    
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"arch4_results_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[Results saved to {output_path}]")
    
    print("\n✓ Done!")
    return results


if __name__ == "__main__":
    main()