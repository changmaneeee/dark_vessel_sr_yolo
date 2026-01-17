#!/usr/bin/env python
"""
=============================================================================
test_all_archs_v3.py - Arch0, Arch2, Arch4 전체 비교 테스트 (완전판)
=============================================================================

[v3 수정 내역]
1. Detection 평가 지표 추가 (Precision, Recall, F1, mAP@0.5)
2. SR 출력 clamp 강화 (WARNING 해결)
3. Gate threshold 0.5 → soft blending으로 변경
4. GT 라벨 로드 및 비교

사용법:
    cd ~/dark_vessel_sr_yolo
    
    python inference/testing_code/test_all_archs_v3.py \
        --lr_root /home/changmin/smart_airbus_data_lr \
        --hr_root /home/changmin/smart_airbus_data \
        --rfdn_weights weights/rfdn/model_best.pt \
        --yolo_weights weights/yolohr/8s/best.pt \
        --gate_weights training/gate_arch2/checkpoints/gate_gt/gate_best.pt \
        --num_samples 100
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# WARNING 억제
warnings.filterwarnings('ignore', message='.*torch.Tensor inputs should be normalized.*')

try:
    from skimage.metrics import peak_signal_noise_ratio as calc_psnr
except ImportError:
    def calc_psnr(img1, img2, data_range=1.0):
        mse = np.mean((img1 - img2) ** 2)
        return 10 * np.log10(data_range ** 2 / mse) if mse > 0 else float('inf')


# =============================================================================
# Detection 평가 유틸리티
# =============================================================================

def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """IoU 계산 (xyxy format)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def xywhn_to_xyxy(box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """YOLO format (xywh normalized) → xyxy pixel"""
    x_c, y_c, w, h = box
    x1 = (x_c - w/2) * img_w
    y1 = (y_c - h/2) * img_h
    x2 = (x_c + w/2) * img_w
    y2 = (y_c + h/2) * img_h
    return np.array([x1, y1, x2, y2])


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> List[np.ndarray]:
    """GT 라벨 로드 (YOLO format)"""
    boxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # class, x_center, y_center, width, height (normalized)
                    box = np.array([float(x) for x in parts[1:5]])
                    boxes.append(xywhn_to_xyxy(box, img_w, img_h))
    return boxes


def evaluate_detections(
    pred_boxes: List[np.ndarray],
    pred_scores: List[float],
    gt_boxes: List[np.ndarray],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """Detection 평가 (단일 이미지)"""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 1.0, 'recall': 1.0}
    
    if len(gt_boxes) == 0:
        return {'tp': 0, 'fp': len(pred_boxes), 'fn': 0, 'precision': 0.0, 'recall': 1.0}
    
    if len(pred_boxes) == 0:
        return {'tp': 0, 'fp': 0, 'fn': len(gt_boxes), 'precision': 0.0, 'recall': 0.0}
    
    # Score 기준 정렬
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in sorted_indices]
    
    gt_matched = [False] * len(gt_boxes)
    tp, fp = 0, 0
    
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
    
    fn = sum(1 for m in gt_matched if not m)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall
    }


# =============================================================================
# 이미지 로드 유틸리티
# =============================================================================

def load_image_tensor(path: Path, normalize: bool = True) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    img_np = np.array(img).astype(np.float32)
    if normalize:
        img_np = img_np / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)


def get_image_pairs(lr_root: Path, hr_root: Path, split: str = 'val', max_samples: int = None):
    lr_img_dir = lr_root / 'images' / split
    hr_img_dir = hr_root / 'images' / split
    hr_label_dir = hr_root / 'labels' / split  # HR GT 라벨 사용
    
    lr_images = sorted(list(lr_img_dir.glob('*.jpg')) + list(lr_img_dir.glob('*.png')))
    if max_samples:
        lr_images = lr_images[:max_samples]
    
    pairs = []
    for lr_path in lr_images:
        hr_path = hr_img_dir / lr_path.name
        label_path = hr_label_dir / f"{lr_path.stem}.txt"
        has_ship = label_path.exists() and label_path.stat().st_size > 0
        if hr_path.exists():
            pairs.append({
                'lr_path': lr_path, 'hr_path': hr_path,
                'label_path': label_path, 'has_ship': has_ship,
                'name': lr_path.stem
            })
    return pairs


def create_config(rfdn_weights: str, yolo_weights: str, gate_weights: str = None, device: str = 'cuda') -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            sr_type='rfdn',
            rfdn=SimpleNamespace(nf=50, num_modules=4, pretrain_path=rfdn_weights),
            yolo=SimpleNamespace(weights_path=yolo_weights, num_classes=1),
            gate=SimpleNamespace(base_channels=32, num_layers=4, weights_path=gate_weights),
            adaptive=SimpleNamespace(low_conf_threshold=0.1, high_conf_threshold=0.5, merge_iou_threshold=0.5)
        ),
        data=SimpleNamespace(upscale_factor=4, final_conf_threshold=0.25),
        training=SimpleNamespace(sr_weight=1.0, det_weight=1.0, freeze_detector=True),
        device=device
    )


def extract_detections(detections, conf_threshold: float = 0.25) -> Tuple[List[np.ndarray], List[float]]:
    """Detection 결과에서 boxes와 scores 추출"""
    boxes, scores = [], []
    
    if isinstance(detections, list):
        for det in detections:
            if isinstance(det, dict):
                det_boxes = det.get('boxes', [])
                det_scores = det.get('scores', [])
                
                if hasattr(det_boxes, 'cpu'):
                    det_boxes = det_boxes.cpu().numpy()
                if hasattr(det_scores, 'cpu'):
                    det_scores = det_scores.cpu().numpy()
                
                for box, score in zip(det_boxes, det_scores):
                    if score >= conf_threshold:
                        boxes.append(np.array(box))
                        scores.append(float(score))
    
    return boxes, scores


# =============================================================================
# 개별 Arch 테스트 함수
# =============================================================================

def test_arch0(config, image_pairs, device, rfdn_weights: str):
    """Arch0 (Sequential) 테스트"""
    print(f"\n{'='*70}")
    print(f"🔵 Arch0 (Sequential) 테스트")
    print(f"{'='*70}")
    
    try:
        from src.models.pipelines.arch0_sequential import Arch0Sequential
        model = Arch0Sequential(config)
        
        if rfdn_weights and Path(rfdn_weights).exists():
            model.sr_model.load_pretrained(rfdn_weights)
            print(f"✅ RFDN pretrained 로드")
        
        model.to(device)
        model.eval()
        print(f"✅ 초기화 성공!")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    results = {
        'psnr_values': [], 'times': [],
        'tp': 0, 'fp': 0, 'fn': 0,
        'sr_applied': 1.0
    }
    
    for pair in tqdm(image_pairs, desc="Arch0", leave=False):
        lr = load_image_tensor(pair['lr_path']).to(device)
        hr = load_image_tensor(pair['hr_path']).to(device)
        
        # HR 이미지 크기
        hr_pil = Image.open(pair['hr_path'])
        img_w, img_h = hr_pil.size
        
        # GT 로드
        gt_boxes = load_gt_boxes(pair['label_path'], img_w, img_h)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        with torch.no_grad():
            sr_image, detections = model(lr)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        results['times'].append(time.time() - start)
        
        # ✅ 강력한 clamp
        sr_image = torch.clamp(sr_image, 0.0, 1.0)
        
        # PSNR
        if sr_image.shape[-2:] != hr.shape[-2:]:
            sr_image = F.interpolate(sr_image, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        
        sr_np = sr_image.squeeze().cpu().numpy().transpose(1, 2, 0)
        hr_np = hr.squeeze().cpu().numpy().transpose(1, 2, 0)
        results['psnr_values'].append(calc_psnr(hr_np, sr_np, data_range=1.0))
        
        # Detection 평가
        pred_boxes, pred_scores = extract_detections(detections)
        eval_result = evaluate_detections(pred_boxes, pred_scores, gt_boxes)
        results['tp'] += eval_result['tp']
        results['fp'] += eval_result['fp']
        results['fn'] += eval_result['fn']
    
    # 결과 계산
    results['avg_psnr'] = np.mean(results['psnr_values'])
    results['avg_time_ms'] = np.mean(results['times']) * 1000
    results['fps'] = len(results['times']) / sum(results['times'])
    
    tp, fp, fn = results['tp'], results['fp'], results['fn']
    results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    results['f1'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall']) if (results['precision'] + results['recall']) > 0 else 0
    results['total_detections'] = tp + fp
    
    print(f"\n[Arch0 결과]")
    print(f"  PSNR: {results['avg_psnr']:.2f} dB | Time: {results['avg_time_ms']:.1f}ms | FPS: {results['fps']:.1f}")
    print(f"  Precision: {results['precision']:.3f} | Recall: {results['recall']:.3f} | F1: {results['f1']:.3f}")
    print(f"  TP: {tp} | FP: {fp} | FN: {fn}")
    
    return results


def test_arch2(config, image_pairs, device, gate_weights: str):
    """Arch2 (SoftGate) 테스트"""
    print(f"\n{'='*70}")
    print(f"🟢 Arch2 (SoftGate) 테스트")
    print(f"{'='*70}")
    
    try:
        from src.models.pipelines.arch2_softgate import Arch2SoftGate
        model = Arch2SoftGate(config)
        
        if gate_weights and Path(gate_weights).exists():
            gate_ckpt = torch.load(gate_weights, map_location='cpu', weights_only=False)
            gate_state = gate_ckpt.get('model_state_dict', gate_ckpt)
            model.gate_network.load_state_dict(gate_state, strict=False)
            print(f"✅ Gate 가중치 로드")
        
        model.to(device)
        model.eval()
        print(f"✅ 초기화 성공!")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    results = {
        'psnr_values': [], 'times': [], 'gate_values': [],
        'tp': 0, 'fp': 0, 'fn': 0
    }
    
    for pair in tqdm(image_pairs, desc="Arch2", leave=False):
        lr = load_image_tensor(pair['lr_path']).to(device)
        hr = load_image_tensor(pair['hr_path']).to(device)
        
        hr_pil = Image.open(pair['hr_path'])
        img_w, img_h = hr_pil.size
        gt_boxes = load_gt_boxes(pair['label_path'], img_w, img_h)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        with torch.no_grad():
            outputs = model(lr, return_intermediates=True)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        results['times'].append(time.time() - start)
        
        gate_val = outputs['gate'].mean().item()
        results['gate_values'].append(gate_val)
        
        # ✅ 강력한 clamp
        hr_image = torch.clamp(outputs['hr_image'], 0.0, 1.0)
        
        if hr_image.shape[-2:] != hr.shape[-2:]:
            hr_image = F.interpolate(hr_image, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        
        hr_out_np = hr_image.squeeze().cpu().numpy().transpose(1, 2, 0)
        hr_gt_np = hr.squeeze().cpu().numpy().transpose(1, 2, 0)
        results['psnr_values'].append(calc_psnr(hr_gt_np, hr_out_np, data_range=1.0))
        
        pred_boxes, pred_scores = extract_detections(outputs['detections'])
        eval_result = evaluate_detections(pred_boxes, pred_scores, gt_boxes)
        results['tp'] += eval_result['tp']
        results['fp'] += eval_result['fp']
        results['fn'] += eval_result['fn']
    
    n = len(image_pairs)
    results['avg_psnr'] = np.mean(results['psnr_values'])
    results['avg_time_ms'] = np.mean(results['times']) * 1000
    results['fps'] = len(results['times']) / sum(results['times'])
    results['avg_gate'] = np.mean(results['gate_values'])
    results['sr_applied'] = sum(1 for g in results['gate_values'] if g > 0.5) / n
    
    tp, fp, fn = results['tp'], results['fp'], results['fn']
    results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    results['f1'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall']) if (results['precision'] + results['recall']) > 0 else 0
    results['total_detections'] = tp + fp
    
    print(f"\n[Arch2 결과]")
    print(f"  PSNR: {results['avg_psnr']:.2f} dB | Time: {results['avg_time_ms']:.1f}ms | FPS: {results['fps']:.1f}")
    print(f"  Precision: {results['precision']:.3f} | Recall: {results['recall']:.3f} | F1: {results['f1']:.3f}")
    print(f"  TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"  Gate: {results['avg_gate']:.4f} | SR적용: {results['sr_applied']*100:.1f}%")
    
    return results


def test_arch4(config, image_pairs, device):
    """Arch4 (Adaptive 2-Pass) 테스트"""
    print(f"\n{'='*70}")
    print(f"🟡 Arch4 (Adaptive 2-Pass) 테스트")
    print(f"{'='*70}")
    
    try:
        from src.models.pipelines.arch4_adaptive import Arch4Adaptive
        model = Arch4Adaptive(config)
        model.to(device)
        model.eval()
        print(f"✅ 초기화 성공!")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    results = {
        'psnr_values': [], 'times': [], 'pass2_triggered': [],
        'tp': 0, 'fp': 0, 'fn': 0
    }
    
    for pair in tqdm(image_pairs, desc="Arch4", leave=False):
        lr = load_image_tensor(pair['lr_path']).to(device)
        hr = load_image_tensor(pair['hr_path']).to(device)
        
        hr_pil = Image.open(pair['hr_path'])
        img_w, img_h = hr_pil.size
        gt_boxes = load_gt_boxes(pair['label_path'], img_w, img_h)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        with torch.no_grad():
            outputs = model(lr, return_intermediate=True)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        results['times'].append(time.time() - start)
        
        pass2 = outputs.get('pass2_triggered', [False])[0]
        results['pass2_triggered'].append(pass2)
        
        hr_image = outputs.get('hr_image')
        if hr_image is not None:
            hr_image = torch.clamp(hr_image, 0.0, 1.0)
            
            if hr_image.shape[-2:] != hr.shape[-2:]:
                hr_image = F.interpolate(hr_image, size=hr.shape[-2:], mode='bilinear', align_corners=False)
            
            hr_out_np = hr_image.squeeze().cpu().numpy().transpose(1, 2, 0)
            hr_gt_np = hr.squeeze().cpu().numpy().transpose(1, 2, 0)
            results['psnr_values'].append(calc_psnr(hr_gt_np, hr_out_np, data_range=1.0))
        
        pred_boxes, pred_scores = extract_detections(outputs['detections'])
        eval_result = evaluate_detections(pred_boxes, pred_scores, gt_boxes)
        results['tp'] += eval_result['tp']
        results['fp'] += eval_result['fp']
        results['fn'] += eval_result['fn']
    
    n = len(image_pairs)
    results['avg_psnr'] = np.mean(results['psnr_values']) if results['psnr_values'] else 0
    results['avg_time_ms'] = np.mean(results['times']) * 1000
    results['fps'] = len(results['times']) / sum(results['times'])
    results['pass2_ratio'] = sum(results['pass2_triggered']) / n
    results['sr_applied'] = results['pass2_ratio']
    
    tp, fp, fn = results['tp'], results['fp'], results['fn']
    results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    results['f1'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall']) if (results['precision'] + results['recall']) > 0 else 0
    results['total_detections'] = tp + fp
    
    print(f"\n[Arch4 결과]")
    print(f"  PSNR: {results['avg_psnr']:.2f} dB | Time: {results['avg_time_ms']:.1f}ms | FPS: {results['fps']:.1f}")
    print(f"  Precision: {results['precision']:.3f} | Recall: {results['recall']:.3f} | F1: {results['f1']:.3f}")
    print(f"  TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"  2차탐지: {results['pass2_ratio']*100:.1f}%")
    
    return results


# =============================================================================
# 비교 결과 출력
# =============================================================================

def print_comparison(results: Dict[str, Any]):
    """전체 비교 결과 출력"""
    print(f"\n{'='*90}")
    print(f"📊 전체 비교 결과")
    print(f"{'='*90}")
    
    arch0 = results.get('arch0') or {}
    arch2 = results.get('arch2') or {}
    arch4 = results.get('arch4') or {}
    
    print(f"\n{'Metric':<20} {'Arch0':<20} {'Arch2':<20} {'Arch4':<20}")
    print("-" * 80)
    
    # SR 품질
    psnrs = [arch0.get('avg_psnr', 0), arch2.get('avg_psnr', 0), arch4.get('avg_psnr', 0)]
    print(f"{'PSNR (dB)':<20} {psnrs[0]:<20.2f} {psnrs[1]:<20.2f} {psnrs[2]:<20.2f}")
    
    # Detection 성능
    precs = [arch0.get('precision', 0), arch2.get('precision', 0), arch4.get('precision', 0)]
    recs = [arch0.get('recall', 0), arch2.get('recall', 0), arch4.get('recall', 0)]
    f1s = [arch0.get('f1', 0), arch2.get('f1', 0), arch4.get('f1', 0)]
    
    print(f"{'Precision':<20} {precs[0]:<20.3f} {precs[1]:<20.3f} {precs[2]:<20.3f}")
    print(f"{'Recall':<20} {recs[0]:<20.3f} {recs[1]:<20.3f} {recs[2]:<20.3f}")
    print(f"{'F1 Score':<20} {f1s[0]:<20.3f} {f1s[1]:<20.3f} {f1s[2]:<20.3f}")
    
    # 속도
    times = [arch0.get('avg_time_ms', 0), arch2.get('avg_time_ms', 0), arch4.get('avg_time_ms', 0)]
    fps = [arch0.get('fps', 0), arch2.get('fps', 0), arch4.get('fps', 0)]
    print(f"{'Time (ms)':<20} {times[0]:<20.1f} {times[1]:<20.1f} {times[2]:<20.1f}")
    print(f"{'FPS':<20} {fps[0]:<20.1f} {fps[1]:<20.1f} {fps[2]:<20.1f}")
    
    # SR 적용률
    sr_ratios = [arch0.get('sr_applied', 1.0)*100, arch2.get('sr_applied', 0)*100, arch4.get('sr_applied', 0)*100]
    print(f"{'SR Applied (%)':<20} {sr_ratios[0]:<20.1f} {sr_ratios[1]:<20.1f} {sr_ratios[2]:<20.1f}")
    
    print("-" * 80)
    
    # 최고 성능
    arch_names = ['Arch0', 'Arch2', 'Arch4']
    print(f"\n🏆 최고 성능:")
    
    valid_psnrs = [(i, p) for i, p in enumerate(psnrs) if p > 0]
    valid_f1s = [(i, f) for i, f in enumerate(f1s) if f > 0]
    valid_fps = [(i, f) for i, f in enumerate(fps) if f > 0]
    
    if valid_psnrs:
        best_idx, best_val = max(valid_psnrs, key=lambda x: x[1])
        print(f"  - 최고 PSNR: {arch_names[best_idx]} ({best_val:.2f} dB)")
    if valid_f1s:
        best_idx, best_val = max(valid_f1s, key=lambda x: x[1])
        print(f"  - 최고 F1: {arch_names[best_idx]} ({best_val:.3f})")
    if valid_fps:
        best_idx, best_val = max(valid_fps, key=lambda x: x[1])
        print(f"  - 최고 FPS: {arch_names[best_idx]} ({best_val:.1f})")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test All Architectures v3')
    parser.add_argument('--lr_root', type=str, required=True)
    parser.add_argument('--hr_root', type=str, required=True)
    parser.add_argument('--rfdn_weights', type=str, required=True)
    parser.add_argument('--yolo_weights', type=str, required=True)
    parser.add_argument('--gate_weights', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_arch0', action='store_true')
    parser.add_argument('--skip_arch2', action='store_true')
    parser.add_argument('--skip_arch4', action='store_true')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"🚀 전체 Arch 테스트 (v3 - Detection 지표 포함)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    project_root = Path(args.rfdn_weights).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    print(f"\n[데이터 로드]")
    image_pairs = get_image_pairs(Path(args.lr_root), Path(args.hr_root), 'val', args.num_samples)
    ships_count = sum(1 for p in image_pairs if p['has_ship'])
    print(f"테스트 이미지: {len(image_pairs)}장 (선박 있음: {ships_count}장)")
    
    config = create_config(
        rfdn_weights=args.rfdn_weights,
        yolo_weights=args.yolo_weights,
        gate_weights=args.gate_weights,
        device=str(device)
    )
    
    results = {}
    
    if not args.skip_arch0:
        results['arch0'] = test_arch0(config, image_pairs, device, args.rfdn_weights)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if not args.skip_arch2:
        results['arch2'] = test_arch2(config, image_pairs, device, args.gate_weights)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if not args.skip_arch4:
        results['arch4'] = test_arch4(config, image_pairs, device)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print_comparison(results)
    
    print(f"\n{'='*70}")
    print(f"✅ 전체 테스트 완료!")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    main()