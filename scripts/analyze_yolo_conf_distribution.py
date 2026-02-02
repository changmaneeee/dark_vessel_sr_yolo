#!/usr/bin/env python
"""
=============================================================================
analyze_yolo_conf_distribution.py - YOLO Confidence 값별 정확도 분석 (Fixed)
=============================================================================

[버그 수정]
- GT 라벨 다중 선박 올바르게 카운트
- 배치 처리 최적화
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import argparse
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml
from collections import defaultdict
from ultralytics import YOLO


def load_labels(label_path: Path) -> list:
    """YOLO 형식 라벨 로드 - 모든 선박 반환"""
    if not label_path.exists():
        return []
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])
                labels.append({
                    'class': cls,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': w,
                    'height': h
                })
    return labels


def yolo_to_xyxy(label: dict, img_w: int, img_h: int) -> list:
    """YOLO 형식 → [x1, y1, x2, y2] 변환"""
    x_center = label['x_center'] * img_w
    y_center = label['y_center'] * img_h
    w = label['width'] * img_w
    h = label['height'] * img_h
    
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    
    return [x1, y1, x2, y2]


def calculate_iou(box1: list, box2: list) -> float:
    """IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def match_detections(detections: list, gt_labels: list, img_w: int, img_h: int, iou_threshold: float = 0.5):
    """Detection과 GT 매칭"""
    gt_boxes = [yolo_to_xyxy(label, img_w, img_h) for label in gt_labels]
    gt_matched = [False] * len(gt_boxes)
    
    results = []
    
    # confidence 높은 순으로 정렬
    sorted_dets = sorted(detections, key=lambda x: x['conf'], reverse=True)
    
    for det in sorted_dets:
        det_box = det['box']
        det_conf = det['conf']
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = calculate_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            # True Positive
            gt_matched[best_gt_idx] = True
            results.append({
                'conf': det_conf,
                'type': 'TP',
                'iou': best_iou
            })
        else:
            # False Positive
            results.append({
                'conf': det_conf,
                'type': 'FP',
                'iou': best_iou
            })
    
    # False Negatives (놓친 GT)
    fn_count = sum(1 for matched in gt_matched if not matched)
    
    return results, fn_count, len(gt_labels)


class ConfidenceAnalyzer:
    """YOLO Confidence 분포 분석기"""
    
    def __init__(
        self,
        yolo_weights: str,
        data_yaml: str,
        device: str = 'cuda'
    ):
        self.device = device
        
        # Data 경로 파싱
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        data_path = Path(data_config.get('path', ''))
        self.images_dir = data_path / 'images' / 'val'
        self.labels_dir = data_path / 'labels' / 'val'
        
        print(f"\n{'='*70}")
        print(f"📊 YOLO Confidence 분포 분석")
        print(f"{'='*70}")
        print(f"YOLO weights: {yolo_weights}")
        print(f"Images: {self.images_dir}")
        print(f"Labels: {self.labels_dir}")
        
        # YOLO 로드
        self.yolo = YOLO(yolo_weights, verbose=False)
        
        # 결과 저장
        self.all_detections = []
        self.total_gt = 0
        self.total_fn = 0
        self.images_with_ships = 0
        self.images_without_ships = 0
        
        # 이미지별 상세 정보
        self.per_image_stats = []
        
    def analyze(self, max_images: int = None, conf_threshold: float = 0.001):
        """분석 실행"""
        
        image_files = sorted(self.images_dir.glob('*.jpg'))
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"\n분석할 이미지 수: {len(image_files)}")
        print(f"Detection conf threshold: {conf_threshold}")
        
        # 먼저 전체 GT 수 계산
        total_gt_check = 0
        for img_path in image_files:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            gt_labels = load_labels(label_path)
            total_gt_check += len(gt_labels)
        
        print(f"전체 GT 선박 수 (사전 계산): {total_gt_check}")
        
        for img_path in tqdm(image_files, desc="분석 중"):
            # 이미지 크기
            img = Image.open(img_path)
            img_w, img_h = img.size
            
            # GT 라벨 로드 (★ 모든 선박 포함)
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            gt_labels = load_labels(label_path)
            
            gt_count = len(gt_labels)
            self.total_gt += gt_count
            
            if gt_count > 0:
                self.images_with_ships += 1
            else:
                self.images_without_ships += 1
            
            # YOLO 추론 (아주 낮은 threshold)
            results = self.yolo.predict(
                str(img_path),
                conf=conf_threshold,
                iou=0.45,
                verbose=False
            )
            
            # Detection 파싱
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    det_box = boxes.xyxy[i].cpu().numpy().tolist()
                    det_conf = float(boxes.conf[i].cpu().numpy())
                    detections.append({
                        'box': det_box,
                        'conf': det_conf
                    })
            
            # 매칭
            matched_results, fn_count, _ = match_detections(
                detections, gt_labels, img_w, img_h, iou_threshold=0.5
            )
            
            self.all_detections.extend(matched_results)
            self.total_fn += fn_count
            
            # 이미지별 통계
            self.per_image_stats.append({
                'image': img_path.name,
                'gt_count': gt_count,
                'det_count': len(detections),
                'tp': sum(1 for d in matched_results if d['type'] == 'TP'),
                'fp': sum(1 for d in matched_results if d['type'] == 'FP'),
                'fn': fn_count
            })
        
        print(f"\n✓ 분석 완료")
        print(f"  - 선박 있는 이미지: {self.images_with_ships}")
        print(f"  - 선박 없는 이미지: {self.images_without_ships}")
        print(f"  - 총 GT 선박 수: {self.total_gt}")
        print(f"  - 총 Detection 수: {len(self.all_detections)}")
        print(f"  - 총 TP: {sum(1 for d in self.all_detections if d['type'] == 'TP')}")
        print(f"  - 총 FP: {sum(1 for d in self.all_detections if d['type'] == 'FP')}")
        print(f"  - 총 FN: {self.total_fn}")
    
    def generate_report(self, output_path: str = None):
        """분석 결과 리포트 생성"""
        
        # Conf 구간 정의
        conf_bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
        
        print(f"\n{'='*70}")
        print(f"📊 Confidence 구간별 분석")
        print(f"{'='*70}")
        
        print(f"\n{'Conf 구간':<15} {'Detection':<12} {'TP':<8} {'FP':<8} {'Precision':<12}")
        print("-" * 60)
        
        bin_stats = []
        
        for i in range(len(conf_bins) - 1):
            low = conf_bins[i]
            high = conf_bins[i + 1]
            
            bin_dets = [d for d in self.all_detections if low <= d['conf'] < high]
            
            tp = sum(1 for d in bin_dets if d['type'] == 'TP')
            fp = sum(1 for d in bin_dets if d['type'] == 'FP')
            total = tp + fp
            precision = tp / total if total > 0 else 0
            
            bin_stats.append({
                'range': f"{low:.2f}-{high:.2f}",
                'total': total,
                'tp': tp,
                'fp': fp,
                'precision': precision
            })
            
            print(f"{low:.2f} - {high:.2f}    {total:<12} {tp:<8} {fp:<8} {precision:.2%}")
        
        # Conf threshold별 누적 통계
        print(f"\n{'='*70}")
        print(f"📊 Conf Threshold별 누적 성능")
        print(f"{'='*70}")
        
        print(f"\n{'Threshold':<12} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 80)
        
        threshold_stats = []
        
        for threshold in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            filtered = [d for d in self.all_detections if d['conf'] >= threshold]
            
            tp = sum(1 for d in filtered if d['type'] == 'TP')
            fp = sum(1 for d in filtered if d['type'] == 'FP')
            fn = self.total_gt - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / self.total_gt if self.total_gt > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            threshold_stats.append({
                'threshold': threshold,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            print(f"{threshold:<12.2f} {tp:<8} {fp:<8} {fn:<8} {precision:<12.2%} {recall:<12.2%} {f1:<12.4f}")
        
        # 핵심 인사이트
        print(f"\n{'='*70}")
        print(f"💡 핵심 인사이트")
        print(f"{'='*70}")
        
        # conf < 0.05
        very_low_conf = [d for d in self.all_detections if d['conf'] < 0.05]
        very_low_tp = sum(1 for d in very_low_conf if d['type'] == 'TP')
        very_low_precision = very_low_tp / len(very_low_conf) if very_low_conf else 0
        
        print(f"\n[conf < 0.05 구간]")
        print(f"  - Detection 수: {len(very_low_conf)}")
        print(f"  - 실제 선박 (TP): {very_low_tp}")
        print(f"  - 정밀도: {very_low_precision:.2%}")
        print(f"  → {'무시해도 됨 ✓' if very_low_precision < 0.1 else '주의 필요!'}")
        
        # conf 0.05-0.15
        low_conf = [d for d in self.all_detections if 0.05 <= d['conf'] < 0.15]
        low_tp = sum(1 for d in low_conf if d['type'] == 'TP')
        low_precision = low_tp / len(low_conf) if low_conf else 0
        
        print(f"\n[conf 0.05-0.15 구간]")
        print(f"  - Detection 수: {len(low_conf)}")
        print(f"  - 실제 선박 (TP): {low_tp}")
        print(f"  - 정밀도: {low_precision:.2%}")
        
        # conf 0.15-0.25
        mid_conf = [d for d in self.all_detections if 0.15 <= d['conf'] < 0.25]
        mid_tp = sum(1 for d in mid_conf if d['type'] == 'TP')
        mid_precision = mid_tp / len(mid_conf) if mid_conf else 0
        
        print(f"\n[conf 0.15-0.25 구간]")
        print(f"  - Detection 수: {len(mid_conf)}")
        print(f"  - 실제 선박 (TP): {mid_tp}")
        print(f"  - 정밀도: {mid_precision:.2%}")
        
        # conf >= 0.25
        high_conf = [d for d in self.all_detections if d['conf'] >= 0.25]
        high_tp = sum(1 for d in high_conf if d['type'] == 'TP')
        high_precision = high_tp / len(high_conf) if high_conf else 0
        
        print(f"\n[conf >= 0.25 구간]")
        print(f"  - Detection 수: {len(high_conf)}")
        print(f"  - 실제 선박 (TP): {high_tp}")
        print(f"  - 정밀도: {high_precision:.2%}")
        
        # 놓친 선박 분석
        print(f"\n[놓친 선박 (FN) 분석]")
        
        tp_at_025 = sum(1 for d in self.all_detections if d['conf'] >= 0.25 and d['type'] == 'TP')
        fn_at_025 = self.total_gt - tp_at_025
        recall_at_025 = tp_at_025 / self.total_gt if self.total_gt > 0 else 0
        
        print(f"  - 총 GT 선박: {self.total_gt}")
        print(f"  - conf >= 0.25로 찾은 선박: {tp_at_025}")
        print(f"  - conf >= 0.25로 놓친 선박: {fn_at_025}")
        print(f"  - Recall @ 0.25: {recall_at_025:.2%}")
        
        tp_at_015 = sum(1 for d in self.all_detections if d['conf'] >= 0.15 and d['type'] == 'TP')
        fn_at_015 = self.total_gt - tp_at_015
        recall_at_015 = tp_at_015 / self.total_gt if self.total_gt > 0 else 0
        
        print(f"\n  - conf >= 0.15로 찾은 선박: {tp_at_015}")
        print(f"  - conf >= 0.15로 놓친 선박: {fn_at_015}")
        print(f"  - Recall @ 0.15: {recall_at_015:.2%}")
        print(f"  → 0.25 → 0.15 낮추면 추가로 찾는 선박: {tp_at_015 - tp_at_025}개")
        
        tp_at_005 = sum(1 for d in self.all_detections if d['conf'] >= 0.05 and d['type'] == 'TP')
        fn_at_005 = self.total_gt - tp_at_005
        recall_at_005 = tp_at_005 / self.total_gt if self.total_gt > 0 else 0
        
        print(f"\n  - conf >= 0.05로 찾은 선박: {tp_at_005}")
        print(f"  - conf >= 0.05로 놓친 선박: {fn_at_005}")
        print(f"  - Recall @ 0.05: {recall_at_005:.2%}")
        print(f"  → 0.15 → 0.05 낮추면 추가로 찾는 선박: {tp_at_005 - tp_at_015}개")
        
        # 탐지 0개인 이미지 분석
        print(f"\n[탐지 0개인 이미지 분석]")
        zero_det_images = [s for s in self.per_image_stats if s['det_count'] == 0]
        zero_det_with_ships = [s for s in zero_det_images if s['gt_count'] > 0]
        
        print(f"  - 탐지 0개 이미지 수: {len(zero_det_images)}")
        print(f"  - 그 중 실제 선박 있는 이미지: {len(zero_det_with_ships)}")
        print(f"  - 그 중 놓친 총 선박 수: {sum(s['gt_count'] for s in zero_det_with_ships)}")
        
        # 권장 threshold
        print(f"\n{'='*70}")
        print(f"🎯 Arch4 Threshold 권장값")
        print(f"{'='*70}")
        
        best_f1_stat = max(threshold_stats, key=lambda x: x['f1'])
        
        print(f"\n[데이터 기반 권장]")
        print(f"  - Best F1 threshold: {best_f1_stat['threshold']:.2f}")
        print(f"    (F1={best_f1_stat['f1']:.4f}, P={best_f1_stat['precision']:.2%}, R={best_f1_stat['recall']:.2%})")
        print(f"  - HIGH (confident): {best_f1_stat['threshold']:.2f}")
        print(f"  - MID (uncertain): {max(0.05, best_f1_stat['threshold'] - 0.10):.2f}")
        print(f"  - LOWEST (noise): 0.05 (정밀도 {very_low_precision:.1%})")
        
        # 결과 저장
        if output_path:
            results = {
                'bin_stats': bin_stats,
                'threshold_stats': threshold_stats,
                'per_image_stats': self.per_image_stats[:100],  # 처음 100개만
                'summary': {
                    'total_images': self.images_with_ships + self.images_without_ships,
                    'images_with_ships': self.images_with_ships,
                    'images_without_ships': self.images_without_ships,
                    'total_gt': self.total_gt,
                    'total_detections': len(self.all_detections),
                    'total_tp': sum(1 for d in self.all_detections if d['type'] == 'TP'),
                    'total_fp': sum(1 for d in self.all_detections if d['type'] == 'FP'),
                    'total_fn': self.total_fn,
                    'zero_det_images': len(zero_det_images),
                    'zero_det_with_ships': len(zero_det_with_ships),
                    'missed_ships_in_zero_det': sum(s['gt_count'] for s in zero_det_with_ships),
                    'best_f1_threshold': best_f1_stat['threshold'],
                    'best_f1': best_f1_stat['f1']
                }
            }
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n결과 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLO Confidence 분포 분석')
    
    parser.add_argument('--yolo_weights', type=str, required=True)
    parser.add_argument('--data_yaml', type=str, required=True)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--conf_threshold', type=float, default=0.001)
    parser.add_argument('--output', type=str, default='results/yolo_conf_analysis.json')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    analyzer = ConfidenceAnalyzer(
        yolo_weights=args.yolo_weights,
        data_yaml=args.data_yaml,
        device=args.device
    )
    
    analyzer.analyze(
        max_images=args.max_images,
        conf_threshold=args.conf_threshold
    )
    
    analyzer.generate_report(output_path=args.output)


if __name__ == '__main__':
    main()