"""
선박 크기 분포 분석 스크립트

YOLO format label 파일에서 선박 bounding box 크기를 추출하고
통계 및 시각화를 수행합니다.

사용법:
    python scripts/analyze_ship_sizes.py --label_dir /path/to/labels --image_size 768

YOLO format:
    class x_center y_center width height
    (모든 값은 0~1 정규화됨)

출력:
    - 콘솔: 통계 요약
    - ship_size_distribution.png: 히스토그램
    - ship_size_analysis.csv: 상세 데이터
"""

import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict


def parse_yolo_label(label_path: str) -> list:
    """
    YOLO format label 파일 파싱
    
    Args:
        label_path: label 파일 경로
        
    Returns:
        list of (class_id, x_center, y_center, width, height)
        width, height는 정규화된 값 (0~1)
    """
    boxes = []
    
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append((class_id, x_center, y_center, width, height))
    
    return boxes


def analyze_ship_sizes(
    label_dir: str,
    image_width: int = 768,
    image_height: int = 768,
    gsd: float = 1.5,  # Ground Sample Distance (meters per pixel)
    output_dir: str = '/home/octolab-rtx4090/Desktop/changmin/dark_vessel_sr_yolo/datas'
):
    """
    선박 크기 분포 분석
    
    Args:
        label_dir: label 파일 디렉토리
        image_width: 이미지 가로 픽셀
        image_height: 이미지 세로 픽셀
        gsd: Ground Sample Distance (m/pixel)
        output_dir: 출력 디렉토리 (None이면 label_dir 사용)
    """
    
    if output_dir is None:
        output_dir = label_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 수집
    all_widths_px = []      # 픽셀 단위 너비
    all_heights_px = []     # 픽셀 단위 높이
    all_lengths_px = []     # 긴 쪽 (선박 길이로 가정)
    all_areas_px = []       # 면적 (픽셀²)
    all_lengths_m = []      # 미터 단위 길이
    
    ship_per_image = []     # 이미지당 선박 수
    
    label_files = list(Path(label_dir).glob("*.txt"))
    
    print(f"\n{'='*60}")
    print(f"선박 크기 분포 분석")
    print(f"{'='*60}")
    print(f"Label 디렉토리: {label_dir}")
    print(f"총 Label 파일 수: {len(label_files)}")
    print(f"이미지 크기: {image_width}×{image_height} pixels")
    print(f"GSD: {gsd} m/pixel")
    print(f"{'='*60}\n")
    
    for label_path in label_files:
        boxes = parse_yolo_label(str(label_path))
        ship_per_image.append(len(boxes))
        
        for class_id, x_c, y_c, w, h in boxes:
            # 정규화 → 픽셀 변환
            width_px = w * image_width
            height_px = h * image_height
            
            # 긴 쪽 = 선박 길이 (선박은 보통 길쭉함)
            length_px = max(width_px, height_px)
            short_px = min(width_px, height_px)
            
            # 면적
            area_px = width_px * height_px
            
            # 미터 변환
            length_m = length_px * gsd
            
            all_widths_px.append(width_px)
            all_heights_px.append(height_px)
            all_lengths_px.append(length_px)
            all_areas_px.append(area_px)
            all_lengths_m.append(length_m)
    
    # NumPy 배열로 변환
    lengths_px = np.array(all_lengths_px)
    lengths_m = np.array(all_lengths_m)
    areas_px = np.array(all_areas_px)
    ship_counts = np.array(ship_per_image)
    
    total_ships = len(lengths_px)
    
    if total_ships == 0:
        print("⚠️ 선박이 발견되지 않았습니다!")
        return
    
    # ========================================
    # 통계 계산
    # ========================================
    
    print(f"📊 기본 통계")
    print(f"-" * 40)
    print(f"총 선박 수: {total_ships:,}")
    print(f"선박 있는 이미지: {(ship_counts > 0).sum():,} / {len(ship_counts):,}")
    print(f"이미지당 평균 선박: {ship_counts.mean():.2f}")
    print(f"이미지당 최대 선박: {ship_counts.max()}")
    print()
    
    print(f"📏 선박 길이 (픽셀 기준, HR {gsd}m GSD)")
    print(f"-" * 40)
    print(f"최소: {lengths_px.min():.1f} px ({lengths_m.min():.1f} m)")
    print(f"최대: {lengths_px.max():.1f} px ({lengths_m.max():.1f} m)")
    print(f"평균: {lengths_px.mean():.1f} px ({lengths_m.mean():.1f} m)")
    print(f"중간값: {np.median(lengths_px):.1f} px ({np.median(lengths_m):.1f} m)")
    print(f"표준편차: {lengths_px.std():.1f} px ({lengths_m.std():.1f} m)")
    print()
    
    # ========================================
    # 크기별 분류 (픽셀 기준)
    # ========================================
    
    # COCO 기준 (면적 기준)
    small_coco = (areas_px < 32**2).sum()
    medium_coco = ((areas_px >= 32**2) & (areas_px < 96**2)).sum()
    large_coco = (areas_px >= 96**2).sum()
    
    print(f"📦 COCO 크기 분류 (면적 기준)")
    print(f"-" * 40)
    print(f"Small  (<32²px):    {small_coco:,} ({small_coco/total_ships*100:.1f}%)")
    print(f"Medium (32²~96²px): {medium_coco:,} ({medium_coco/total_ships*100:.1f}%)")
    print(f"Large  (>96²px):    {large_coco:,} ({large_coco/total_ships*100:.1f}%)")
    print()
    
    # 우리 기준 (길이 기준, 미터)
    # 15m 미만, 15~30m, 30~60m, 60m 이상
    tiny_m = (lengths_m < 15).sum()
    small_m = ((lengths_m >= 15) & (lengths_m < 30)).sum()
    medium_m = ((lengths_m >= 30) & (lengths_m < 60)).sum()
    large_m = (lengths_m >= 60).sum()
    
    print(f"🚢 선박 크기 분류 (길이 기준, 미터)")
    print(f"-" * 40)
    print(f"Tiny   (<15m):      {tiny_m:,} ({tiny_m/total_ships*100:.1f}%)")
    print(f"Small  (15~30m):    {small_m:,} ({small_m/total_ships*100:.1f}%)")
    print(f"Medium (30~60m):    {medium_m:,} ({medium_m/total_ships*100:.1f}%)")
    print(f"Large  (>60m):      {large_m:,} ({large_m/total_ships*100:.1f}%)")
    print()
    
    # LR에서의 픽셀 크기 (6m GSD 가정)
    lr_gsd = 6.0  # 6m GSD
    lengths_lr_px = lengths_m / lr_gsd
    
    print(f"📉 LR (6m GSD)에서의 픽셀 크기")
    print(f"-" * 40)
    print(f"최소: {lengths_lr_px.min():.1f} px")
    print(f"최대: {lengths_lr_px.max():.1f} px")
    print(f"평균: {lengths_lr_px.mean():.1f} px")
    print(f"중간값: {np.median(lengths_lr_px):.1f} px")
    print()
    
    # LR 픽셀 기준 분류
    lr_1_3 = (lengths_lr_px < 3).sum()
    lr_3_5 = ((lengths_lr_px >= 3) & (lengths_lr_px < 5)).sum()
    lr_5_10 = ((lengths_lr_px >= 5) & (lengths_lr_px < 10)).sum()
    lr_10_plus = (lengths_lr_px >= 10).sum()
    
    print(f"⚠️ LR (6m GSD) 탐지 난이도 분류")
    print(f"-" * 40)
    print(f"매우 어려움 (<3px LR):  {lr_1_3:,} ({lr_1_3/total_ships*100:.1f}%) ← 거의 탐지 불가!")
    print(f"어려움 (3~5px LR):      {lr_3_5:,} ({lr_3_5/total_ships*100:.1f}%) ← SR 필수!")
    print(f"보통 (5~10px LR):       {lr_5_10:,} ({lr_5_10/total_ships*100:.1f}%) ← SR 도움됨")
    print(f"쉬움 (>10px LR):        {lr_10_plus:,} ({lr_10_plus/total_ships*100:.1f}%) ← LR에서도 가능")
    print()
    
    # ========================================
    # 시각화
    # ========================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 선박 길이 분포 (픽셀)
    ax1 = axes[0, 0]
    ax1.hist(lengths_px, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(lengths_px.mean(), color='red', linestyle='--', label=f'Mean: {lengths_px.mean():.1f}px')
    ax1.axvline(np.median(lengths_px), color='orange', linestyle='--', label=f'Median: {np.median(lengths_px):.1f}px')
    ax1.set_xlabel('Ship Length (pixels, HR)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Ship Length Distribution (HR, {gsd}m GSD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 선박 길이 분포 (미터)
    ax2 = axes[0, 1]
    ax2.hist(lengths_m, bins=50, edgecolor='black', alpha=0.7, color='forestgreen')
    ax2.axvline(15, color='red', linestyle='--', linewidth=2, label='15m (소형 어선)')
    ax2.axvline(30, color='orange', linestyle='--', linewidth=2, label='30m')
    ax2.axvline(60, color='purple', linestyle='--', linewidth=2, label='60m')
    ax2.set_xlabel('Ship Length (meters)')
    ax2.set_ylabel('Count')
    ax2.set_title('Ship Length Distribution (Real Size)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. LR 픽셀 크기 분포
    ax3 = axes[1, 0]
    ax3.hist(lengths_lr_px, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax3.axvline(3, color='red', linestyle='--', linewidth=2, label='3px (매우 어려움)')
    ax3.axvline(5, color='orange', linestyle='--', linewidth=2, label='5px (어려움)')
    ax3.axvline(10, color='green', linestyle='--', linewidth=2, label='10px (보통)')
    ax3.set_xlabel('Ship Length (pixels, LR 6m GSD)')
    ax3.set_ylabel('Count')
    ax3.set_title('Ship Length in LR (Detection Difficulty)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 크기별 비율 파이 차트
    ax4 = axes[1, 1]
    sizes = [tiny_m, small_m, medium_m, large_m]
    labels = [f'<15m\n({tiny_m/total_ships*100:.1f}%)',
              f'15-30m\n({small_m/total_ships*100:.1f}%)',
              f'30-60m\n({medium_m/total_ships*100:.1f}%)',
              f'>60m\n({large_m/total_ships*100:.1f}%)']
    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1']
    explode = (0.05, 0.05, 0, 0)
    
    ax4.pie(sizes, labels=labels, colors=colors, explode=explode,
            autopct='', startangle=90, shadow=True)
    ax4.set_title('Ship Size Distribution by Real Length')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'ship_size_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 히스토그램 저장: {output_path}")
    
    plt.close()
    
    # ========================================
    # CSV 저장
    # ========================================
    
    csv_path = os.path.join(output_dir, 'ship_size_analysis.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['length_px_hr', 'length_m', 'length_px_lr', 'area_px', 'difficulty'])
        
        for i in range(len(lengths_px)):
            if lengths_lr_px[i] < 3:
                difficulty = 'very_hard'
            elif lengths_lr_px[i] < 5:
                difficulty = 'hard'
            elif lengths_lr_px[i] < 10:
                difficulty = 'medium'
            else:
                difficulty = 'easy'
            
            writer.writerow([
                f'{lengths_px[i]:.2f}',
                f'{lengths_m[i]:.2f}',
                f'{lengths_lr_px[i]:.2f}',
                f'{areas_px[i]:.2f}',
                difficulty
            ])
    
    print(f"📄 상세 데이터 저장: {csv_path}")
    
    # ========================================
    # 요약 리포트
    # ========================================
    
    print()
    print(f"{'='*60}")
    print(f"📋 요약 리포트")
    print(f"{'='*60}")
    print()
    print(f"🎯 우리 연구 목표 (15m 선박 탐지):")
    print(f"   - 15m 미만 선박: {tiny_m:,}개 ({tiny_m/total_ships*100:.1f}%)")
    print(f"   - 15m 이상 선박: {total_ships - tiny_m:,}개 ({(total_ships-tiny_m)/total_ships*100:.1f}%)")
    print()
    print(f"⚠️ LR (6m GSD)에서 문제:")
    print(f"   - 3픽셀 미만 (탐지 거의 불가): {lr_1_3:,}개 ({lr_1_3/total_ships*100:.1f}%)")
    print(f"   - 5픽셀 미만 (SR 필수): {lr_1_3 + lr_3_5:,}개 ({(lr_1_3+lr_3_5)/total_ships*100:.1f}%)")
    print()
    print(f"💡 권장사항:")
    
    if lr_1_3 / total_ships > 0.3:
        print(f"   - ⚠️ 30% 이상이 LR에서 3픽셀 미만!")
        print(f"   - → GSD 3m로 변경 강력 권장")
        print(f"   - → 또는 2단계 탐지 (Arch 6) 적용")
    elif (lr_1_3 + lr_3_5) / total_ships > 0.3:
        print(f"   - ⚠️ 30% 이상이 LR에서 5픽셀 미만!")
        print(f"   - → SR 기반 Enhancement 필수")
        print(f"   - → Feature Fusion (Arch 5-B) 적용 권장")
    else:
        print(f"   - ✅ 대부분 선박이 LR에서도 탐지 가능 크기")
        print(f"   - → 기본 SR+Detection으로도 성능 개선 기대")
    
    print(f"{'='*60}")
    
    return {
        'total_ships': total_ships,
        'lengths_px': lengths_px,
        'lengths_m': lengths_m,
        'lengths_lr_px': lengths_lr_px,
        'size_distribution': {
            'tiny_m': tiny_m,
            'small_m': small_m,
            'medium_m': medium_m,
            'large_m': large_m
        },
        'lr_difficulty': {
            'very_hard': lr_1_3,
            'hard': lr_3_5,
            'medium': lr_5_10,
            'easy': lr_10_plus
        }
    }


def main():
    parser = argparse.ArgumentParser(description='선박 크기 분포 분석')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='YOLO format label 디렉토리')
    parser.add_argument('--image_width', type=int, default=768,
                        help='이미지 가로 크기 (default: 768)')
    parser.add_argument('--image_height', type=int, default=768,
                        help='이미지 세로 크기 (default: 768)')
    parser.add_argument('--gsd', type=float, default=1.5,
                        help='Ground Sample Distance in meters (default: 1.5)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='출력 디렉토리 (default: label_dir)')
    
    args = parser.parse_args()
    
    analyze_ship_sizes(
        label_dir=args.label_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        gsd=args.gsd,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
