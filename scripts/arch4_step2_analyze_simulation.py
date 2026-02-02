#!/usr/bin/env python
"""
=============================================================================
arch4_step2_analyze_simulation.py - Efficiency vs Safety Trade-off Analyzer
=============================================================================
- Low Threshold 변화에 따른 [Recall 감소] vs [SR 부하 감소]를 시각화 (그래프/표)
- Recall 99%를 고집하지 않고, 실질적인 효율성(Efficiency) 타협점을 찾음
- 최종적으로 Cost Function을 통해 최적의 Low/High 값을 추천
"""

import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# ANSI 색상 코드 (수정됨: CYAN 확실히 포함)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'  # <--- 이 부분이 누락되었었습니다.
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def simulate_cost(all_dets, total_gt, low, high, penalty_miss=100, cost_sr=1):
    """비용 계산 함수"""
    detected_tp = sum(1 for d in all_dets if d['type'] == 'TP')
    tp_below_low = sum(1 for d in all_dets if d['type'] == 'TP' and d['conf'] < low)
    undetected_gt = total_gt - detected_tp
    
    total_missed = tp_below_low + undetected_gt
    sr_calls = sum(1 for d in all_dets if low <= d['conf'] < high)
    
    return (total_missed * penalty_miss) + (sr_calls * cost_sr), total_missed, sr_calls

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to step1 json output')
    parser.add_argument('--output_dir', type=str, default='result/analysis', help='Output directory for graphs')
    args = parser.parse_args()

    # 1. 데이터 로드
    print(f"{Colors.HEADER}📂 데이터 로드 중: {args.input} ...{Colors.END}")
    with open(args.input, 'r') as f:
        raw_data = json.load(f)
    
    total_gt = raw_data['total_gt_ships']
    
    all_detections = []
    for img in raw_data['data']:
        all_detections.extend(img['detections'])
    
    total_detections = len(all_detections)
    all_detections.sort(key=lambda x: x['conf'], reverse=True)
    
    print(f"   총 Detection 후보: {total_detections}개")
    print(f"   총 GT 선박 수: {total_gt}개")
    
    # ---------------------------------------------------------
    # 2. Threshold별 통계 산출 (표 출력)
    # ---------------------------------------------------------
    thresholds = np.arange(0.00, 0.51, 0.01)
    stats = {'thresh': [], 'recall': [], 'sr_workload': []}
    
    print(f"\n{Colors.BOLD}{'='*90}{Colors.END}")
    print(f"{Colors.YELLOW}📊 Low Threshold 타협점 분석 테이블{Colors.END}")
    print(f"{Colors.BOLD}{'='*90}{Colors.END}")
    print(f"{'Conf':<8} | {'Recall (안전도)':<15} | {'SR Workload (부하)':<18} | {'Precision':<12} | {'판단'}")
    print("-" * 90)
    
    for t in thresholds:
        filtered = [d for d in all_detections if d['conf'] >= t]
        count_filtered = len(filtered)
        
        tp = sum(1 for d in filtered if d['type'] == 'TP')
        fp = sum(1 for d in filtered if d['type'] == 'FP')
        
        recall = tp / total_gt if total_gt > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        workload = count_filtered / total_detections
        
        stats['thresh'].append(t)
        stats['recall'].append(recall)
        stats['sr_workload'].append(workload)
        
        if t % 0.05 == 0 or t == 0.01:
            comment = ""
            if recall >= 0.99: comment = f"{Colors.GREEN}Safe{Colors.END}"
            elif recall >= 0.97: comment = f"{Colors.CYAN}Balanced{Colors.END}"
            elif recall >= 0.90: comment = f"{Colors.YELLOW}Aggressive{Colors.END}"
            else: comment = f"{Colors.RED}Risky{Colors.END}"
            
            print(f"{t:<8.2f} | {recall*100:<13.2f}% | {workload*100:<16.2f}% | {precision*100:<10.2f}% | {comment}")

    # ---------------------------------------------------------
    # 3. 그래프 시각화
    # ---------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()
    line1, = ax1.plot(stats['thresh'], stats['recall'], 'b-', linewidth=2, label='Safety (Recall)')
    ax1.set_xlabel('Low Confidence Threshold')
    ax1.set_ylabel('Recall (Safety)', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    line2, = ax2.plot(stats['thresh'], stats['sr_workload'], 'r--', linewidth=2, label='SR Workload (Cost)')
    ax2.set_ylabel('SR Workload (% kept)', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Elbow Point (Recall >= 0.97 만족하면서 Workload 최소)
    candidates = [(t, r, w) for t, r, w in zip(stats['thresh'], stats['recall'], stats['sr_workload']) if r >= 0.97]
    if candidates:
        best_cand = min(candidates, key=lambda x: x[2])
        plt.title(f"Efficiency-Safety Trade-off Analysis\nElbow Point: Conf={best_cand[0]:.2f}")
        ax1.plot(best_cand[0], best_cand[1], 'go', markersize=10)
        ax1.annotate(f'Rec: {best_cand[0]:.2f}', xy=(best_cand[0], best_cand[1]), xytext=(best_cand[0]+0.05, best_cand[1]-0.05), arrowprops=dict(facecolor='black', shrink=0.05))

    ax1.legend([line1, line2], [l.get_label() for l in [line1, line2]], loc='center right')
    save_path = output_dir / 'arch4_tradeoff_analysis.png'
    plt.savefig(save_path)
    print(f"\n{Colors.BOLD}📊 그래프 저장 완료: {Colors.CYAN}{save_path}{Colors.END}")

    # ---------------------------------------------------------
    # 4. 최종 추천 값 계산 (Cost Simulation)
    # ---------------------------------------------------------
    print(f"\n{Colors.BOLD}{'='*90}{Colors.END}")
    print(f"{Colors.BLUE}🎲 최적값 자동 계산 중...{Colors.END}")
    
    min_cost = float('inf')
    opt_low = 0
    opt_high = 0
    
    low_range = np.arange(0.01, 0.3, 0.01)
    high_range = np.arange(0.4, 0.95, 0.05)
    
    for l in low_range:
        for h in high_range:
            if l >= h: continue
            cost, _, _ = simulate_cost(all_detections, total_gt, l, h)
            if cost < min_cost:
                min_cost = cost
                opt_low = l
                opt_high = h
                
    print(f"🏆 {Colors.YELLOW}[Best Cost Combination]{Colors.END}")
    print(f"   Low Conf: {Colors.GREEN}{opt_low:.2f}{Colors.END}")
    print(f"   High Conf: {Colors.GREEN}{opt_high:.2f}{Colors.END}")

    print(f"\n{Colors.BOLD}📢 최종 적용 가이드{Colors.END}")
    print(f"1. 추론 코드 설정: {Colors.CYAN}model.predict(conf={opt_low:.2f}){Colors.END}")
    print(f"2. 아키텍처 로직:")
    print(f"   if conf >= {opt_high:.2f}: [확정] (SR 패스)")
    print(f"   elif conf >= {opt_low:.2f}: [의심] (SR 실행)")
    print(f"   else: [무시]")

if __name__ == '__main__':
    main()