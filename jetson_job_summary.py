#!/usr/bin/env python3
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

PREFERRED_POWER_KEYS = [
    'VDD_IN',
    'POM_5V_IN',
    'VIN_SYS_5V0',
    'VDD_TOTAL',
]


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)


def get_nested(d: Any, path: str):
    cur = d
    for p in path.split('.'):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def find_first(d: Any, candidates):
    for c in candidates:
        val = get_nested(d, c)
        if val is not None:
            return val
    return None


def extract_metrics(d: dict):
    out = {}
    if not isinstance(d, dict):
        return out

    rd = None
    if isinstance(d.get('runs'), list) and d['runs']:
        first = d['runs'][0]
        if isinstance(first, dict):
            rd = first.get('results_dict')

    if isinstance(rd, dict):
        out['precision'] = rd.get('metrics/precision(B)')
        out['recall'] = rd.get('metrics/recall(B)')
        out['map50'] = rd.get('metrics/mAP50(B)')
        out['map5095'] = rd.get('metrics/mAP50-95(B)')
        out['tp50'] = rd.get('direct/tp50')
        out['fp50'] = rd.get('direct/fp50')
        out['fn50'] = rd.get('direct/fn50')
        out['precision50_direct'] = rd.get('direct/precision50')
        out['recall50_direct'] = rd.get('direct/recall50')

    candidates = {
        'avg_ms_per_image': ['meta.avg_ms_per_image', 'avg_ms_per_image', 'avg_ms', 'latency.avg_ms', 'summary.avg_ms'],
        'median_ms': ['median_ms', 'latency.median_ms', 'summary.median_ms'],
        'p95_ms': ['p95_ms', 'latency.p95_ms', 'summary.p95_ms'],
        'fps': ['fps', 'avg_fps', 'summary.fps'],
    }
    for k, paths in candidates.items():
        if k not in out or out[k] is None:
            out[k] = find_first(d, paths)

    return out


def parse_tegrastats_power(log_path: Path):
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors='ignore').splitlines()
    samples = {k: [] for k in PREFERRED_POWER_KEYS}
    for line in text:
        for key in PREFERRED_POWER_KEYS:
            # VDD_IN 10385mW/10385mW OR POM_5V_IN 2621/2621
            m = re.search(rf'\b{re.escape(key)}\s+(\d+)(?:mW)?/(\d+)(?:mW)?', line)
            if m:
                samples[key].append(float(m.group(1)))
                break
    for key in PREFERRED_POWER_KEYS:
        vals = samples[key]
        if vals:
            return {
                'power_source': key,
                'power_samples': len(vals),
                'avg_power_mw': statistics.mean(vals),
                'max_power_mw': max(vals),
                'min_power_mw': min(vals),
                'p95_power_mw': vals[min(len(vals)-1, max(0, math.ceil(len(vals)*0.95)-1))] if vals else None,
            }
    return {}


def main():
    if len(sys.argv) != 5:
        print('usage: jetson_job_summary.py <job_name> <metrics_json> <tegrastats_log> <out_json>')
        sys.exit(1)
    job_name = sys.argv[1]
    metrics_path = Path(sys.argv[2])
    tg_path = Path(sys.argv[3])
    out_path = Path(sys.argv[4])

    data = load_json(metrics_path)
    metrics = extract_metrics(data or {})
    power = parse_tegrastats_power(tg_path)

    avg_ms = metrics.get('avg_ms_per_image')
    avg_power_mw = power.get('avg_power_mw')
    energy_per_image_j = None
    if isinstance(avg_ms, (int, float)) and isinstance(avg_power_mw, (int, float)):
        energy_per_image_j = (avg_power_mw / 1000.0) * (avg_ms / 1000.0)

    out = {
        'job_name': job_name,
        'metrics_json': str(metrics_path),
        'tegrastats_log': str(tg_path),
        'metrics': metrics,
        'power': power,
        'derived': {
            'energy_per_image_j': energy_per_image_j,
            'images_per_joule': (1.0 / energy_per_image_j) if energy_per_image_j and energy_per_image_j > 0 else None,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
