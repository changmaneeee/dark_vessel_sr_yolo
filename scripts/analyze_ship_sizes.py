"""Analyze Ship Sizes

Analyze ship size distribution in dataset to determine small ship threshold.
"""

import argparse
import json
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze ship sizes in dataset")
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=768)
    parser.add_argument("--gsd", type=float, default=1.5, help="Ground Sample Distance (m)")
    parser.add_argument("--output", type=str, default="analysis/ship_sizes.json")
    return parser.parse_args()


def main():
    """Analyze ship sizes

    TODO: Implement ship size analysis
    - Read YOLO labels
    - Convert normalized boxes to pixel sizes
    - Convert to meters using GSD
    - Generate histogram
    - Save statistics
    """
    args = parse_args()
    # TODO: Implement
    pass


if __name__ == "__main__":
    main()
