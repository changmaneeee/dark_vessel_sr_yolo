"""Split Dataset

Split dataset into train/val/test sets.
"""

import argparse
import shutil
from pathlib import Path
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15], help="Train/Val/Test split")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    """Split dataset

    TODO: Implement dataset splitting
    - Get list of all images
    - Shuffle and split
    - Copy/move files to train/val/test directories
    - Create split_info.yaml
    """
    args = parse_args()
    # TODO: Implement
    pass


if __name__ == "__main__":
    main()
