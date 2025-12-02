"""Create LR Dataset

Apply degradation to HR images to create LR dataset.
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create LR dataset from HR")
    parser.add_argument("--hr_dir", type=str, required=True)
    parser.add_argument("--lr_dir", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--degradation", type=str, default="bicubic", choices=["bicubic", "realistic", "sensor_sim"])
    parser.add_argument("--noise_level", type=float, default=0.0)
    return parser.parse_args()


def main():
    """Create LR dataset

    TODO: Implement LR dataset creation
    - Load HR images
    - Apply degradation pipeline
    - Save LR images
    """
    args = parse_args()
    # TODO: Implement
    pass


if __name__ == "__main__":
    main()
