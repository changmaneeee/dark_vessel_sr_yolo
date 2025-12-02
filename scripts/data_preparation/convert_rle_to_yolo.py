"""Convert RLE Masks to YOLO Format

Convert Airbus RLE segmentation masks to YOLO bounding box format.
"""

import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert RLE to YOLO format")
    parser.add_argument("--csv", type=str, required=True, help="train_ship_segmentations_v2.csv")
    parser.add_argument("--images", type=str, required=True, help="Image directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--min_ship_size", type=int, default=15, help="Min ship size (pixels)")
    return parser.parse_args()


def rle_to_bbox(rle_mask, image_shape):
    """Convert RLE mask to bounding box

    TODO: Implement RLE to bbox conversion
    """
    pass


def main():
    """Main conversion function

    TODO: Implement RLE to YOLO conversion
    - Read CSV with RLE masks
    - Decode RLE to masks
    - Convert masks to bounding boxes
    - Save in YOLO format (class x_center y_center width height)
    """
    args = parse_args()
    # TODO: Implement
    pass


if __name__ == "__main__":
    main()
