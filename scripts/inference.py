"""Inference Script

Run inference on single images or image directory.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="AIS-SAT Pipeline Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image_dir", type=str, help="Image directory")
    parser.add_argument("--output", type=str, default="results/inference")
    parser.add_argument("--save_sr", action="store_true", help="Save SR images")
    return parser.parse_args()


def main():
    """Main inference function

    TODO: Implement inference
    - Load model
    - Run on image(s)
    - Visualize and save results
    """
    args = parse_args()
    # TODO: Implement
    pass


if __name__ == "__main__":
    main()
