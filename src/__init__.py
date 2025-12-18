"""AIS-SAT-PIPELINE Source Package

SR-Detection Feature Fusion Pipeline for VLEO Satellite Ship Detection
"""

__version__ = "0.1.1"
__author__ = "AIS-SAT Team"

from src import models, losses, utils

__all__ = [
    "models",
    "losses",
    "utils",
]
