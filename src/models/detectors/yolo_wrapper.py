"""YOLO Wrapper for Ship Detection

Wrapper class for integrating YOLO models into the pipeline.
Supports YOLO from ultralytics.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOWrapper(nn.Module):
    """
    YOLOWrapper class for AIS-SAT-PIPELINE
    """

    def __init__(self, model_path, task='detect'):
        super(YOLOWrapper, self).__init__()

        print(f"Loading YOLO model from: {model_path}")
        self.yolo_model = YOLO(model_path)
        self.model = self.yolo_model.model

    def forward(self, x):
        """
        For Arch 0
        LR -> SR -> [HR Image] -> YOLO -> Results
        """
        return self.model(x)

    def extract_features(self, x):
        """
        For Arch 5
        When build Arch 5, this module help to Fusion SR+YOLO feature extraction
        """
        #추후 제작 예정
        pass

    def get_loss(self, preds, targets):
        """
        Calculate YOLO loss
        """
        
        return torch.tensor(0.0, device=preds[0].device if isinstance(preds, list) else preds.device)

