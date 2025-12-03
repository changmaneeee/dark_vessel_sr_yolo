"""RFDN: Residual Feature Distillation Network

Lightweight SR model with residual feature distillation.
Reference: "Residual Feature Distillation Network for Lightweight Image Super-Resolution"
"""

import torch
import torch.nn as nn
from src.models.sr_models.base_sr import BaseSRModel
import torch.nn.functional as F

# =========================================================================
# License: MIT License
# Original Author: njulj (https://github.com/njulj/rfdn)
# Adapted by: AIS-SAT-PIPELINE Team for VLEO Ship Detection
# =========================================================================

def conv_layer(in_channels, out_channels, kernel_size, stride=1, strdie=1, dilation=1, groups=1):
    padding = ((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias = True, dilation=dilation, groups=groups)
    
class ESA(nn.Module):
    """Enhanced Spatial Attention (ESA) Module
    """
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        return x * m
    

class RFDB(nn.Module):
    """Residual Feature Distillation Block (RFDB)
    """
    def __init__(self, in_channels, distillation_rate=0.5):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = int(in_channels * distillation_rate)
        self.rc = self.remaining_channels = int(in_channels * (1 - distillation_rate))

        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)

        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)

        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)

        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)

        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)

        self.c5 = conv_layer(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)


    def forward(self, input):
        
        distilled_c1 = self.act(self.c1_d(input))

        r_c1 = self.c1_r(input)
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused + input
    





class RFDN(BaseSRModel):
    """Residual Feature Distillation Network for Super-Resolution

    TODO: Implement RFDN architecture
    - Shallow feature extraction
    - Multiple RFD blocks
    - Upsampling module (PixelShuffle)
    - Reconstruction
    """

    def __init__(self, scale_factor: int = 4, in_channels: int = 3, out_channels: int = 3):
        super().__init__(scale_factor, in_channels, out_channels)

        # TODO: Implement architecture
        pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode LR image to features

        TODO: Implement feature extraction
        """
        # Placeholder
        return x

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features to HR image

        TODO: Implement upsampling and reconstruction
        """
        # Placeholder
        return features
