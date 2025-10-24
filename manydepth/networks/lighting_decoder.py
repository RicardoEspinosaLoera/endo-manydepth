from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *  # ConvBlock, Conv3x3, upsample

class LightingDecoder(nn.Module):
    """
    Two-head lighting decoder:
      ("contrast", s): ReLU >= 0
      ("brightness", s): Tanh in [-1, 1]
    """
    def __init__(self, num_ch_enc, scales=range(4), use_skips=False):
        super(LightingDecoder, self).__init__()
        self.use_skips = use_skips
        self.scales = list(scales)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # shared decoder trunk
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # per-scale heads
        for s in self.scales:
            self.convs[("contrast_conv", s)] = Conv3x3(self.num_ch_dec[s], 1)
            self.convs[("brightness_conv", s)] = Conv3x3(self.num_ch_dec[s], 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.act_contrast = nn.ReLU(inplace=True)
        self.act_brightness = nn.Tanh()

    def forward(self, input_features):
        outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]  # <-- removed mode kwarg
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                c = self.convs[("contrast_conv", i)](x)
                b = self.convs[("brightness_conv", i)](x)
                outputs[("contrast", i)] = self.act_contrast(c)
                outputs[("brightness", i)] = self.act_brightness(b)

        return outputs
