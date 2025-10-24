# Copyright Niantic 2019. Patent Pending. All rights reserved.
# Licensed for non-commercial use under the Monodepth2 license.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *  # expects ConvBlock, Conv3x3, upsample

class LightingDecoder(nn.Module):
    """
    Lighting decoder with TWO HEADS:
      - contrast head: predicts contrast scale (>= 0), activated with ReLU
      - brightness head: predicts brightness shift in [-1, 1], activated with Tanh

    Returns outputs at requested `scales` under keys:
      ("contrast", s)   : [B, 1, H_s, W_s]
      ("brightness", s) : [B, 1, H_s, W_s]
    """
    def __init__(self, num_ch_enc, scales=range(4), use_skips=False):
        super(LightingDecoder, self).__init__()

        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = list(scales)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # ---------- shared decoder trunk ----------
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # ---------- per-scale HEADS ----------
        # separate 1-channel conv heads for contrast and brightness
        for s in self.scales:
            self.convs[("contrast_conv", s)] = Conv3x3(self.num_ch_dec[s], 1)
            self.convs[("brightness_conv", s)] = Conv3x3(self.num_ch_dec[s], 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))

        # activations
        self.act_contrast = nn.ReLU(inplace=True)
        self.act_brightness = nn.Tanh()

    def forward(self, input_features):
        outputs = {}

        # shared decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x, mode=self.upsample_mode)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            # heads at this scale
            if i in self.scales:
                c = self.convs[("contrast_conv", i)](x)
                b = self.convs[("brightness_conv", i)](x)

                # apply head-specific activations
                c = self.act_contrast(c)     # >= 0
                b = self.act_brightness(b)   # [-1, 1]

                outputs[("contrast", i)] = c
                outputs[("brightness", i)] = b

        return outputs
