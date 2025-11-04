# Copyright Niantic 2019. Patent Pending. All rights reserved.
# Licensed under the Monodepth2 non-commercial license.

from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from layers import ConvBlock, Conv3x3, upsample  # upsample is unused but kept for compatibility


class LightingDecoder(nn.Module):
    """
    Illumination (contrast/brightness) decoder that is geometry-friendly:
      - preserves your multi-scale dict interface:
        outputs[("lighting", i)], outputs[("contrast", i)], outputs[("brightness", i)]
      - uses bilinear upsampling (no checkerboard)
      - bounds magnitude: contrast g ~ [1-α, 1+α], brightness b ~ [-β, β]
      - enforces low-frequency via adaptive Gaussian blur (kernel never larger than map)
    """
    def __init__(self,
                 num_ch_enc,
                 scales=range(4),
                 num_output_channels=2,
                 use_skips=False,
                 alpha=0.10,     # max multiplicative deviation (±10%)
                 beta=0.05,      # max additive shift
                 kmax=11):       # max Gaussian kernel size (odd, adaptive)
        super().__init__()

        self.num_output_channels = num_output_channels  # expect 2 (contrast, brightness)
        self.use_skips = use_skips
        self.scales = list(scales)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kmax = int(kmax)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # ----- U-Net-like decoder (same structure as your original) -----
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

        for s in self.scales:
            self.convs[("lighting_conv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))

    # ----------------------- helpers -----------------------
    @staticmethod
    def _upsample_bilinear(x, scale_factor=2, size=None):
        return F.interpolate(x, scale_factor=scale_factor, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _gaussian_1d(k, sigma=None, device=None, dtype=None):
        if sigma is None:
            sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8  # common heuristic
        ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
        g = torch.exp(-(ax ** 2) / (2 * (sigma ** 2)))
        g = g / g.sum()
        return g  # [k]

    def _gaussian_kernel2d(self, k, device, dtype):
        g1 = self._gaussian_1d(k, device=device, dtype=dtype)          # [k]
        g2d = torch.outer(g1, g1).unsqueeze(0).unsqueeze(0)             # [1,1,k,k]
        return g2d

    def _gaussian_blur_depthwise(self, x):
        """
        Depthwise Gaussian blur with adaptive odd kernel (<= min(H,W)).
        x: [B,C,H,W]
        """
        B, C, H, W = x.shape
        k = min(self.kmax, H, W)
        if k < 3:
            return x
        if (k % 2) == 0:
            k -= 1
        if k < 3:
            return x

        kernel = self._gaussian_kernel2d(k, device=x.device, dtype=x.dtype)  # [1,1,k,k]
        kernel = kernel.expand(C, 1, k, k).contiguous()                       # [C,1,k,k] depthwise
        pad = k // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x, kernel, groups=C)

    # ----------------------- forward -----------------------
    def forward(self, input_features):
        """
        Returns:
            self.outputs: dict with keys
              ("lighting", i)   -> raw 2ch map before bounding (B,2,H,W)
              ("contrast", i)   -> multiplicative gain g in ~[1-α, 1+α] (B,1,H,W)
              ("brightness", i) -> additive bias b in ~[-β, β]        (B,1,H,W)
        """
        self.outputs = {}

        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            # use bilinear instead of nearest to avoid blocky artifacts
            x = self._upsample_bilinear(x)
            if self.use_skips and i > 0:
                x = torch.cat([x, input_features[i - 1]], dim=1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                # raw prediction (2 channels)
                light_raw = self.convs[("lighting_conv", i)](x)  # [B,2,H,W]
                self.outputs[("lighting", i)] = light_raw

                # bound with tanh, then map to safe ranges
                c_hat = torch.tanh(light_raw[:, 0:1, :, :])      # [-1,1]
                b_hat = torch.tanh(light_raw[:, 1:2, :, :])      # [-1,1]

                Ct = 1.0 + self.alpha * c_hat                    # multiplicative gain
                Bt = self.beta * b_hat                           # additive bias

                # enforce low-frequency so it can't explain geometry edges
                Ct = self._gaussian_blur_depthwise(Ct)
                Bt = self._gaussian_blur_depthwise(Bt)

                self.outputs[("contrast", i)] = Ct
                self.outputs[("brightness", i)] = Bt

        return self.outputs