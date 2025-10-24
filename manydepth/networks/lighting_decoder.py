# Copyright Niantic 2019. Patent Pending.
# Licensed for non-commercial use under the Monodepth2 license.

from __future__ import absolute_import, division, print_function
from typing import Dict, List, Tuple

import math
import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *  # expects ConvBlock, upsample

def _inv_softplus_scalar(y: float) -> float:
    """Inverse softplus for a positive scalar y."""
    return math.log(math.exp(y) - 1.0)

class LightingDecoder(nn.Module):
    """
    Lighting decoder with TWO HEADS per scale:

      - ("contrast", s): strictly positive contrast scale via Softplus(+eps), optionally clamped
      - ("brightness", s): bounded brightness shift via Tanh scaled by b_range

    Optionally also returns legacy 2-ch map:
      - ("lighting", s) = concat[contrast, brightness] for drop-in compatibility

    Args:
        num_ch_enc: list of encoder channel sizes (Monodepth-style)
        scales:     iterable of decoder scales to output (e.g., range(4) -> 0..3)
        use_skips:  if True, uses skip connections from encoder features
        eps:        small positive to keep contrast away from zero
        c_max:      upper clamp for contrast to avoid exploding photometric loss (None disables)
        b_range:    brightness range -> output in [-b_range, b_range]
        return_legacy: if True, also fill ("lighting", s) outputs

    Outputs:
        Dict with keys ("contrast", s), ("brightness", s), and optionally ("lighting", s).
        Each tensor has shape [B, 1, H_s, W_s].
    """
    def __init__(
        self,
        num_ch_enc: List[int],
        scales=range(4),
        use_skips: bool = False,
        eps: float = 1e-3,
        c_max: float = 4.0,
        b_range: float = 0.5,
        return_legacy: bool = True,
    ):
        super().__init__()
        self.use_skips = use_skips
        self.scales = list(scales)
        self.eps = float(eps)
        self.c_max = float(c_max) if c_max is not None else None
        self.b_range = float(b_range)
        self.return_legacy = return_legacy

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # ---------- shared decoder trunk ----------
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            n_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            n_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(n_in, n_out)

            # upconv_1
            n_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                n_in += self.num_ch_enc[i - 1]
            n_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(n_in, n_out)

        # ---------- per-scale heads (raw Conv2d so we can control init) ----------
        for s in self.scales:
            self.convs[("contrast_conv", s)]   = nn.Conv2d(self.num_ch_dec[s], 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.convs[("brightness_conv", s)] = nn.Conv2d(self.num_ch_dec[s], 1, kernel_size=3, stride=1, padding=1, bias=True)

        self.decoder = nn.ModuleList(list(self.convs.values()))

        # activations
        self._softplus = nn.Softplus()   # smooth strictly-positive for contrast
        self._tanh = nn.Tanh()           # bounded brightness

        # ---------- stable, identity-like initialization ----------
        c_bias = _inv_softplus_scalar(1.0 - self.eps)  # softplus(bias) ~ 1
        with torch.no_grad():
            for s in self.scales:
                k_c = self.convs[("contrast_conv", s)]
                k_b = self.convs[("brightness_conv", s)]
                nn.init.zeros_(k_c.weight); nn.init.zeros_(k_b.weight)
                nn.init.constant_(k_c.bias, c_bias)  # contrast ~ 1 at start
                nn.init.constant_(k_b.bias, 0.0)     # brightness ~ 0 at start

    def forward(self, input_features: List[torch.Tensor]) -> Dict[Tuple[str, int], torch.Tensor]:
        outputs: Dict[Tuple[str, int], torch.Tensor] = {}

        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]  # project helper (nearest 2x)
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, dim=1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                # raw heads
                c_raw = self.convs[("contrast_conv", i)](x)
                b_raw = self.convs[("brightness_conv", i)](x)

                # stable ranges
                c = self._softplus(c_raw) + self.eps       # (eps, ∞)
                if self.c_max is not None:
                    c = torch.clamp(c, max=self.c_max)     # cap explosions
                b = self.b_range * self._tanh(b_raw)       # [-b_range, b_range]

                outputs[("contrast", i)] = c
                outputs[("brightness", i)] = b

                if self.return_legacy:
                    outputs[("lighting", i)] = torch.cat([c, b], dim=1)

        return outputs
