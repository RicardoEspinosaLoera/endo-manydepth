import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *  # ConvBlock, Conv3x3, upsample

def inv_softplus(y):  # y>0
    return torch.log(torch.exp(torch.tensor(y)) - 1.0)

class LightingDecoder(nn.Module):
    """
    Two-head lighting decoder with stable ranges:
      contrast  c ∈ (0, c_max], via Softplus + eps, init ≈ 1
      brightness b ∈ [-b_range, b_range], via Tanh, init = 0
    """
    def __init__(self, num_ch_enc, scales=range(4), use_skips=False,
                 eps=1e-3, c_max=4.0, b_range=0.5, return_legacy=True):
        super().__init__()
        self.use_skips = use_skips
        self.scales = list(scales)
        self.eps = eps
        self.c_max = c_max
        self.b_range = b_range
        self.return_legacy = return_legacy

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # shared trunk
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            n_in  = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i+1]
            n_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(n_in, n_out)

            n_in = self.num_ch_dec[i] + (self.num_ch_enc[i-1] if self.use_skips and i > 0 else 0)
            n_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(n_in, n_out)

        for s in self.scales:
            self.convs[("contrast_conv", s)]   = Conv3x3(self.num_ch_dec[s], 1)
            self.convs[("brightness_conv", s)] = Conv3x3(self.num_ch_dec[s], 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.softplus = nn.Softplus()  # smooth, strictly positive
        self.tanh = nn.Tanh()

        # -------- stable initialization --------
        with torch.no_grad():
            for s in self.scales:
                # start near identity: contrast ≈ 1, brightness = 0
                k_c = self.convs[("contrast_conv", s)]
                k_b = self.convs[("brightness_conv", s)]
                nn.init.zeros_(k_c.weight); nn.init.zeros_(k_b.weight)
                nn.init.constant_(k_c.bias, float(inv_softplus(1.0 - self.eps)))  # softplus(bias) ≈ 1
                nn.init.constant_(k_b.bias, 0.0)

    def forward(self, input_features):
        outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                c_raw = self.convs[("contrast_conv", i)](x)
                b_raw = self.convs[("brightness_conv", i)](x)

                c = self.softplus(c_raw) + self.eps        # (eps, ∞)
                c = torch.clamp(c, max=self.c_max)         # cap to avoid explosions
                b = self.b_range * self.tanh(b_raw)        # [-b_range, b_range]

                outputs[("contrast", i)] = c
                outputs[("brightness", i)] = b
                if self.return_legacy:
                    outputs[("lighting", i)] = torch.cat([c, b], dim=1)  # for old code paths

        return outputs
