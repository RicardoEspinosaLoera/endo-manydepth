import torch
import torch.nn as nn
import torch.nn.functional as F

class LightingDecoder(nn.Module):
    """
    Low-capacity, low-frequency illumination head:
    outputs multiplicative gain g and additive bias b.
    """
    def __init__(self, num_ch_enc, out_scales=(0,), base_ch=64, alpha=0.10, beta=0.05, blur_ks=11):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.out_scales = out_scales
        self.blur_ks = blur_ks
        bottleneck_ch = num_ch_enc[-1]

        # tiny decoder from bottleneck only (no skips)
        self.dec = nn.Sequential(
            nn.Conv2d(bottleneck_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch//2, 2, 3, padding=1)  # 2 channels: contrast, brightness
        )
        # Predict at 1/8 resolution (you can change to /16)
        self.pred_scale = 8

        # fixed gaussian blur (depth-friendly: suppress HF)
        self.register_buffer("gauss", self._make_gaussian_kernel(self.blur_ks))

    def _make_gaussian_kernel2d(k, sigma=None, device=None, dtype=None):
        if sigma is None:
            sigma = 0.3*((k-1)*0.5 - 1) + 0.8
        ax = torch.arange(k, device=device, dtype=dtype) - (k-1)/2.0
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        g = torch.exp(-(xx**2 + yy**2) / (2*sigma**2))
        g = g / g.sum()
        # weight shape for depthwise conv: (C, 1, k, k) – we'll expand later
        return g[None, None, ...]  # [1,1,k,k]

    def _gaussian_blur(self, x, k_max=11):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        k = min(k_max, H, W)
        # ensure odd and >=3
        if k < 3:
            return x
        if k % 2 == 0:
            k -= 1
        if k < 3:
            return x

        k2d = _make_gaussian_kernel2d(k, device=x.device, dtype=x.dtype)
        k2d = k2d.expand(C, 1, k, k).contiguous()  # depthwise
        pad = k // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        x = F.conv2d(x, k2d, groups=C)  # single 2D blur is enough
        return x

    def forward(self, feats):
        B, _, H, W = feats[-1].shape
        # predict low-res illumination from deepest feature only
        lr_h, lr_w = H // self.pred_scale, W // self.pred_scale
        x = F.adaptive_avg_pool2d(feats[-1], (lr_h, lr_w))
        x = self.dec(x)                                  # [B,2,lr_h,lr_w]

        # bound magnitude
        c_hat = torch.tanh(x[:, 0:1])                   # [-1,1]
        b_hat = torch.tanh(x[:, 1:1+1])                 # [-1,1]
        g = 1.0 + self.alpha * c_hat                    # ~[1-α, 1+α]
        b = self.beta * b_hat                           # ~[-β, β]

        # upsample smoothly to input size
        g = F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False)
        b = F.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)

        # enforce low-frequency behavior
        g = self._gaussian_blur(g)
        b = self._gaussian_blur(b)

        return {"contrast": g, "brightness": b}
