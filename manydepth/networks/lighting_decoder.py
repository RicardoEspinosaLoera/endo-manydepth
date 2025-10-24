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

    @staticmethod
    def _make_gaussian_kernel(k, sigma=None):
        if sigma is None: sigma = 0.3*((k-1)*0.5 - 1) + 0.8
        ax = torch.arange(k) - (k-1)/2.0
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        g = torch.exp(-(xx**2 + yy**2)/(2*sigma**2))
        g = (g / g.sum()).view(1,1,k,k)
        return g

    def _gaussian_blur(self, x):
        # depth-safe separable blur
        k = self.gauss
        padding = (self.blur_ks//2,)*4
        x = F.pad(x, (self.blur_ks//2,)*4, mode="reflect")
        x = F.conv2d(x, k, groups=x.shape[1])
        x = F.conv2d(x, k.transpose(2,3), groups=x.shape[1])
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
