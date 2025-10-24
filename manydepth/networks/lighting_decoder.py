import torch
import torch.nn.functional as F

# --- helpers ---------------------------------------------------------------

def _make_gaussian_kernel2d(k, sigma=None, device=None, dtype=None):
    if sigma is None:
        sigma = 0.3*((k-1)*0.5 - 1) + 0.8
    ax = torch.arange(k, device=device, dtype=dtype) - (k-1)/2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx*xx + yy*yy) / (2*sigma*sigma))
    g = (g / g.sum()).view(1, 1, k, k)  # [1,1,k,k]
    return g

def _gaussian_blur_depthwise(x, kmax=11):
    # x: [B,C,H,W]; adaptive odd kernel <= min(H,W)
    B, C, H, W = x.shape
    k = min(kmax, H, W)
    if k < 3:    # too small -> skip
        return x
    if k % 2 == 0:
        k -= 1
    if k < 3:
        return x
    kernel = _make_gaussian_kernel2d(k, device=x.device, dtype=x.dtype)
    kernel = kernel.expand(C, 1, k, k).contiguous()     # depthwise
    pad = k // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x, kernel, groups=C)

def _upsample_bilinear(x, size=None, scale_factor=2):
    return F.interpolate(
        x, size=size, scale_factor=scale_factor,
        mode="bilinear", align_corners=False
    )

# --- in your class ---------------------------------------------------------
# add safe bounds as attributes (you can move these to __init__)
_ALPHA = 0.10   # max +/-10% multiplicative gain
_BETA  = 0.05   # max +/-0.05 additive bias
_KMAX  = 11     # max gaussian kernel

def forward(self, input_features):
    self.outputs = {}

    x = input_features[-1]
    for i in range(4, -1, -1):
        x = self.convs[("upconv", i, 0)](x)
        # use bilinear instead of nearest to avoid blocky artifacts
        x = _upsample_bilinear(x)  # replaces upsample(x)
        if self.use_skips and i > 0:
            x = torch.cat([x, input_features[i - 1]], dim=1)
        x = self.convs[("upconv", i, 1)](x)

        if i in self.scales:
            # raw 2-ch lighting prediction at scale i
            light_raw = self.convs[("lighting_conv", i)](x)  # [B,2,H,W]
            self.outputs[("lighting", i)] = light_raw

            # bound with tanh, then map to safe ranges
            c_hat = torch.tanh(light_raw[:, 0:1])            # [-1,1]
            b_hat = torch.tanh(light_raw[:, 1:2])            # [-1,1]

            # multiplicative contrast around 1.0; additive brightness around 0
            Ct = 1.0 + _ALPHA * c_hat                        # ~[1-α, 1+α]
            Bt = _BETA * b_hat                               # ~[-β, β]

            # enforce low-frequency so it can't explain geometry edges
            Ct = _gaussian_blur_depthwise(Ct, kmax=_KMAX)
            Bt = _gaussian_blur_depthwise(Bt, kmax=_KMAX)

            # store exactly as your callers expect
            self.outputs[("contrast", i)]  = Ct
            self.outputs[("brightness", i)] = Bt

    return self.outputs
