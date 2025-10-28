from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import readlines
from options import MonodepthOptions
import datasets
import networks.endodac as endodac  # <— use EndoDAC

cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")
_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)
STEREO_SCALE_FACTOR = 5.4


def disp_to_depth(disp, min_depth, max_depth):
    """Monodepth2 convention: returns (depth, scaled_disp)."""
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1.0 / scaled_disp
    return depth, scaled_disp


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    vis = inputs
    if normalize:
        ma = float(vis.max()); mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d
    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)[:, :, :, 0, :3]
        if torch_transpose: vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)[:, :, :, :3]
        if torch_transpose: vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)[..., :3]
        if torch_transpose: vis = vis.transpose(2, 0, 1)
    return vis


def evaluate(opt):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150.0

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Choose mono or stereo evaluation with --eval_mono or --eval_stereo"

    # ---------- Load data ----------
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    HEIGHT, WIDTH = (getattr(opt, "height", 256), getattr(opt, "width", 320))
    #HEIGHT, WIDTH = 224, 280   # both divisible by 14
    img_ext = '.png' if getattr(opt, "png", False) else '.jpg'

    if opt.eval_split == 'endovis':
        dataset = datasets.SCAREDRAWDataset(opt.data_path, filenames, HEIGHT, WIDTH, [0], 4, is_train=False, img_ext=img_ext)
    elif opt.eval_split == 'hamlyn':
        dataset = datasets.HamlynDataset(opt.data_path, HEIGHT, WIDTH, [0], 4, is_train=False)
    elif opt.eval_split == 'c3vd':
        dataset = datasets.C3VDDataset(opt.data_path, HEIGHT, WIDTH, [0], 4, is_train=False)
        MAX_DEPTH = 100.0
    else:
        raise ValueError(f"Unknown eval_split: {opt.eval_split}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    # ---------- Load EndoDAC ----------
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), f"Cannot find {opt.load_weights_folder}"
    print("-> Loading weights from", opt.load_weights_folder)

    depther_path = os.path.join(opt.load_weights_folder, "depth.pth")
    depther_state = torch.load(depther_path, map_location="cpu")

    # Build EndoDAC with safe defaults (override via opt if present)
    depther = endodac.endodac(
        backbone_size="base",
        r=self.opt.lora_rank,
        lora_type=self.opt.lora_type,
        image_shape=(224, 280),
        pretrained_path=self.opt.pretrained_path,
        residual_block_indexes=self.opt.residual_block_indexes,
        include_cls_token=self.opt.include_cls_token,
    )
    
    model_dict = depther.state_dict()
    depther.load_state_dict({k: v for k, v in depther_state.items() if k in model_dict}, strict=False)
    depther.cuda().eval()

    # ---------- Load GT depths ----------
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, allow_pickle=True, encoding='latin1')["data"]

    if opt.eval_stereo:
        print(f"   Stereo evaluation — scaling by {STEREO_SCALE_FACTOR}")
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation — using median scaling")

    errors, ratios = [], []

    print(f"-> Computing predictions with size {WIDTH}x{HEIGHT}")
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            input_color = data[("color", 0, 0)].cuda()  # (1,3,H,W)

            # forward EndoDAC
            output = depther(input_color)
            disp = output[("disp", 0)]  # (1,1,h,w) sigmoid disparity

            # Convert to depth (Monodepth2 convention)
            pred_depth_t, _ = disp_to_depth(disp, opt.min_depth, opt.max_depth)
            pred_depth = pred_depth_t.cpu().numpy()[0, 0]

            # Resize to GT resolution
            gt_depth = gt_depths[i]
            gh, gw = gt_depth.shape[:2]
            pred_depth = cv2.resize(pred_depth, (gw, gh), interpolation=cv2.INTER_LINEAR)

            # Build robust valid mask (add circular FOV if desired)
            valid = (np.isfinite(gt_depth) & (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH) &
                     np.isfinite(pred_depth) & (pred_depth > MIN_DEPTH) & (pred_depth < MAX_DEPTH))
            if valid.sum() == 0:
                continue

            gt = gt_depth[valid]
            pred = pred_depth[valid]

            # Optional global factor (kept for parity)
            if hasattr(opt, "pred_depth_scale_factor"):
                pred = pred * float(opt.pred_depth_scale_factor)

            # Per-image scale (median-of-ratios) unless stereo eval
            if not getattr(opt, "disable_median_scaling", False):
                scale = np.median(gt / np.maximum(pred, 1e-8))
                if np.isfinite(scale):
                    pred *= scale
                    ratios.append(scale)

            pred = np.clip(pred, MIN_DEPTH, MAX_DEPTH)
            errors.append(compute_errors(gt, pred))

    # ----- Report -----
    if not getattr(opt, "disable_median_scaling", False) and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(f" Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}")

    mean_errors = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
