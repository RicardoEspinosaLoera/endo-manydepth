from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth  # keep your original utility
from utils import readlines
from options import MonodepthOptions
import datasets
import networks.endodac as endodac  # <-- use EndoDAC

import matplotlib.pyplot as plt

# _DEPTH_COLORMAP for visualization only
_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)

cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Stereo scale kept for compatibility, though you’re doing mono here.
STEREO_SCALE_FACTOR = 5.4


def disp_to_depth_local(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction (Monodepth-style)."""
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1.0 / scaled_disp
    return scaled_disp, depth


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


def batch_post_process_disparity(l_disp, r_disp):
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained EndoDAC model on a specified test split."""
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)
        print("-> Loading weights from {}".format(opt.load_weights_folder))

        # Files / dataset
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        HEIGHT = getattr(opt, "height", 256)
        WIDTH = getattr(opt, "width", 320)
        img_ext = '.png' if getattr(opt, "png", False) else '.jpg'

        dataset = datasets.SCAREDRAWDataset(
            opt.data_path, filenames, HEIGHT, WIDTH, [0], 4, is_train=False, img_ext=img_ext
        )
        dataloader = DataLoader(dataset, 8, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=True, drop_last=False)

        # -------- EndoDAC init & weights --------
        depther_path = os.path.join(opt.load_weights_folder, "depth.pth")
        depther_dict = torch.load(depther_path, map_location="cpu")

        # Optional args (use defaults if your MonodepthOptions doesn’t define them)
        backbone_size = getattr(opt, "backbone_size", "base")
        lora_rank = getattr(opt, "lora_rank", 0)
        lora_type = getattr(opt, "lora_type", "none")
        pretrained_path = getattr(opt, "pretrained_path", None)
        residual_block_indexes = getattr(opt, "residual_block_indexes", None)
        include_cls_token = getattr(opt, "include_cls_token", False)

        # EndoDAC expects (image_shape = (H, W)) matching training resolution
        image_shape = (getattr(opt, "endo_h", 224), getattr(opt, "endo_w", 280))

        depther = endodac.endodac(
            backbone_size=backbone_size,
            r=lora_rank,
            lora_type=lora_type,
            image_shape=image_shape,
            pretrained_path=pretrained_path,
            residual_block_indexes=residual_block_indexes,
            include_cls_token=include_cls_token
        )
        model_dict = depther.state_dict()
        depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict}, strict=False)

        depther.cuda().eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(WIDTH, HEIGHT))
        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # If you want MDv1 post-process, run two passes (orig + flipped)
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                # EndoDAC forward
                outputs = depther(input_color)
                output_disp = outputs[("disp", 0)]  # EndoDAC returns this key

                pred_disp, _ = disp_to_depth_local(output_disp, opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                # If using post-process, average the two disparities
                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps, axis=0)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))
            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        return

    # Load GT depths for SCARED/EndoVIS split
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, allow_pickle=True, encoding='latin1')["data"]

    print("-> Evaluating")
    if opt.eval_stereo:
        print("   Stereo evaluation - disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors, ratios = [], []
    MIN_DEPTH, MAX_DEPTH = 1e-3, 150

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1.0 / pred_disp

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        # Optional dataset-scale factor
        pred_depth *= getattr(opt, "pred_depth_scale_factor", 1.0)

        if not getattr(opt, "disable_median_scaling", False):
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not getattr(opt, "disable_median_scaling", False):
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    results_edit = open('results.txt', mode='a')
    results_edit.write("\n " + 'model_name: %s ' % (opt.load_weights_folder))
    results_edit.write("\n " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    results_edit.write("\n " + ("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    results_edit.close()

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
