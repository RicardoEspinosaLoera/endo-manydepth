# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
import time
import random
import json
import math
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure  # unused but kept for parity
import wandb

from utils import *
from layers import *

import datasets
import networks
import networks.endodac as endodac

wandb.init(project="IISfMLearner-ENDOVIS", entity="respinosa")

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer_Monodepth:
    """
    Monodepth2 training loop augmented with:
      - Lighting calibration head (per-pixel contrast/gain and brightness/bias)
      - Illumination-invariant (II) photometric term
    Optical flow / appearance flow are NOT used.
    """
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # ------------------------------
        # Depth encoder/decoder (ENDODAC)
        # ------------------------------
        # ResNet encoder for pose (separate) + ENDODAC depth net
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = endodac.endodac(
            backbone_size="base",
            r=self.opt.lora_rank,
            lora_type=self.opt.lora_type,
            image_shape=(224, 280),
            pretrained_path=self.opt.pretrained_path,
            residual_block_indexes=self.opt.residual_block_indexes,
            include_cls_token=self.opt.include_cls_token
        )
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(filter(lambda p: p.requires_grad, self.models["depth"].parameters()))

        # ------------------------------
        # Pose + Lighting heads
        # ------------------------------
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames
                )
                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2
                )

                # Lighting decoder: predicts per-pixel contrast (c) and brightness (b)
                # for each scale. Your networks.LightingDecoder should output:
                #   dict[("brightness", s)]: Bx3xhxw, dict[("contrast", s)]: Bx3xhxw
                self.models["lighting"] = networks.LightingDecoder(
                    self.models["pose_encoder"].num_ch_enc, self.opt.scales
                )
                self.models["lighting"].to(self.device)
                self.parameters_to_train += list(self.models["lighting"].parameters())

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames
                )
            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2
                )

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        # ------------------------------
        # Optimizer & LR scheduler
        # ------------------------------
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.model_optimizer, 0.9)

        # ------------------------------
        # Load weights (optional)
        # ------------------------------
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # ------------------------------
        # DATA
        # ------------------------------
        datasets_dict = {
            "kitti": datasets.KITTIRAWDataset,
            "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
            "kitti_odom": datasets.KITTIOdomDataset,
            "endovis": datasets.SCAREDDataset,
            "RNNSLAM": datasets.SCAREDDataset,
            "colon10k": datasets.SCAREDDataset,
            "C3VD": datasets.SCAREDDataset
        }
        self.dataset = datasets_dict[self.opt.dataset]
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        g = torch.Generator()
        g.manual_seed(self.opt.seed if hasattr(self.opt, "seed") else 42)

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext
        )
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker, generator=g
        )
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext
        )
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker, generator=g
        )
        self.val_iter = iter(self.val_loader)

        # ------------------------------
        # Loss helpers
        # ------------------------------
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
        self.spatial_transform.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w).to(self.device)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w).to(self.device)

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    # ------------------------------
    # Modes
    # ------------------------------
    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    # ------------------------------
    # Train loop
    # ------------------------------
    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            if self.epoch > 1:
                for p in self.models["lighting"].parameters():
                    p.requires_grad = False
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        print("Training", self.epoch)
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0
            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()

    # ------------------------------
    # Forward passes
    # ------------------------------
    def process_batch(self, inputs):
        """Forward: depth (ENDODAC), pose, lighting; synthesize warped views; compute losses."""
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # Depth
        if self.opt.pose_model_type == "shared":
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
            features = {k: [f[i] for f in all_features] for i, k in enumerate(self.opt.frame_ids)}
            outputs = self.models["depth"](features[0])
        else:
            features0 = self.models["encoder"](inputs["color_aug", 0, 0])  # kept only for shared API
            outputs = self.models["depth"](inputs["color_aug", 0, 0])

        # Pose + lighting fields
        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features0 if self.opt.pose_model_type != "shared" else features))

        # Geometry-based warps
        self.generate_images_pred(inputs, outputs)

        # Losses
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict relative poses and per-pixel lighting fields (contrast c, brightness b)."""
        outputs = {}
        if self.num_pose_frames == 2:
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i == "s":
                    continue

                pose_inputs = [pose_feats[f_i], pose_feats[0]]
                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                elif self.opt.pose_model_type == "posecnn":
                    pose_inputs = torch.cat(pose_inputs, 1)

                axisangle, translation = self.models["pose"](pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0]
                )

                # Lighting decoder on the same pose feature tensor (separate_resnet case)
                if self.opt.pose_model_type == "separate_resnet":
                    lighting_dict = self.models["lighting"](pose_inputs[0])
                    for scale in self.opt.scales:
                        # Store low-res maps
                        outputs[f"b_{scale}_{f_i}"] = lighting_dict[("brightness", scale)]  # Bx3xhxw
                        outputs[f"c_{scale}_{f_i}"] = lighting_dict[("contrast", scale)]   # Bx3xhxw

            # Upsample lighting maps to full res for each frame/scale
            for f_i in self.opt.frame_ids[1:]:
                if f_i == "s":
                    continue
                for scale in self.opt.scales:
                    outputs[("bh", scale, f_i)] = F.interpolate(
                        outputs[f"b_{scale}_{f_i}"], [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False
                    )
                    outputs[("ch", scale, f_i)] = F.interpolate(
                        outputs[f"c_{scale}_{f_i}"], [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False
                    )

        return outputs

    def val(self):
        """Validate on a single minibatch."""
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Geometry-based view synthesis using predicted depth + pose."""
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for _, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]
                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0
                    )

                cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords
                warped = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True
                )
                outputs[("color", frame_id, scale)] = warped

                # Apply lighting calibration: refined = c * warped + b (clamped)
                if ("ch", scale, frame_id) in outputs and ("bh", scale, frame_id) in outputs:
                    refined = outputs[("ch", scale, frame_id)] * warped + outputs[("bh", scale, frame_id)]
                    outputs[("color_refined", frame_id, scale)] = torch.clamp(refined, 0.0, 1.0)
                else:
                    outputs[("color_refined", frame_id, scale)] = warped  # fallback

    # ------------------------------
    # Losses
    # ------------------------------
    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

    def ms_ssim(self, img1, img2):
        # Simple pyramidal SSIM wrapper using single-scale SSIM at each level
        scale_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        ssim_vals = []
        for _ in range(5):
            ssim_val = self.ssim(img1, img2).mean()
            ssim_vals.append(ssim_val)
            img1 = F.avg_pool2d(img1, kernel_size=2, stride=2, padding=0)
            img2 = F.avg_pool2d(img2, kernel_size=2, stride=2, padding=0)
        lM = ssim_vals[0]
        contrast_and_structure = ssim_vals[:-1]
        cs_prod = 1.0
        for j, s in enumerate(contrast_and_structure):
            cs_prod *= s ** scale_weights[j]
        ms_ssim_val = (lM ** scale_weights[0]) * cs_prod * cs_prod
        return ms_ssim_val

    # ---- Illumination-invariant features (robust, local) ----
    @staticmethod
    def _illumination_invariant_features(img, eps=1e-3):
        """
        Compute simple illumination-invariant representation per pixel:
            f = log(I+eps) - mean_c(log(I+eps))
        Keeps 3 channels, removing per-pixel lighting level.
        """
        x = torch.clamp(img, 0.0, 1.0)
        x = torch.log(x + eps)
        mean_c = x.mean(1, keepdim=True)
        return x - mean_c

    def get_ilumination_invariant_loss(self, pred, target):
        fp = self._illumination_invariant_features(pred)
        ft = self._illumination_invariant_features(target)
        # SSIM on II features (treat as images)
        ssim_loss = self.ssim(fp, ft).mean(1, True)
        return ssim_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss, _):
        """Auto-masking as in Monodepth2."""
        if identity_reprojection_loss is None:
            return torch.ones_like(reprojection_loss)
        all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
        idxs = torch.argmin(all_losses, dim=1, keepdim=True)
        mask = (idxs == 0).float()
        return mask

    def compute_losses(self, inputs, outputs):
        """
        Total loss per scale:
          L = Lphoto_cal (on color_refined)
            + λ_II * L_II (on II features, unrefined vs target)
            + λ_ds * disp_smooth
        """
        losses = {}
        total_loss = 0.0

        # weights
        w_ii = getattr(self.opt, "illumination_invariant", 0.15)  # lambda for II term
        w_ds = self.opt.disparity_smoothness

        for scale in self.opt.scales:
            loss_reprojection = 0.0
            loss_ilumination_invariant = 0.0
            loss = 0.0

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color_t = inputs[("color", 0, 0)]  # target full-res

            # per-source losses
            for frame_id in self.opt.frame_ids[1:]:
                if frame_id == "s":
                    continue

                # auto-mask using standard (unrefined) reprojection
                target = color_t
                pred_warp = outputs[("color", frame_id, scale)]
                rep = self.compute_reprojection_loss(pred_warp, target)

                pred_ident = inputs[("color", frame_id, source_scale)]
                rep_identity = self.compute_reprojection_loss(pred_ident, target)

                reprojection_mask = self.compute_loss_masks(rep, rep_identity, target)  # Bx1xHxW
                reprojection_mask_iil = get_feature_oclution_mask(reprojection_mask)   # from utils

                # (a) Calibrated photometric loss (refined)
                pred_cal = outputs[("color_refined", frame_id, scale)]
                loss_reprojection += (self.compute_reprojection_loss(pred_cal, target) * reprojection_mask).sum() / (reprojection_mask.sum() + 1e-7)

                # (b) Illumination-invariant loss (no calibration)
                loss_ilumination_invariant += (self.get_ilumination_invariant_loss(pred_cal, target) * reprojection_mask_iil).sum() / (reprojection_mask_iil.sum() + 1e-7)

            # (c) Disparity smoothness
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            color_for_smooth = F.interpolate(color_t, size=disp.shape[-2:], mode="bilinear", align_corners=False)
            smooth_loss = get_smooth_loss(norm_disp, color_for_smooth)

            # accumulate
            loss += (loss_reprojection / max(1, len(self.opt.frame_ids) - 1))
            loss += w_ii * (loss_ilumination_invariant / max(1, len(self.opt.frame_ids) - 1))
            loss += w_ds * smooth_loss / (2 ** scale)

            total_loss += loss
            losses[f"loss/{scale}"] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    # ------------------------------
    # Metrics & logging
    # ------------------------------
    def compute_depth_losses(self, inputs, outputs, losses):
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)
        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        for l, v in losses.items():
            wandb.log({mode + "{}".format(l): v}, step=self.step)

        for j in range(min(4, self.opt.batch_size)):  # max 4 images
            s = 0  # log only max scale
            for frame_id in self.opt.frame_ids:
                wandb.log({f"color_{frame_id}_{s}/{j}": wandb.Image(inputs[("color", frame_id, s)][j].data)}, step=self.step)

                if s == 0 and frame_id != 0:
                    wandb.log({f"color_pred_{frame_id}_{s}/{j}": wandb.Image(outputs[("color", frame_id, s)][j].data)}, step=self.step)
                    wandb.log({f"color_pred_refined_{frame_id}_{s}/{j}": wandb.Image(outputs[("color_refined", frame_id, s)][j].data)}, step=self.step)
                    if ("ch", s, frame_id) in outputs:
                        wandb.log({f"contrast_{frame_id}_{s}/{j}": wandb.Image(outputs[("ch", s, frame_id)][j].data)}, step=self.step)
                    if ("bh", s, frame_id) in outputs:
                        wandb.log({f"brightness_{frame_id}_{s}/{j}": wandb.Image(outputs[("bh", s, frame_id)][j].data)}, step=self.step)

            disp = self.colormap(outputs[("disp", s)][j, 0])
            wandb.log({f"disp_multi_{s}/{j}": wandb.Image(disp.transpose(1, 2, 0))}, step=self.step)

    # ------------------------------
    # Save / load
    # ------------------------------
    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    # ------------------------------
    # Viz helpers
    # ------------------------------
    def flow2rgb(self, flow_map, max_value):
        flow_map_np = flow_map.detach().cpu().numpy()
        _, h, w = flow_map_np.shape
        flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
        rgb_map = np.ones((3, h, w)).astype(np.float32)
        if max_value is not None:
            normalized_flow_map = flow_map_np / max_value
        else:
            normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
        rgb_map[0] += normalized_flow_map[0]
        rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
        rgb_map[2] += normalized_flow_map[1]
        return rgb_map.clip(0, 1)

    def colormap(self, inputs, normalize=True, torch_transpose=True):
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

    def visualize_normal_image(self, xyz_image):
        normal_image_np = xyz_image.cpu().numpy()
        normal_image_np /= np.linalg.norm(normal_image_np, axis=0)
        normal_image_np = np.transpose(normal_image_np, (1, 2, 0))
        normal_image_np = 0.5 * normal_image_np + 0.5
        return normal_image_np

    def visualize_normals(self, batch_normals):
        batch_normals = batch_normals.cpu().numpy()
        scaled_normals = ((batch_normals + 1) / 2 * 255).astype(np.uint8)
        transposed_normals = np.transpose(scaled_normals, (1, 2, 0))
        return transposed_normals

    def norm_to_rgb(self, norm):
        pred_norm = norm.detach().cpu().permute(1, 2, 0).numpy()
        norm_rgb = ((pred_norm + 1) / 2) * 255
        norm_rgb = np.clip(norm_rgb, a_min=0, a_max=255).astype(np.uint8)
        return norm_rgb
