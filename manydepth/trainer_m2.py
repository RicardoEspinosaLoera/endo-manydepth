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


class Trainer_Monodepth:
    """
    Monodepth2-style training loop augmented with:
      - Per-pixel lighting calibration (contrast c, brightness b)
      - Illumination-invariant (II) photometric term
    Optical/appearance flow are NOT used.
    """
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        #self.parameters_to_train = []
        #self.parameters_to_train_0 = []
        self.params_pose_light = []
        self.params_depth = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # ------------------------------
        # Encoder (only needed if pose_model_type == "shared")
        # ------------------------------
        # self.models["encoder"] = networks.ResnetEncoder(
        #     self.opt.num_layers, self.opt.weights_init == "pretrained")
        # self.models["encoder"].to(self.device)
        # self.parameters_to_train += list(self.models["encoder"].parameters())

        # ------------------------------
        # Depth backbone (ENDODAC)
        # ------------------------------
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
        self.params_depth += list(filter(lambda p: p.requires_grad, self.models["depth"].parameters()))

        # ------------------------------
        # Pose + Lighting heads
        # ------------------------------
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames
                ).to(self.device)
                self.params_pose_light += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2
                ).to(self.device)
                self.params_pose_light += list(self.models["pose"].parameters())

                # Lighting decoder: predicts per-pixel contrast (c) and brightness (b)
                # Expected outputs per scale:
                #   dict[("brightness", s)]: Bx3xhxw
                #   dict[("contrast",  s)]: Bx3xhxw
                self.models["lighting"] = networks.LightingDecoder(
                    self.models["pose_encoder"].num_ch_enc, self.opt.scales
                ).to(self.device)
                self.params_pose_light += list(self.models["lighting"].parameters())


        # ---- parameter groups ----


        # # depth params
        # self.params_depth += list(filter(lambda p: p.requires_grad, self.models["depth"].parameters()))

        # # pose params
        # if self.use_pose_net:
        #     self.params_pose_light += list(self.models["pose"].parameters())
        #     if "pose_encoder" in self.models:
        #         self.params_pose_light += list(self.models["pose_encoder"].parameters())

        # # lighting params (the lighting decoder + any related heads)
        # if "lighting" in self.models:
        #     self.params_pose_light += list(self.models["lighting"].parameters())

        # # (optional) if pose_model_type=="shared", the shared encoder might also feed pose;
        # # usually keep it in depth group; if you prefer, move it to pose_light instead.
        # if self.opt.pose_model_type == "shared":
        #     self.params_depth += list(self.models["encoder"].parameters())

        # ---- two optimizers ----
        self.opt_pose = optim.AdamW(self.params_pose_light, lr=self.opt.learning_rate)
        self.opt_depth = optim.AdamW(self.params_depth, lr=self.opt.learning_rate)
        

        # ---- schedulers ----
        self.sched_pose = optim.lr_scheduler.StepLR(
            self.opt_pose, self.opt.scheduler_step_size, 0.1)
        self.sched_depth = optim.lr_scheduler.StepLR(
            self.opt_depth, self.opt.scheduler_step_size, 0.1)

        #self.sched_depth = optim.lr_scheduler.ExponentialLR(self.opt_depth,0.9)
        #self.sched_pose = optim.lr_scheduler.StepLR(self.opt_pose, self.opt.scheduler_step_size, 0.1)
        #self.sched_depth = optim.lr_scheduler.StepLR(self.opt_depth, self.opt.scheduler_step_size, 0.1)
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
        g.manual_seed(getattr(self.opt, "seed", 42))

        
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)


        # ------------------------------
        # Loss helpers
        # ------------------------------
        if not self.opt.no_ssim:
            self.ssim = SSIM().to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w).to(self.device)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w).to(self.device)

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print(f"There are {len(train_dataset)} training items and {len(val_dataset)} validation items\n")

        self.save_opts()

    # ------------------------------
    # Modes
    # ------------------------------
    def set_train_pose_light(self):
        """Enable grads for pose+lighting; freeze depth."""

        for param in self.models["pose_encoder"].parameters():
            param.requires_grad = True
        for param in self.models["pose"].parameters():
            param.requires_grad = True
        for param in self.models["lighting"].parameters():
                param.requires_grad = True

        for param in self.models["depth"].parameters():
            param.requires_grad = False

            
        self.models["pose_encoder"].train()
        self.models["pose"].train()
        self.models["depth"].eval()
        

    def set_train_depth(self):
        """Enable grads for depth (and shared encoder); freeze pose+lighting."""
        for param in self.models["pose_encoder"].parameters():
            param.requires_grad = False
        for param in self.models["pose"].parameters():
            param.requires_grad = False
        for param in self.models["lighting"].parameters():
                param.requires_grad = False

        for name, param in self.models["depth"].named_parameters():
            if "seed_" not in name:
                param.requires_grad = True
        if self.step < self.opt.warm_up_step:
            warm_up = True
        else:
            warm_up = False
        endodac.mark_only_part_as_trainable(self.models["depth"], warm_up=warm_up)

        self.models["pose_encoder"].eval()
        self.models["pose"].eval()

        self.models["depth"].train()

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
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        print("Training", self.epoch)

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            # -------- Phase A: pose + lighting step --------
            self.set_train_pose_light()
            # forward (depth is frozen but still used for warping)
            outputs_A, losses_A = self.process_batch_0(inputs)
            self.opt_pose.zero_grad(set_to_none=True)
            losses_A["loss"].backward()
            self.opt_pose.step()

            # -------- Phase B: depth step --------
            self.set_train_depth()
            # re-forward to refresh graph after pose/light changed
            outputs_B, losses_B = self.process_batch_0(inputs)
            self.opt_depth.zero_grad(set_to_none=True)
            losses_B["loss"].backward()
            self.opt_depth.step()

            # pick what to log (depth phase)
            duration = time.time() - before_op_time
            losses = losses_B
            outputs = outputs_B

            early_phase = (batch_idx % self.opt.log_frequency == 0) and (self.step < 2000)
            late_phase = (self.step % max(1, self.opt.log_frequency * 5) == 0)
            if early_phase or late_phase:
                self.log_time(batch_idx, duration, float(losses["loss"].detach().cpu()))
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.sched_pose.step()
        self.sched_depth.step()

    # ------------------------------
    # Forward passes
    # ------------------------------
    def process_batch_0(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = self.models["depth"](inputs["color_aug", 0, 0])
        outputs.update(self.predict_poses_0(inputs))
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses_0(inputs, outputs)

        return outputs, losses
    
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        outputs = self.models["depth"](inputs["color_aug", 0, 0])

        if self.use_pose_net:
            outputs.update(self.predict_poses_0(inputs, outputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        
        return outputs

    def predict_poses_0(self, inputs, shared_features=None):
        """Predict relative poses and per-pixel lighting fields (contrast c, brightness b)."""
        outputs = {}
        if self.num_pose_frames == 2:
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: shared_features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i == "s":
                    continue

                pose_pair = [pose_feats[f_i], pose_feats[0]]
                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_pair, 1))]
                elif self.opt.pose_model_type == "posecnn":
                    pose_inputs = torch.cat(pose_pair, 1)
                else:  # shared
                    pose_inputs = [pose_pair]

                axisangle, translation = self.models["pose"](pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0]
                )
                
                # Lighting decoder (only defined for separate_resnet here)
                if self.opt.pose_model_type == "separate_resnet":
                    lighting_dict = self.models["lighting"](pose_inputs[0])
                    for scale in self.opt.scales:
                        outputs[(f"b_{scale}", f_i)] = lighting_dict[("brightness", scale)]  # Bx3xhxw
                        outputs[(f"c_{scale}", f_i)] = lighting_dict[("contrast", scale)]   # Bx3xhxw

            # Upsample lighting maps to full res (store at keys ("bh",scale,frame_id)/("ch",...))
            if self.opt.pose_model_type == "separate_resnet":
                for f_i in self.opt.frame_ids[1:]:
                    if f_i == "s":
                        continue
                    for scale in self.opt.scales:
                        outputs[("bh", scale, f_i)] = F.interpolate(
                            outputs[(f"b_{scale}", f_i)], [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=True
                        )
                        outputs[("ch", scale, f_i)] = F.interpolate(
                            outputs[(f"c_{scale}", f_i)], [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=True
                        )

        return outputs

    def set_train(self):
        """Put all modules in train() mode without touching requires_grad flags.
        The two-phase loop (set_train_pose_light / set_train_depth) controls grads."""
        for m in self.models.values():
            m.train()

    def val(self):
        """Validate on a single minibatch."""
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

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
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
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
                
                # Apply lighting calibration if available: refined = c * warped + b (clamped)
                if (("ch", scale, frame_id) in outputs) and (("bh", scale, frame_id) in outputs):
                    refined = outputs[("ch", scale, frame_id)] * warped + outputs[("bh", scale, frame_id)]
                    outputs[("color_refined", frame_id, scale)] = outputs[("ch",scale, frame_id)] * outputs[("color", frame_id, scale)] + outputs[("bh", scale, frame_id)]
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

    def get_ilumination_invariant_loss(self, pred, target):
        features_p = get_ilumination_invariant_features(pred)
        features_t = get_ilumination_invariant_features(target)
        ssim_loss = self.ssim(features_p, features_t).mean(1, True)
 
        return ssim_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss, _target_unused):
        """Auto-masking as in Monodepth2."""
        if identity_reprojection_loss is None:
            return torch.ones_like(reprojection_loss)
        all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
        idxs = torch.argmin(all_losses, dim=1, keepdim=True)
        mask = (idxs == 0).float()
        return mask

    def compute_losses_0(self, inputs, outputs):
        """
        Total loss per scale:
          L = Lphoto_cal (on color_refined)
            + λ_II * L_II (on II features, refined vs target; mask via get_feature_oclution_mask)
        """
        losses = {}
        total_loss = 0.0

        # weights
        w_ii = getattr(self.opt, "illumination_invariant", 0.15)  # lambda for II term
        #w_ds = self.opt.disparity_smoothness

        for scale in self.opt.scales:
            loss_reprojection = 0.0
            loss_ilumination_invariant = 0.0
            loss = 0.0

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            #disp = outputs[("disp", scale)]
            color_t = inputs[("color", 0, 0)]  # target full-res

            # per-source losses
            valid_sources = 0
            for frame_id in self.opt.frame_ids[1:]:
                if frame_id == "s":
                    continue
                valid_sources += 1

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

                # (b) Illumination-invariant loss (use refined vs target; mask is feature-occlusion)
                loss_ilumination_invariant += (self.get_ilumination_invariant_loss(pred_cal, target) * reprojection_mask_iil).sum() / (reprojection_mask_iil.sum() + 1e-7)

            # average across sources
            denom = max(1, valid_sources)

            # accumulate
            loss += (loss_reprojection / denom)
            loss += w_ii * (loss_ilumination_invariant / denom)

            total_loss += loss
            losses[f"loss/{scale}"] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_losses(self, inputs, outputs):
        """
        Total loss per scale:
          L = Lphoto_cal (on color_refined)
            + λ_II * L_II (on II features, refined vs target; mask via get_feature_oclution_mask)
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
            valid_sources = 0
            for frame_id in self.opt.frame_ids[1:]:
                if frame_id == "s":
                    continue
                valid_sources += 1

                # auto-mask using standard (unrefined) reprojection
                target = color_t
                pred_warp = outputs[("color", frame_id, scale)]
                rep = self.compute_reprojection_loss(pred_warp, target)

                pred_ident = inputs[("color", frame_id, source_scale)]
                rep_identity = self.compute_reprojection_loss(pred_ident, target)

                reprojection_mask = self.compute_loss_masks(rep, rep_identity, target)  # Bx1xHxW
                reprojection_mask_iil = get_feature_oclution_mask(reprojection_mask)   # from utils

                # (a) Calibrated photometric loss (refined)
                #pred_cal = outputs[("color_refined", frame_id, scale)]
                #pred_cal = outputs[("color", frame_id, scale)]
                loss_reprojection += (self.compute_reprojection_loss(pred_warp, target) * reprojection_mask).sum() / (reprojection_mask.sum() + 1e-7)

                # (b) Illumination-invariant loss (use refined vs target; mask is feature-occlusion)
                loss_ilumination_invariant += (self.get_ilumination_invariant_loss(pred_warp, target) * reprojection_mask_iil).sum() / (reprojection_mask_iil.sum() + 1e-7)

            # average across sources
            denom = max(1, valid_sources)

            # (c) Disparity smoothness
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            color_for_smooth = F.interpolate(color_t, size=disp.shape[-2:], mode="bilinear", align_corners=False)
            smooth_loss = get_smooth_loss(norm_disp, color_for_smooth)

            # accumulate
            loss += (loss_reprojection / denom)
            loss += w_ii * (loss_ilumination_invariant / denom)
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
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        #writer = self.writers[mode]
        for l, v in losses.items():
            wandb.log({mode+"{}".format(l):v},step =self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            s = 0  # log only max scale
            for frame_id in self.opt.frame_ids:

                wandb.log({ "color_{}_{}/{}".format(frame_id, s, j): wandb.Image(inputs[("color", frame_id, s)][j].data)},step=self.step)
                
                if s == 0 and frame_id != 0:
                    wandb.log({"color_pred_{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs[("color", frame_id, s)][j].data)},step=self.step)
                    wandb.log({"color_pred_refined_{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs[("color_refined", frame_id,s)][j].data)},step=self.step)
                    wandb.log({"contrast_{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs[("ch",s, frame_id)][j].data)},step=self.step)
                    wandb.log({"brightness_{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs[("bh",s, frame_id)][j].data)},step=self.step)
            disp = self.colormap(outputs[("disp", s)][j, 0])
            #wandb.log({f"{mode}/disp_inv/0": wandb.Image(vis)}, step=self.step)
            wandb.log({"disp_multi_{}/{}".format(s, j): wandb.Image(disp.transpose(1, 2, 0))},step=self.step)

    # ------------------------------
    # Save / load
    # ------------------------------
    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        os.makedirs(save_folder, exist_ok=True)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.opt_pose.state_dict(), save_path)
        torch.save(self.opt_depth.state_dict(), save_path)
        #self.opt_pose = optim.AdamW(self.params_pose_light, lr=self.opt.learning_rate)
        #self.opt_depth = optim.AdamW(self.params_depth, lr=self.opt.learning_rate)

    def load_model(self):
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
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
            normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max() + 1e-8)
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
        normal_image_np /= np.linalg.norm(normal_image_np, axis=0) + 1e-8
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
