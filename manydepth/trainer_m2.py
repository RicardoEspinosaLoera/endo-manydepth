# Copyright Niantic 2019. Patent Pending. All rights reserved.
# Monodepth2 license (non-commercial use).
from __future__ import absolute_import, division, print_function

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import time
import random
import json
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from utils import *
from layers import *
import datasets
import networks
import networks.endodac as endodac

wandb.init(project="IISfMLearner-ENDOVIS", entity="respinosa")

_DEPTH_COLORMAP = plt.get_cmap('plasma_r', 256)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer_Monodepth:
    """
    One-phase training with blocked gradients:
      - Forward depth, pose, lighting ONCE
      - Phase-A (pose+lighting) terms use disp.detach()  -> no depth grads
      - Phase-B (depth) terms use T.detach(), (c,b).detach() -> no pose/light grads
    """
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height' must be multiple of 32"
        assert self.opt.width  % 32 == 0, "'width' must be multiple of 32"

        self.models = {}
        # Two disjoint param groups (two optimizers)
        self.params_pose_light = []
        self.params_depth = []

        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.opt.no_cuda else "cpu")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # ---------------- Models ----------------
        # EndoDAC depth (LoRA-enabled). Ensure image_shape matches your input aspect.
        self.models["depth"] = endodac.endodac(
            backbone_size="base",
            r=self.opt.lora_rank,
            lora_type=self.opt.lora_type,
            image_shape=(224, 280),
            pretrained_path=self.opt.pretrained_path,
            residual_block_indexes=self.opt.residual_block_indexes,
            include_cls_token=self.opt.include_cls_token
        )
        self.params_depth += list(filter(lambda p: p.requires_grad, self.models["depth"].parameters()))

        # ResNet encoder (used if pose_model_type == "shared")
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained"
        ).to(self.device)
        self.params_pose_light += list(self.models["encoder"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=self.num_pose_frames
                ).to(self.device)
                self.params_pose_light += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2
                ).to(self.device)
                self.params_pose_light += list(self.models["pose"].parameters())

                # LightingDecoder takes pose features and predicts (c, b)
                self.models["lighting"] = networks.LightingDecoder(
                    self.models["pose_encoder"].num_ch_enc, self.opt.scales
                ).to(self.device)
                self.params_pose_light += list(self.models["lighting"].parameters())

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames
                ).to(self.device)
                self.params_pose_light += list(self.models["pose"].parameters())

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2
                ).to(self.device)
                self.params_pose_light += list(self.models["pose"].parameters())

        # ---------------- Optimizers / Schedulers ----------------
        self.opt_pose  = optim.Adam(self.params_pose_light, lr=self.opt.learning_rate)
        self.opt_depth = optim.Adam(self.params_depth,      lr=self.opt.learning_rate)

        self.sched_pose  = optim.lr_scheduler.StepLR(self.opt_pose,  self.opt.scheduler_step_size, 0.1)
        self.sched_depth = optim.lr_scheduler.StepLR(self.opt_depth, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        for name, model in self.models.items():
            self.models[name] = model.to(self.device)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and events saved to:\n  ", self.opt.log_dir)
        print("Using device:\n  ", self.device)

        # ---------------- Data ----------------
        datasets_dict = {
            "kitti": datasets.KITTIRAWDataset,
            "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
            "kitti_odom": datasets.KITTIOdomDataset,
            "endovis": datasets.SCAREDDataset,
            "RNNSLAM": datasets.SCAREDDataset,
            "colon10k": datasets.SCAREDDataset,
            "C3VD": datasets.SCAREDDataset,
        }
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames   = readlines(fpath.format("val"))
        img_ext = ".jpg"

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, worker_init_fn=seed_worker)

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)

        # IMPORTANT: validation shouldn't shuffle
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False, worker_init_fn=seed_worker)
        self.val_iter = iter(self.val_loader)

        if not self.opt.no_ssim:
            self.ssim = SSIM().to(self.device)

        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width)).to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width  // (2 ** scale)
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w).to(self.device)
            self.project_3d[scale]       = Project3D(self.opt.batch_size, h, w).to(self.device)

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(len(train_dataset), len(val_dataset)))

        self.save_opts()

        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for name, param in self.models["depth"].named_parameters():
            mulValue = np.prod(param.size())
            Total_params += mulValue
            if not param.requires_grad:
                NonTrainable_params += mulValue
            else:
                Trainable_params += mulValue

        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')
        print(f'Trainable params ratio: {100 * Trainable_params / Total_params}%')

    # ---------------- Train / Epoch ----------------
    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
        # optional final save
        self.save_model()

    def run_epoch(self):
        """
        One-phase training:
          - forward depth, pose, lighting ONCE
          - Phase-A terms use disp.detach()  (no depth grads)
          - Phase-B terms use T.detach(), (c,b).detach()  (no pose/light grads)
          - Sum losses -> single backward -> step both optimizers
        """
        print("Training", self.epoch)

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            # ---- single forward + losses ----
            outputs, losses = self.process_batch_joint(inputs)

            # ---- joint step (two opt groups) ----
            self.opt_pose.zero_grad(set_to_none=True)
            self.opt_depth.zero_grad(set_to_none=True)
            losses["loss"].backward()

            torch.nn.utils.clip_grad_norm_(self.params_pose_light, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.params_depth,      max_norm=1.0)

            self.opt_pose.step()
            self.opt_depth.step()

            # ---- logging / quick val ----
            duration = time.time() - before_op_time
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase  = self.step % 2000 == 0
            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].detach().cpu().data)
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, dict(losses))
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        # schedulers step once per epoch
        self.sched_pose.step()
        self.sched_depth.step()

    # ---------------- Single-phase forward ----------------
    def process_batch_joint(self, inputs):
        """
        Single forward:
          - DEPTH forward (trainable)
          - POSE (+ lighting) forward (trainable)
          - Build A/B streams in loss with detach() to block cross-grads
        """
        # to device
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        outputs = {}

        # ----- DEPTH forward -----
        if self.opt.pose_model_type == "shared":
            # shared encoder: features for all frames (encoder grads belong to pose/light)
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            with torch.no_grad():
                all_features = self.models["encoder"](all_color_aug)
                all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
                features = {k: [f[i] for f in all_features] for i, k in enumerate(self.opt.frame_ids)}
            depth_out = self.models["depth"](features[0])
        else:
            depth_out = self.models["depth"](inputs["color_aug", 0, 0])

        outputs.update(depth_out)  # contains ("disp", s)

        # ----- POSE (+ lighting) forward -----
        if self.use_pose_net:
            if self.opt.pose_model_type == "shared":
                outputs.update(self.predict_poses(inputs, features))
            else:
                outputs.update(self.predict_poses(inputs, None))

            # upsample lighting maps (if present)
            if self.opt.pose_model_type == "separate_resnet":
                for f_i in self.opt.frame_ids[1:]:
                    if f_i == "s":
                        continue
                    for s in self.opt.scales:
                        outputs[("bh", s, f_i)] = F.interpolate(
                            outputs["b_"+str(s)+"_"+str(f_i)],
                            [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                        outputs[("ch", s, f_i)] = F.interpolate(
                            outputs["c_"+str(s)+"_"+str(f_i)],
                            [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

        # ----- per-scale depth tensors (for metrics/vis) -----
        for s in self.opt.scales:
            disp = outputs[("disp", s)]
            if not self.opt.v1_multiscale:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, s)] = depth

        # ----- Compute joint losses -----
        losses = self._compute_joint_losses(inputs, outputs)

        return outputs, losses

    # ---------------- Joint losses (A + B) ----------------
    def _compute_joint_losses(self, inputs, outputs):
        """
        Phase-A (pose+lighting) terms:
          - warps with disp_detached and live T, live (c,b)
          -> grads to pose/lighting only
        Phase-B (depth) terms:
          - warps with live depth and T_detached, (c,b)_detached
          -> grads to depth only
        Adds disparity smoothness in Phase-B.
        """
        losses = {}
        total_loss = 0.0

        illum_w = float(self.opt.illumination_invariant)
        num_scales = len(self.opt.scales)

        # non-stereo sources only
        valid_frame_ids = [fid for fid in self.opt.frame_ids[1:] if fid != "s"]
        num_src = max(1, len(valid_frame_ids))

        # accumulators for logging
        accA_reproj = 0.0
        accA_iil    = 0.0
        accB_reproj = 0.0
        accB_iil    = 0.0

        for s in self.opt.scales:
            source_scale = s if self.opt.v1_multiscale else 0

            # ----- depth live & detached -----
            disp_live = outputs[("disp", s)]
            if not self.opt.v1_multiscale:
                disp_live = F.interpolate(disp_live, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            _, depth_live = disp_to_depth(disp_live, self.opt.min_depth, self.opt.max_depth)

            disp_det = disp_live.detach()
            _, depth_det = disp_to_depth(disp_det, self.opt.min_depth, self.opt.max_depth)

            # intrinsics
            K     = inputs[("K",     source_scale)]
            inv_K = inputs[("inv_K", source_scale)]

            # targets
            target = inputs[("color", 0, 0)]
            color0 = inputs[("color", 0, s)]  # for smoothness

            # per-scale sums
            A_reproj_s = 0.0
            A_iil_s    = 0.0
            B_reproj_s = 0.0
            B_iil_s    = 0.0

            for f_i in valid_frame_ids:
                src = inputs[("color", f_i, source_scale)]

                # SE(3) live/detached
                T_live = outputs[("cam_T_cam", 0, f_i)]
                T_det  = T_live.detach()

                # Lighting live/detached (if present)
                if self.use_pose_net and self.opt.pose_model_type == "separate_resnet":
                    c_live = outputs[("ch", s, f_i)]
                    b_live = outputs[("bh", s, f_i)]
                    c_det  = c_live.detach()
                    b_det  = b_live.detach()
                else:
                    c_live = c_det = 1.0
                    b_live = b_det = 0.0

                # -------- Phase A images (no depth grad) --------
                cam_points_A = self.backproject_depth[source_scale](depth_det, inv_K)
                pix_A = self.project_3d[source_scale](cam_points_A, K, T_live)
                color_warp_A = F.grid_sample(src, pix_A, padding_mode="border", align_corners=True)
                color_ref_A  = c_live * color_warp_A + b_live

                # -------- Phase B images (no pose/light grad) --------
                cam_points_B = self.backproject_depth[source_scale](depth_live, inv_K)
                pix_B = self.project_3d[source_scale](cam_points_B, K, T_det)
                color_warp_B = F.grid_sample(src, pix_B, padding_mode="border", align_corners=True)
                color_ref_B  = c_det * color_warp_B + b_det

                # store for visualization (B stream)
                outputs[("color", f_i, s)]         = color_warp_B
                outputs[("color_refined", f_i, s)] = color_ref_B

                # identity reprojection (automask baseline)
                rep_id = self.compute_reprojection_loss(src, target)

                # ----- Phase A losses -----
                rep_A   = self.compute_reprojection_loss(color_ref_A, target)
                mask_A  = self.compute_loss_masks(rep_A, rep_id)
                iil_mA  = get_feature_oclution_mask(mask_A)

                A_reproj_s += (rep_A * mask_A).sum() / (mask_A.sum() + 1e-8)
                A_iil_s    += (self.get_ilumination_invariant_loss(color_ref_A, target) * iil_mA).sum() / (iil_mA.sum() + 1e-8)

                # ----- Phase B losses -----
                rep_B   = self.compute_reprojection_loss(color_ref_B, target)
                mask_B  = self.compute_loss_masks(rep_B, rep_id)
                iil_mB  = get_feature_oclution_mask(mask_B)

                B_reproj_s += (rep_B * mask_B).sum() / (mask_B.sum() + 1e-8)
                B_iil_s    += (self.get_ilumination_invariant_loss(color_ref_B, target) * iil_mB).sum() / (iil_mB.sum() + 1e-8)

            # average over sources
            A_reproj_s /= num_src; A_iil_s /= num_src
            B_reproj_s /= num_src; B_iil_s /= num_src

            # disparity smoothness for Phase-B (depth only)
            mean_disp  = outputs[("disp", s)].mean(2, True).mean(3, True)
            norm_disp  = outputs[("disp", s)] / (mean_disp + 1e-7)
            smoothness = get_smooth_loss(norm_disp, color0) / (2 ** s)

            # assemble per-scale
            phaseA_s = A_reproj_s + illum_w * A_iil_s
            phaseB_s = B_reproj_s + illum_w * B_iil_s + self.opt.disparity_smoothness * smoothness

            total_loss += (phaseA_s + phaseB_s)

            # logs
            losses[f"phaseA/loss/{s}"] = phaseA_s
            losses[f"phaseB/loss/{s}"] = phaseB_s

            accA_reproj += A_reproj_s; accA_iil += A_iil_s
            accB_reproj += B_reproj_s; accB_iil += B_iil_s

        # average over scales
        total_loss /= num_scales
        losses["loss"] = total_loss
        losses["phaseA/loss/reprojection"] = torch.as_tensor(accA_reproj / num_scales, device=self.device)
        losses["phaseA/loss/iil"]          = torch.as_tensor((accA_iil / num_scales) * illum_w, device=self.device)
        losses["phaseB/loss/reprojection"] = torch.as_tensor(accB_reproj / num_scales, device=self.device)
        losses["phaseB/loss/iil"]          = torch.as_tensor((accB_iil / num_scales) * illum_w, device=self.device)

        return losses

    # ---------------- Pose / Lighting ----------------
    def predict_poses(self, inputs, features):
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
                    axisangle[:, 0], translation[:, 0])

                # Lighting maps (predicted from pose features)
                if self.opt.pose_model_type == "separate_resnet":
                    lighting_out = self.models["lighting"](pose_inputs[0])
                    for scale in self.opt.scales:
                        outputs["b_"+str(scale)+"_"+str(f_i)] = lighting_out[("brightness", scale)]
                        outputs["c_"+str(scale)+"_"+str(f_i)] = lighting_out[("contrast",  scale)]
        return outputs

    # ---------------- Loss helpers ----------------
    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            return l1_loss
        ssim_loss = self.ssim(pred, target).mean(1, True)
        return 0.85 * ssim_loss + 0.15 * l1_loss

    def get_ilumination_invariant_loss(self, pred, target):
        features_p = get_ilumination_invariant_features(pred)
        features_t = get_ilumination_invariant_features(target)
        return self.ssim(features_p, features_t).mean(1, True)

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """Automasking: keep pixels where reprojection beats identity."""
        if identity_reprojection_loss is None:
            return torch.ones_like(reprojection_loss)
        all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
        idxs = torch.argmin(all_losses, dim=1, keepdim=True)
        return (idxs == 0).float()

    # ---------------- Validation & logging ----------------
    def val(self):
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch_joint(inputs)
            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        # return to train mode
        self.set_train()

    def compute_depth_losses(self, inputs, outputs, losses):
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80).detach()

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

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def set_train(self):
        for m in self.models.values():
            m.train()

    def log_time(self, batch_idx, duration, loss):
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print("epoch {:>3} | batch {:>6} | examples/s: {:5.1f} | loss: {:.5f} | time elapsed: {} | time left: {}".format(
            self.epoch, batch_idx, samples_per_sec, loss, sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        for l, v in losses.items():
            wandb.log({mode + "{}".format(l): v}, step=self.step)

        for j in range(min(4, self.opt.batch_size)):
            s = 0
            for frame_id in self.opt.frame_ids:
                wandb.log({"color_{}_{}/{}".format(frame_id, s, j): wandb.Image(inputs[("color", frame_id, s)][j].data)}, step=self.step)
                if s == 0 and frame_id != 0 and ("color", frame_id, s) in outputs:
                    wandb.log({"color_pred_{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs[("color", frame_id, s)][j].data)}, step=self.step)
                    wandb.log({"color_pred_refined_{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs[("color_refined", frame_id, s)][j].data)}, step=self.step)
                    if self.use_pose_net and self.opt.pose_model_type == "separate_resnet":
                        if ("ch", s, frame_id) in outputs:
                            wandb.log({"contrast_{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs[("ch", s, frame_id)][j].data)}, step=self.step)
                        if ("bh", s, frame_id) in outputs:
                            wandb.log({"brightness_{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs[("bh", s, frame_id)][j].data)}, step=self.step)
            disp = self.colormap(outputs[("disp", s)][j, 0])
            wandb.log({"disp_multi_{}/{}".format(s, j): wandb.Image(disp.transpose(1, 2, 0))}, step=self.step)

    # ---------------- Save / Load ----------------
    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(self.opt.__dict__.copy(), f, indent=2)

    def save_model(self):
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        os.makedirs(save_folder, exist_ok=True)
        for model_name, model in self.models.items():
            path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width']  = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, path)
        torch.save(self.opt_pose.state_dict(),  os.path.join(save_folder, "adam_pose.pth"))
        torch.save(self.opt_depth.state_dict(), os.path.join(save_folder, "adam_depth.pth"))

    def load_model(self):
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        pose_path  = os.path.join(self.opt.load_weights_folder, "adam_pose.pth")
        depth_path = os.path.join(self.opt.load_weights_folder, "adam_depth.pth")
        if os.path.isfile(pose_path):
            print("Loading Adam (pose/light)")
            self.opt_pose.load_state_dict(torch.load(pose_path, map_location=self.device))
        else:
            print("Adam pose/light randomly initialized")
        if os.path.isfile(depth_path):
            print("Loading Adam (depth)")
            self.opt_depth.load_state_dict(torch.load(depth_path, map_location=self.device))
        else:
            print("Adam depth randomly initialized")

    # ---------------- Viz helpers ----------------
    def flow2rgb(self, flow_map, max_value):
        flow_map_np = flow_map.detach().cpu().numpy()
        _, h, w = flow_map_np.shape
        flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
        rgb_map = np.ones((3, h, w)).astype(np.float32)
        normalized = flow_map_np / (max_value if max_value is not None else np.abs(flow_map_np).max())
        rgb_map[0] += normalized[0]
        rgb_map[1] -= 0.5 * (normalized[0] + normalized[1])
        rgb_map[2] += normalized[1]
        return rgb_map.clip(0, 1)

    def colormap(self, inputs, normalize=True, torch_transpose=True):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()
        vis = inputs
        if normalize:
            ma = float(vis.max()); mi = float(vis.min())
            d = ma - mi if ma != mi else 1e5
            vis = (vis - mi) / d
        if vis.ndim == 4:
            vis = vis.transpose([0, 2, 3, 1]); vis = _DEPTH_COLORMAP(vis); vis = vis[:, :, :, 0, :3]
            if torch_transpose: vis = vis.transpose(0, 3, 1, 2)
        elif vis.ndim == 3:
            vis = _DEPTH_COLORMAP(vis); vis = vis[:, :, :, :3]
            if torch_transpose: vis = vis.transpose(0, 3, 1, 2)
        elif vis.ndim == 2:
            vis = _DEPTH_COLORMAP(vis); vis = vis[..., :3]
            if torch_transpose: vis = vis.transpose(2, 0, 1)
        return vis

    def visualize_normal_image(self, xyz_image):
        n = xyz_image.cpu().numpy()
        n /= np.linalg.norm(n, axis=0)
        n = np.transpose(n, (1, 2, 0))
        return 0.5 * n + 0.5

    def visualize_normals(self, batch_normals):
        arr = batch_normals.cpu().numpy()
        arr = ((arr + 1) / 2 * 255).astype(np.uint8)
        return np.transpose(arr, (1, 2, 0))

    def norm_to_rgb(self, norm):
        pred = norm.detach().cpu().permute(1, 2, 0).numpy()
        rgb = ((pred + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return rgb
