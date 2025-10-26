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
        # Encoder (only needed if pose_model_type == "shared")
        # ------------------------------
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

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
                ).to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2
                ).to(self.device)
                self.parameters_to_train += list(self.models["pose"].parameters())

                # Lighting decoder: predicts per-pixel contrast (c) and brightness (b)
                # Expected outputs per scale:
                #   dict[("brightness", s)]: Bx3xhxw
                #   dict[("contrast",  s)]: Bx3xhxw
                self.models["lighting"] = networks.LightingDecoder(
                    self.models["pose_encoder"].num_ch_enc, self.opt.scales
                ).to(self.device)
                self.parameters_to_train += list(self.models["lighting"].parameters())

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames
                ).to(self.device)
                self.parameters_to_train += list(self.models["pose"].parameters())

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2
                ).to(self.device)
                self.parameters_to_train += list(self.models["pose"].parameters())

        # ---- parameter groups ----
        self.params_pose_light = []
        self.params_depth = []

        # depth params
        self.params_depth += list(filter(lambda p: p.requires_grad, self.models["depth"].parameters()))

        # pose params
        if self.use_pose_net:
            self.params_pose_light += list(self.models["pose"].parameters())
            if "pose_encoder" in self.models:
                self.params_pose_light += list(self.models["pose_encoder"].parameters())

        # lighting params (the lighting decoder + any related heads)
        if "lighting" in self.models:
            self.params_pose_light += list(self.models["lighting"].parameters())

        # (optional) if pose_model_type=="shared", the shared encoder might also feed pose;
        # usually keep it in depth group; if you prefer, move it to pose_light instead.
        if self.opt.pose_model_type == "shared":
            self.params_depth += list(self.models["encoder"].parameters())

        # ---- two optimizers ----
        self.opt_pose = optim.AdamW(self.params_pose_light, lr=self.opt.learning_rate)
        self.opt_depth = optim.AdamW(self.params_depth, lr=self.opt.learning_rate)

        # ---- schedulers ----
        self.sched_pose = optim.lr_scheduler.ExponentialLR(self.opt_pose, gamma=0.9)
        self.sched_depth = optim.lr_scheduler.ExponentialLR(self.opt_depth, gamma=0.9)

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
            val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False
        )
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
        for m in self.models.values():
            m.train()
        # freeze depth
        for p in self.models["depth"].parameters():
            p.requires_grad = False
        # shared encoder (if any) stays with depth phase by default
        if self.opt.pose_model_type == "shared":
            for p in self.models["encoder"].parameters():
                p.requires_grad = False
        # enable pose + lighting
        if self.use_pose_net:
            for p in self.models["pose"].parameters():
                p.requires_grad = True
            if "pose_encoder" in self.models:
                for p in self.models["pose_encoder"].parameters():
                    p.requires_grad = True
        if "lighting" in self.models:
            for p in self.models["lighting"].parameters():
                p.requires_grad = True

    def set_train_depth(self):
        """Enable grads for depth (and shared encoder); freeze pose+lighting."""
        for m in self.models.values():
            m.train()
        # enable depth
        for p in self.models["depth"].parameters():
            p.requires_grad = True
        if self.opt.pose_model_type == "shared":
            for p in self.models["encoder"].parameters():
                p.requires_grad = True
        # freeze pose + lighting
        if self.use_pose_net:
            for p in self.models["pose"].parameters():
                p.requires_grad = False
            if "pose_encoder" in self.models:
                for p in self.models["pose_encoder"].parameters():
                    p.requires_grad = False
        if "lighting" in self.models:
            for p in self.models["lighting"].parameters():
                p.requires_grad = False

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
            outputs_A, losses_A = self.process_batch(inputs)
            self.opt_pose.zero_grad(set_to_none=True)
            losses_A["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.params_pose_light, max_norm=1.0)
            self.opt_pose.step()

            # -------- Phase B: depth step --------
            self.set_train_depth()
            # re-forward to refresh graph after pose/light changed
            outputs_B, losses_B = self.process_batch(inputs)
            self.opt_depth.zero_grad(set_to_none=True)
            losses_B["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.params_depth, max_norm=1.0)
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
    def process_batch(self, inputs):
        """Forward: depth (ENDODAC), pose, lighting; synthesize warped views; compute losses."""
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # Depth
        if self.opt.pose_model_type == "shared":
            # Build per-frame features if you need them for shared pose
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids], dim=0)
            all_features = self.models["encoder"](all_color_aug)
            # split features per frame id
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
            features = {i: [lvl[k] for lvl in all_features] for k, i in enumerate(self.opt.frame_ids)}
        outputs = self.models["depth"](inputs["color_aug", 0, 0])

        # Pose + lighting fields
        if self.use_pose_net:
            outputs.update(
                self.predict_poses(
                    inputs,
                    features if self.opt.pose_model_type == "shared" else None
                )
            )

        # Geometry-based warps
        self.generate_images_pred(inputs, outputs)

        # Losses
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def predict_poses(self, inputs, shared_features=None):
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
                """
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
                            mode="bilinear", align_corners=False
                        )
                        outputs[("ch", scale, f_i)] = F.interpolate(
                            outputs[(f"c_{scale}", f_i)], [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=False
                        )"""

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
                """
                # Apply lighting calibration if available: refined = c * warped + b (clamped)
                if (("ch", scale, frame_id) in outputs) and (("bh", scale, frame_id) in outputs):
                    refined = outputs[("ch", scale, frame_id)] * warped + outputs[("bh", scale, frame_id)]
                    outputs[("color_refined", frame_id, scale)] = torch.clamp(refined, 0.0, 1.0)
                else:
                    outputs[("color_refined", frame_id, scale)] = warped  # fallback"""

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

    @staticmethod
    def _illumination_invariant_features(img, eps=1e-3):
        """
        Illumination-invariant representation per pixel:
            f = log(I+eps) - mean_c(log(I+eps))
        Keeps 3 channels, removing per-pixel lighting level.
        """
        x = torch.clamp(img, 0.0, 1.0)
        x = torch.log(x + eps)
        mean_c = x.mean(1, keepdim=True)
        return x - mean_c

    def get_ilumination_invariant_loss(self, pred, target):
        """SSIM on II features (treated as images)."""
        fp = self._illumination_invariant_features(pred)
        ft = self._illumination_invariant_features(target)
        # Use same mixing as photometric if desired; here we follow your provided version:
        return self.ssim(fp, ft).mean(1, True)

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss, _target_unused):
        """Auto-masking as in Monodepth2."""
        if identity_reprojection_loss is None:
            return torch.ones_like(reprojection_loss)
        all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
        idxs = torch.argmin(all_losses, dim=1, keepdim=True)
        mask = (idxs == 0).float()
        return mask

    def compute_losses(self, inputs, outputs):
        """
        Loss per scale:
        L = Lphoto (warped vs target, with delayed identity masking)
            + w_ii(t) * L_II  (log-chromaticity)
            + w_ds * L_smooth
            + w_dc * L_disp_center  (anti-collapse prior on mean disparity)
        """
        losses, total_loss = {}, 0.0

        # ---- schedules / weights ----
        w_ii_base = getattr(self.opt, "illumination_invariant", 0.2)
        w_ds = self.opt.disparity_smoothness
        w_dc = getattr(self.opt, "disp_center_weight", 0.01)   # NEW: small anti-collapse prior
        # II warm-up (prevents early saturation)
        ii_warmup_steps = getattr(self.opt, "ii_warmup_steps", 4000)
        w_ii = w_ii_base * min(1.0, float(self.step) / max(1, ii_warmup_steps))

        # Delay identity auto-masking a bit so depth must learn to warp first
        use_identity = (self.step >= getattr(self.opt, "identity_mask_start", 1500))

        for scale in self.opt.scales:
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            target = inputs[("color", 0, 0)]
            loss_photo, loss_ii, loss = 0.0, 0.0, 0.0

            # ---- photo + II over sources ----
            n_src = 0
            for frame_id in self.opt.frame_ids[1:]:
                if frame_id == "s":
                    continue
                n_src += 1

                pred_warp = outputs[("color", frame_id, scale)]
                rep = self.compute_reprojection_loss(pred_warp, target)

                if use_identity:
                    pred_ident = inputs[("color", frame_id, source_scale)]
                    rep_identity = self.compute_reprojection_loss(pred_ident, target)
                    reprojection_mask = self.compute_loss_masks(rep, rep_identity, target)  # Bx1xHxW
                else:
                    # hard enable learning early on
                    reprojection_mask = torch.ones_like(rep)

                # a) photometric
                loss_photo += (rep * reprojection_mask).sum() / (reprojection_mask.sum() + 1e-7)

                # b) illumination invariant (log-chromaticity)
                ii_mask = get_feature_oclution_mask(reprojection_mask)  # your util
                loss_ii += (self.get_ilumination_invariant_loss(pred_warp, target) * ii_mask).sum() / (ii_mask.sum() + 1e-7)

            denom = max(1, n_src)
            loss += loss_photo / denom
            loss += w_ii * (loss_ii / denom)

            # ---- disparity smoothness (edge-aware) ----
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            tgt_for_smooth = F.interpolate(target, size=disp.shape[-2:], mode="bilinear", align_corners=False)
            loss_smooth = get_smooth_loss(norm_disp, tgt_for_smooth)
            loss += w_ds * loss_smooth / (2 ** scale)

            # ---- NEW: anti-collapse prior on disparity mean ----
            # Encourage a small but non-zero mean disparity. Choose a weak target μ0.
            # For monocular indoor/endoscopic, μ0 around 0.08–0.15 works as a gentle anchor.
            mu0 = getattr(self.opt, "disp_center_mu", 0.1)
            disp_mean = disp.mean()
            loss_disp_center = torch.abs(disp_mean - mu0)
            loss += w_dc * loss_disp_center

            total_loss += loss
            losses[f"loss/{scale}"] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        # ---- debug prints every 200 steps ----
        if self.step % 200 == 0:
            d0 = outputs[("disp", 0)]
            print(f"[{self.step}] w_ii={w_ii:.4f}  use_identity={use_identity}  "
                f"disp[min,max,mean]=({float(d0.min()):.4f}, {float(d0.max()):.4f}, {float(d0.mean()):.4f})")

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

    def log_time(self, batch_idx, duration, loss_scalar):
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / max(1, self.step) - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f} | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss_scalar,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        # Scalars
        log_dict = {}
        for l, v in losses.items():
            if isinstance(v, torch.Tensor):
                log_dict[mode + f"/{l}"] = float(v.detach().cpu())
            else:
                log_dict[mode + f"/{l}"] = v
        wandb.log(log_dict, step=self.step)

        # Images (log only scale 0 to keep bandwidth low)
        s = 0
        max_imgs = min(4, self.opt.batch_size)
        for j in range(max_imgs):
            # input frames
            for frame_id in self.opt.frame_ids:
                wandb.log({f"{mode}/color_{frame_id}_{s}/{j}": wandb.Image(inputs[("color", frame_id, s)][j].data)}, step=self.step)

            # predictions for source frames
            for frame_id in self.opt.frame_ids[1:]:
                if (frame_id, ) == ("s", ):
                    continue
                if ("color", frame_id, s) in outputs:
                    wandb.log({f"{mode}/color_pred_{frame_id}_{s}/{j}": wandb.Image(outputs[("color", frame_id, s)][j].data)}, step=self.step)
                if ("color_refined", frame_id, s) in outputs:
                    wandb.log({f"{mode}/color_pred_refined_{frame_id}_{s}/{j}": wandb.Image(outputs[("color_refined", frame_id, s)][j].data)}, step=self.step)
                if ("ch", s, frame_id) in outputs:
                    wandb.log({f"{mode}/contrast_{frame_id}_{s}/{j}": wandb.Image(outputs[("ch", s, frame_id)][j].data)}, step=self.step)
                if ("bh", s, frame_id) in outputs:
                    wandb.log({f"{mode}/brightness_{frame_id}_{s}/{j}": wandb.Image(outputs[("bh", s, frame_id)][j].data)}, step=self.step)

            # disparity colormap
            if ("disp", s) in outputs:
                disp = self.colormap(outputs[("disp", s)][j, 0])
                wandb.log({f"{mode}/disp_multi_{s}/{j}": wandb.Image(disp.transpose(1, 2, 0))}, step=self.step)

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
