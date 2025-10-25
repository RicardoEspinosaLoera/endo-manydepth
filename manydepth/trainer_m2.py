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

_DEPTH_COLORMAP = plt.get_cmap('plasma_r', 256)



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer_Monodepth:
    """
    Lighting is learned ONLY on top of warped images.
    Pipeline per batch:
      - depth (EndoDAC) predicts disparity
      - pose predicts T (and pose features feed the LightingDecoder)
      - warp sources with depth+pose -> color_warped
      - refine: color_refined = c * color_warped + b
      - losses compare color_refined vs target
    """
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height' must be multiple of 32"
        assert self.opt.width  % 32 == 0, "'width' must be multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        #self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
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
            image_shape=(224, 280),           # <-- back to ViT/14-safe size
            pretrained_path=self.opt.pretrained_path,
            residual_block_indexes=self.opt.residual_block_indexes,
            include_cls_token=self.opt.include_cls_token
        )

        # ResNet encoder (only used if pose_model_type == "shared")
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained"
        ).to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=self.num_pose_frames
                ).to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2
                ).to(self.device)
                self.parameters_to_train += list(self.models["pose"].parameters())

                # LightingDecoder takes pose features and predicts (c, b)
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

        # ---------------- Optimizer / Sched ----------------
        self.model_optimizer = optim.AdamW(self.parameters_to_train, lr=self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.model_optimizer, 0.9)

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

        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, worker_init_fn=seed_worker)
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

    # ---------------- Modes ----------------
    def set_train(self):
        for m in self.models.values():
            m.train()
        # Optional: LoRA warm-up support if implemented in endodac
        if getattr(endodac, "mark_only_part_as_trainable", None) is not None:
            warm_up = (self.step < getattr(self.opt, "warm_up_step", 0))
            endodac.mark_only_part_as_trainable(self.models["depth"], warm_up=warm_up)

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    # ---------------- Train / Epoch ----------------
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
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.parameters_to_train, max_norm=1.0)
            self.model_optimizer.step()

            duration = time.time() - before_op_time
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase  = self.step % 2000 == 0
            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].detach().cpu().data)
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()

    # ---------------- Full pipeline ----------------
    def process_batch(self, inputs):
        # to device
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        features = None
        if self.opt.pose_model_type == "shared":
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features  = self.models["encoder"](all_color_aug)
            all_features  = [torch.split(f, self.opt.batch_size) for f in all_features]
            features = {k: [f[i] for f in all_features] for i, k in enumerate(self.opt.frame_ids)}
            outputs = self.models["depth"](features[0])
        else:
            outputs = self.models["depth"](inputs["color_aug", 0, 0])

        if self.use_pose_net:
            if self.opt.pose_model_type == "shared":
                outputs.update(self.predict_poses(inputs, features))
            else:
                outputs.update(self.predict_poses(inputs, None))

        self.generate_images_pred(inputs, outputs)  # builds color_warped and color_refined (warped-only path)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

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

                # Lighting maps (predicted from pose features) – used only AFTER warping
                if self.opt.pose_model_type == "separate_resnet":
                    lighting_out = self.models["lighting"](pose_inputs[0])
                    for scale in self.opt.scales:
                        outputs["b_"+str(scale)+"_"+str(f_i)] = lighting_out[("brightness", scale)]
                        outputs["c_"+str(scale)+"_"+str(f_i)] = lighting_out[("contrast",  scale)]

            if self.opt.pose_model_type == "separate_resnet":
                for f_i in self.opt.frame_ids[1:]:
                    if f_i == "s":
                        continue
                    for scale in self.opt.scales:
                        outputs[("bh", scale, f_i)] = F.interpolate(
                            outputs["b_"+str(scale)+"_"+str(f_i)], [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=False)
                        outputs[("ch", scale, f_i)] = F.interpolate(
                            outputs["c_"+str(scale)+"_"+str(f_i)], [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=False)
        return outputs

    def generate_images_pred(self, inputs, outputs):
        """
        Build warped views and THEN apply lighting:
          color_refined = c * color_warped + b
        No 'direct' (unwarped) lighting anywhere.
        """
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
                    T = transformation_from_parameters(axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)
                outputs[("sample", frame_id, scale)] = pix_coords

                # Geometrically warped source
                color_warped = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)
                outputs[("color", frame_id, scale)] = color_warped

                # Lighting ONLY on the warped image
                if self.use_pose_net and self.opt.pose_model_type == "separate_resnet":
                    ch = outputs[("ch", scale, frame_id)]
                    bh = outputs[("bh", scale, frame_id)]
                    outputs[("color_refined", frame_id, scale)] = ch * color_warped + bh
                else:
                    outputs[("color_refined", frame_id, scale)] = color_warped

    # ---------------- Losses ----------------
    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            return l1_loss
        ssim_loss = self.ssim(pred, target).mean(1, True)
        return 0.85 * ssim_loss + 0.15 * l1_loss

    def ms_ssim(self, img1, img2):
        scale_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        ssim_vals = []
        for _ in range(5):
            ssim_vals.append(self.ssim(img1, img2).mean())
            img1 = F.avg_pool2d(img1, kernel_size=2)
            img2 = F.avg_pool2d(img2, kernel_size=2)
        lM = ssim_vals[0]
        contrast_and_structure = ssim_vals[:-1]
        cs_prod = 1.0
        for j, v in enumerate(contrast_and_structure):
            cs_prod *= v ** scale_weights[j]
        return (lM ** scale_weights[0]) * cs_prod * cs_prod

    def get_ms_simm_loss(self, pred, target):
        l1 = torch.abs(target - pred).mean(1, True)
        return 0.90 * self.ms_ssim(target, pred) + 0.10 * l1

    def get_ilumination_invariant_loss(self, pred, target):
        # computed on warped-refined predictions only
        features_p = get_ilumination_invariant_features(pred)
        features_t = get_ilumination_invariant_features(target)
        return self.ssim(features_p, features_t).mean(1, True)

    # def compute_losses(self, inputs, outputs):
    #     losses = {}
    #     loss_reprojection = 0.0
    #     loss_ilumination_invariant = 0.0
    #     total_loss = 0.0

    #     # optional ramp for illumination loss
    #     illum_w = float(self.opt.illumination_invariant)  # or ramp if you prefer

    #     for scale in self.opt.scales:
    #         loss = 0.0
    #         source_scale = scale if self.opt.v1_multiscale else 0

    #         disp = outputs[("disp", scale)]
    #         color = inputs[("color", 0, scale)]

    #         for frame_id in self.opt.frame_ids[1:]:
    #             if frame_id == "s":  # skip stereo for monocular photometric loss
    #                 continue

    #             target = inputs[("color", 0, 0)]
    #             pred_warped = outputs[("color", frame_id, scale)]  # warped, before lighting
    #             rep = self.compute_reprojection_loss(pred_warped, target)

    #             pred_identity = inputs[("color", frame_id, source_scale)]
    #             rep_identity = self.compute_reprojection_loss(pred_identity, target)

    #             mask = self.compute_loss_masks(rep, rep_identity, target)
    #             iil_mask = get_feature_oclution_mask(mask)

    #             # *** ONLY refined (warped + lighting) is supervised ***
    #             pred_ref = outputs[("color_refined", frame_id, scale)]
    #             loss_reprojection += (self.compute_reprojection_loss(pred_ref, target) * mask).sum() / mask.sum()

    #             loss_ilumination_invariant += (self.get_ilumination_invariant_loss(pred_ref, target) * iil_mask).sum() / iil_mask.sum()

    #         # disparity smoothness
    #         mean_disp = disp.mean(2, True).mean(3, True)
    #         norm_disp = disp / (mean_disp + 1e-7)
    #         smooth_loss = get_smooth_loss(norm_disp, color)

    #         loss += (loss_reprojection / 2.0)
    #         loss += illum_w * (loss_ilumination_invariant / 2.0)
    #         loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

    #         total_loss += loss
    #         losses[f"loss/{scale}"] = loss

    #     total_loss /= self.num_scales
    #     losses["loss"] = total_loss

    #     # For monitoring
    #     losses["loss/reprojection"] = torch.as_tensor(loss_reprojection, device=self.device)
    #     losses["loss/iil"] = torch.as_tensor(loss_ilumination_invariant, device=self.device)

    #     return losses

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0.0

        illum_w = float(self.opt.illumination_invariant)

        num_scales = len(self.opt.scales)
        # count how many non-stereo source frames you actually use
        valid_frame_ids = [fid for fid in self.opt.frame_ids[1:] if fid != "s"]
        num_src = max(1, len(valid_frame_ids))  # avoid div by 0

        # for logging (averaged later)
        log_reproj_acc = 0.0
        log_iil_acc = 0.0

        for scale in self.opt.scales:
            source_scale = scale if self.opt.v1_multiscale else 0

            # per-scale accumulators (reset each scale!)
            loss_reprojection_s = 0.0
            loss_iil_s = 0.0

            disp  = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, scale)]  # <<< use same scale

            for frame_id in valid_frame_ids:
                pred_warped = outputs[("color", frame_id, scale)]       # warped, pre-lighting
                rep = self.compute_reprojection_loss(pred_warped, target)

                pred_identity = inputs[("color", frame_id, source_scale)]
                rep_identity = self.compute_reprojection_loss(pred_identity, target)

                mask = self.compute_loss_masks(rep, rep_identity, target)
                iil_mask = get_feature_oclution_mask(mask)

                # supervise the refined image
                pred_ref = outputs[("color_refined", frame_id, scale)]
                reproj_term = (self.compute_reprojection_loss(pred_ref, target) * mask)
                iil_term    = (self.get_ilumination_invariant_loss(pred_ref, target) * iil_mask)

                # normalize by valid pixels
                loss_reprojection_s += reproj_term.sum() / (mask.sum() + 1e-8)
                loss_iil_s          += iil_term.sum() / (iil_mask.sum() + 1e-8)

            # average over the number of used source frames (no magic /2.0)
            loss_reprojection_s /= num_src
            loss_iil_s          /= num_src

            # disparity smoothness
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color) / (2 ** scale)

            # assemble per-scale loss
            loss_s = loss_reprojection_s + illum_w * loss_iil_s + self.opt.disparity_smoothness * smooth_loss

            total_loss += loss_s

            # log per-scale (optional)
            losses[f"loss/{scale}"] = loss_s

            # accumulators for overall logging (averaged later)
            log_reproj_acc += loss_reprojection_s
            log_iil_acc    += loss_iil_s

        # average across scales once
        total_loss /= num_scales
        losses["loss"] = total_loss

        # averaged monitors (more interpretable)
        device = self.device if hasattr(self, "device") else next(self.parameters()).device
        losses["loss/reprojection"] = torch.as_tensor(log_reproj_acc / num_scales, device=device)
        losses["loss/iil"]          = torch.as_tensor((log_iil_acc / num_scales) * illum_w, device=device)

        return losses


    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss, inputs):
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
            outputs, losses = self.process_batch(inputs)
            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

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
                if s == 0 and frame_id != 0:
                    wandb.log({"color_pred_{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs[("color", frame_id, s)][j].data)}, step=self.step)
                    wandb.log({"color_pred_refined_{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs[("color_refined", frame_id, s)][j].data)}, step=self.step)
                    if self.use_pose_net and self.opt.pose_model_type == "separate_resnet":
                        wandb.log({"contrast_{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs[("ch", s, frame_id)][j].data)}, step=self.step)
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
        torch.save(self.model_optimizer.state_dict(), os.path.join(save_folder, "adam.pth"))

    def load_model(self):
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        opt_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(opt_path):
            print("Loading Adam weights")
            self.model_optimizer.load_state_dict(torch.load(opt_path))
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

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
