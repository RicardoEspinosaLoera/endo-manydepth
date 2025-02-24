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

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
#from tensorboardX import SummaryWriter
import wandb
import math

wandb.init(project="IISfMLearner-ENDOVIS", entity="respinosa")

import json

from utils import *
from layers import *

import datasets
import networks
import matplotlib.pyplot as plt


_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting
#_DEPTH_COLORMAP = plt.get_cmap('viridis', 256)  # for plotting

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer_Monodepth:
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
        
        
        #Transformer
        """
        self.models["encoder"] = networks.mpvit_small()            
        self.models["encoder"].num_ch_enc = [64,64,128,216,288]
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())"""
        #Resnet18 Encoder
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters()) 
        #Resnet18 Encoder
        
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        
        #Transformer"
        """
        self.models["depth"] = networks.DepthDecoderT()
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        """
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc,num_input_features=1,num_frames_to_predict_for=2)
                
                
                self.models["lighting"] = networks.LightingDecoder(self.models["pose_encoder"].num_ch_enc, self.opt.scales)
                self.models["lighting"].to(self.device)
                self.parameters_to_train += list(self.models["lighting"].parameters())
                

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())


        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate)
        """self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)"""
        self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(
		self.model_optimizer,0.9)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # DATA
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "endovis": datasets.SCAREDDataset,
                         "RNNSLAM": datasets.SCAREDDataset,
                         "colon10k": datasets.SCAREDDataset,
                         "C3VD": datasets.SCAREDDataset}

        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

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


        if not self.opt.no_ssim:
            self.ssim = SSIM()
            #self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
            self.ssim.to(self.device)
            #self.ms_ssim.to(self.device)

        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
        self.spatial_transform.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """          

        print("Training",self.epoch)
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
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

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
            
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)
        """
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)"""

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        #outputs["normal_inputs"] = self.models["normal"](features)
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    #if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    #else:
                    #pose_inputs = [pose_feats[0], pose_feats[f_i]]
                    

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)
                    
                    # Original
                    axisangle, translation = self.models["pose"](pose_inputs)

                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    """outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))"""

                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0])
                    
                    outputs_lighting = self.models["lighting"](pose_inputs[0])
                    
                    #Lighting      
                    for scale in self.opt.scales:
                        #outputs["b_"+str(scale)+"_"+str(f_i)] = outputs_lighting[("lighting", scale)][:,0,None,:, :]
                        #outputs["c_"+str(scale)+"_"+str(f_i)] = outputs_lighting[("lighting", scale)][:,1,None,:, :]
                        #print(scale)
                        outputs["b_"+str(scale)+"_"+str(f_i)] = outputs_lighting[("brightness", scale)]
                        outputs["c_"+str(scale)+"_"+str(f_i)] = outputs_lighting[("contrast", scale)] 
                        
            #Lighting
            for f_i in self.opt.frame_ids[1:]:
            #for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                for scale in self.opt.scales:
                    outputs[("bh",scale, f_i)] = F.interpolate(outputs["b_"+str(scale)+"_"+str(f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    outputs[("ch",scale, f_i)] = F.interpolate(outputs["c_"+str(scale)+"_"+str(f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    #outputs[("color_refined", f_i, scale)] = outputs[("ch",scale, f_i)] * inputs[("color", 0, 0)] + outputs[("bh", scale, f_i)]
                    #outputs[("color_refined", f_i, scale)] = outputs[("ch",scale, f_i)] * outputs[("color", f_i, scale)] + outputs[("bh", scale, f_i)]
                    


        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
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
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords
                
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",align_corners=True)

                outputs[("color_refined", frame_id, scale)] = outputs[("ch",scale, frame_id)] * outputs[("color", frame_id, scale)] + outputs[("bh", scale, frame_id)]



    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

      
    def ms_ssim(self,img1, img2):
        scale_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # Pesos por defecto

        ssim_vals = []
        
        for j in range(5):
            # Calculamos el SSIM en la escala actual
            ssim_val = self.ssim(img1, img2).mean()
            ssim_vals.append(ssim_val)

            # Aplicamos downsampling para la siguiente escala
            img1 = F.avg_pool2d(img1, kernel_size=2)
            img2 = F.avg_pool2d(img2, kernel_size=2)
        
        # Luminancia se calcula solo en la última escala (último valor de SSIM)
        lM = ssim_vals[0]
        
        # Los valores anteriores de SSIM representan cj (contraste) y sj (estructura)
        contrast_and_structure = ssim_vals[:-1]
        # Producto acumulado de los términos de contraste y estructura
        contrast_and_structure_product = 1.0
        for j, ssim in enumerate(contrast_and_structure):
            contrast_and_structure_product *= ssim ** scale_weights[j]

        # MS-SSIM final
        ms_ssim_val = (lM ** scale_weights[0]) * (contrast_and_structure_product) * (contrast_and_structure_product)

        return ms_ssim_val

    def get_ms_simm_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ms_ssim(target,pred)
        loss = 0.90 * ssim_loss + 0.10 * l1_loss

        return loss
    
    def get_ilumination_invariant_loss(self, pred, target):
        features_p = get_ilumination_invariant_features(pred)
        features_t = get_ilumination_invariant_features(target)
        ssim_loss = self.ssim(features_p, features_t).mean(1, True)
 
        return ssim_loss

    
    
    def compute_losses(self, inputs, outputs):

        losses = {}
        loss_reprojection = 0
        loss_ilumination_invariant = 0
        total_loss = 0


        for scale in self.opt.scales:
            loss = 0

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            #Losses & compute mask
            for frame_id in self.opt.frame_ids[1:]:
                # Mask
                target = inputs[("color", 0, 0)]
                pred = outputs[("color", frame_id, scale)]                

                rep = self.compute_reprojection_loss(pred, target)

                pred = inputs[("color", frame_id, source_scale)]
                rep_identity = self.compute_reprojection_loss(pred, target)
                
                reprojection_loss_mask = self.compute_loss_masks(rep,rep_identity,target)
                reprojection_loss_mask_iil = get_feature_oclution_mask(reprojection_loss_mask)
                #Losses
                #target = outputs[("color_refined", frame_id, scale)] #Lighting
                pred = outputs[("color_refined", frame_id, scale)]
                #SIMM
                loss_reprojection += (self.compute_reprojection_loss(pred, target) * reprojection_loss_mask).sum() / reprojection_loss_mask.sum()
                #Multiscale SIMM
                #loss_reprojection += (self.get_ms_simm_loss(pred, target) * reprojection_loss_mask).sum() / reprojection_loss_mask.sum()
                #Illuminations invariant loss
                target = inputs[("color", 0, 0)]
                pred = outputs[("color_refined", frame_id, scale)]
                loss_ilumination_invariant += (self.get_ilumination_invariant_loss(pred,target) * reprojection_loss_mask_iil).sum() / reprojection_loss_mask_iil.sum()
 
            
            loss += loss_reprojection / 2.0
            loss += self.opt.illumination_invariant * loss_ilumination_invariant / 2.0
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss, inputs):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            #print(reprojection_loss.shape)
            #print(identity_reprojection_loss.shape)
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float()           
            
        return reprojection_loss_mask

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
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
            wandb.log({"disp_multi_{}/{}".format(s, j): wandb.Image(disp.transpose(1, 2, 0))},step=self.step)

           
                  

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        #self.opt.load_weights_folder = "/workspace/endo-manydepth/manydepth/weights/scared_ill0.5_normal_finalTransformer/models/weights_19"
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

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def flow2rgb(self,flow_map, max_value):
        flow_map_np = flow_map.detach().cpu().numpy()
        _, h, w = flow_map_np.shape
        flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
        rgb_map = np.ones((3,h,w)).astype(np.float32)
        if max_value is not None:
            normalized_flow_map = flow_map_np / max_value
        else:
            normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
        rgb_map[0] += normalized_flow_map[0]
        rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
        rgb_map[2] += normalized_flow_map[1]
        return rgb_map.clip(0,1)

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
        """
        Visualize a 3-channel image with X, Y, and Z components of normal vectors.
        
        Args:
            xyz_image (torch.Tensor): The input normal image with shape (3, height, width).
        """
        # Ensure the input tensor is on the CPU and in numpy format
        normal_image_np = xyz_image.cpu().numpy()

        # Normalize the normal vectors to unit length
        normal_image_np /= np.linalg.norm(normal_image_np, axis=0)

        # Transpose the dimensions to (height, width, channels) for matplotlib
        normal_image_np = np.transpose(normal_image_np, (1, 2, 0))

        # Shift and scale the normal vectors to the [0, 1] range for visualization
        normal_image_np = 0.5 * normal_image_np + 0.5

        return normal_image_np

    def visualize_normals(self,batch_normals):
        """
        Visualize a batch of normalized normal vectors as RGB images.

        Args:
            batch_normals (np.ndarray): A batch of normalized normal vectors with shape (batch, channels, height, width).

        Returns:
            np.ndarray: An array of RGB images representing the normal vectors.
        """
        batch_normals = batch_normals.cpu().numpy()
        # Scale and shift to map the normals to the 0-255 range
        scaled_normals = ((batch_normals + 1) / 2 * 255).astype(np.uint8)
        # Convert channels to (height, width, channels)
        #print(scaled_normals.shape)
        transposed_normals = np.transpose(scaled_normals, (1, 2, 0))
        return transposed_normals

        
    def norm_to_rgb(self,norm):
        pred_norm = norm.detach().cpu().permute(1, 2, 0).numpy()  # (H, W, 3)
        norm_rgb = ((pred_norm[...] + 1)) / 2 * 255
        norm_rgb = np.clip(norm_rgb, a_min=0, a_max=255)
        norm_rgb = norm_rgb.astype(np.uint8)
        return norm_rgb