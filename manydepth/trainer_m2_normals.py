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
#from tensorboardX import SummaryWriter
import wandb


wandb.init(project="MySfMLearner3", entity="respinosa")


import json

from utils import *
from layers import *

import datasets
import networks
import matplotlib.pyplot as plt


_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer_Monodepth2:
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

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
 
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["normal"] = networks.NormalDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["normal"].to(self.device)
        self.parameters_to_train += list(self.models["normal"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

                self.models["lighting"] = networks.LightingDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
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

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        #if self.opt.load_weights_folder is not None:
        self.load_model()
        self.freeze_models()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

                # DATA
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "endovis": datasets.SCAREDDataset}

        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

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

        #self.writers = {}
        #for mode in ["train", "val"]:
        #    self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

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

    def freeze_models(self):
        # Freeze all layers
        for param in self.models["encoder"].parameters():
            param.requires_grad = False
        for param in self.models["depth"].parameters():
            param.requires_grad = False
        for param in self.models["pose_encoder"].parameters():
            param.requires_grad = False
        for param in self.models["lighting"].parameters():
            param.requires_grad = False
        
    
    def unfreeze_models(self):
        for param in self.models["encoder"].parameters():
            param.requires_grad = True
        for param in self.models["depth"].parameters():
            param.requires_grad = True
        for param in self.models["pose_encoder"].parameters():
            param.requires_grad = False
        for param in self.models["lighting"].parameters():
            param.requires_grad = False

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
        #self.set_train()
        

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

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        outputs["normal_inputs"] = self.models["normal"](features)
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
                    """
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]"""
                    
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)
                    """
                    if f_i < 0:
                        iif_all = [get_ilumination_invariant_features(pose_feats[f_i]),get_ilumination_invariant_features(pose_feats[0])] 
                    else:
                        iif_all = [get_ilumination_invariant_features(pose_feats[0]),get_ilumination_invariant_features(pose_feats[f_i])] 
                    """
                    
                    #motion_inputs = [self.models["ii_encoder"](torch.cat(pose_inputs, 1))]
                    #outputs_mf = self.models["motion_flow"](pose_inputs[0])
                    
                    """input_combined = pose_inputs
                    concatenated_list = []
                    # Iterate over the corresponding tensors in list1 and list2 and concatenate them
                    for tensor1, tensor2 in zip(pose_inputs[0], motion_inputs[0]):
                        concatenated_tensor = torch.cat([tensor1, tensor2], dim=1)
                        concatenated_list.append(concatenated_tensor)
                    
                    axisangle, translation = self.models["pose"]([concatenated_list])
                    """
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
                    #outputs_mf = self.models["motion_flow"](pose_inputs[0])
                    
                    for scale in self.opt.scales:
                        outputs["b_"+str(scale)+"_"+str(f_i)] = outputs_lighting[("lighting", scale)][:,0,None,:, :]
                        outputs["c_"+str(scale)+"_"+str(f_i)] = outputs_lighting[("lighting", scale)][:,1,None,:, :]
                        #outputs["mf_"+str(scale)+"_"+str(f_i)] = outputs_mf[("flow", scale)]
                        
            
            for f_i in self.opt.frame_ids[1:]:
                for scale in self.opt.scales:
                    #outputs[("color_motion", f_i, scale)] = self.spatial_transform(inputs[("color", 0, 0)],outputs["mf_"+str(0)+"_"+str(f_i)])
                    outputs[("bh",scale, f_i)] = F.interpolate(outputs["b_"+str(scale)+"_"+str(f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    outputs[("ch",scale, f_i)] = F.interpolate(outputs["c_"+str(scale)+"_"+str(f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    outputs[("color_refined", f_i, scale)] = outputs[("ch",scale, f_i)] * inputs[("color", 0, 0)] + outputs[("bh", scale, f_i)]


        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

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
                
        #Normal prediction        
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            features = self.models["encoder"](outputs[("color", frame_id, 0)])
            outputs[("normal",frame_id)] = self.models["normal"](features)
            


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

    def norm_loss(self, pred, target, rotation_matrix,frame_id):
        #print(pred.shape)
        #print(target.shape)
        
        if frame_id < 0:
            rotation_matrix = rotation_matrix.transpose(1, 2)                

        #rotation_matrix = rotation_matrix[:, :3, :3]

        target = target.permute(0,2,3,1)
        
        #target = torch.nn.functional.normalize(target, p=2, dim=1)

        pred = pred.permute(0,2,3,1)
        batch_size, height, width, channels = pred.shape
        #print(target.permute(0,3,1,2).view(batch_size,channels,-1).shape)
        #print(rotation_matrix[:, :3, :3].shape)
        #print(batch_size, height, width, channels)
        #pred = torch.nn.functional.normalize(pred, p=2, dim=1)
        #print(reshaped_normal_shapes.shape)
        #print(rotation_matrix.unsqueeze(1).shape)
        #print(target.shape)
        #print(rotation_matrix.shape)
        #print(channels)
        #print(target.permute(0, 3, 1, 2).shape)
        rotated_images = torch.matmul(target.view(batch_size,-1,3), rotation_matrix[:, :3, :3]) 
        #print(rotated_images.shape)
        # Reshape the rotated images back to the original shape (12, 3, 256, 320)
        rotated_images = rotated_images.view(batch_size,height,width,channels)
        #rotated_images = rotated_images.permute(0,3,1,2)
        #print(rotated_images.shape)
        #result.view(12, 256, 320, 3)
        #pred = pred.permute(0,2,3,1)
        #print(pred.shape)
        #print(rotated_images.shape)
        abs_diff = torch.abs(pred - rotated_images)
        l1_loss = abs_diff.mean(1, True)
        #print(l1_loss)
        return l1_loss.sum()

    
    def compute_orth_loss(self, disp, N_hat, K_inv):
        _, D = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        #print(D.shape)
        #print(N_hat.shape)
        # Compute orthogonality loss
        orth_loss = 0.0
        
        #D = D.permute(0, 2, 3, 1)
        D_inv = 1.0 / D.permute(0, 2, 3, 1)
        #N_hat = N_hat.permute(0, 2, 3, 1)
        #N_hat = torch.nn.functional.normalize(N_hat, p=2, dim=1)
        
        batch_size, height, width, channels = D_inv.shape
        p1 = (0,1)
        p2 = (0,2)
        p3 = (1,0)
        p4 = (2,0)
 
        # Homogeneous coordinates
        p = torch.arange(height, dtype=torch.float32).view(1, height, 1).to(device=K_inv.device)
        q = torch.arange(width, dtype=torch.float32).view(1, 1, width).to(device=K_inv.device)
     
        p = p.expand(batch_size, height, width).unsqueeze(-1)
        q = q.expand(batch_size, height, width).unsqueeze(-1)
        
        P = torch.cat([p, q, torch.ones_like(p)], dim=-1)
        
        X_tilde_p = torch.matmul(K_inv[:, :3, :3], P.permute(0,3,1,2).view(batch_size,3,-1))

        Cpp = torch.einsum('bijk,bijk->bij', N_hat.permute(0, 2, 3, 1), X_tilde_p.view(batch_size,3,height, width).permute(0,2,3,1))
        #print(P.shape)
        for p_idx in [p1, p2, p3, p4]:
            q = P.roll(shifts=p_idx,dims=(-2, -1))  # Keep only the first two dimensions
            #print(q.shape)
            X_tilde_q = torch.matmul(K_inv[:, :3, :3], q.permute(0, 3, 1, 2).view(batch_size,3,-1))
            Cpq = torch.einsum('bijk,bijk->bij', N_hat.permute(0, 2, 3, 1), X_tilde_q.view(batch_size,3,height, width).permute(0,2,3,1))
            orth_loss += torch.abs(D_inv * torch.unsqueeze(Cpq,0).permute(1,2,3,0) - D_inv * torch.unsqueeze(Cpp,0).permute(1,2,3,0))

        orth_loss = orth_loss.sum()

        return orth_loss

    def compute_orth_loss2(self, disp, N_hat, K_inv):
        _, D = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        orth_loss = 0.0
        
        D = D.permute(0, 2, 3, 1)

        roll_offsets = []
        roll_offsets.append((-1, -1)) #abajo derecha
        roll_offsets.append((1, 1))


        batch_size, height, width, channels = D.shape
 
        # Homogeneous coordinates
        p = torch.arange(height, dtype=torch.float32).view(1, height, 1).to(device=K_inv.device)
        
        q = torch.arange(width, dtype=torch.float32).view(1, 1, width).to(device=K_inv.device)
        p = p.expand(batch_size, height, width).unsqueeze(-1)
        q = q.expand(batch_size, height, width).unsqueeze(-1)
        
        P = torch.cat([p, q, torch.ones_like(p)], dim=-1)

        """print("P")
        print(P.shape)
        print(P[0,:3,:3])"""

        #####################################
               
        pa_tl = torch.roll(P, shifts=1, dims=1)
        pa_tl = torch.roll(pa_tl, shifts=1, dims=2)

        """print("Pa_tl")
        print(Pa_tl.shape)
        print(Pa_tl[0,:3,:3])"""
    
        pb_br = torch.roll(P, shifts=-1, dims=1)
        pb_br = torch.roll(pb_br, shifts=-1, dims=2)


        """print("Pb_br")
        print(Pb_br.shape)
        print(Pb_br[0,:3,:3])"""

        #####################################
        pa_tr = torch.roll(P, shifts=1, dims=1)
        pa_tr = torch.roll(pa_tr, shifts=-1, dims=2)
                
        """print("Pa_tr")
        print(Pa_tr.shape)
        print(Pa_tr[0,:3,:3])"""

        pb_bl = torch.roll(P, shifts=-1, dims=1)
        pb_bl = torch.roll(pb_bl, shifts=1, dims=2)
        
        """
        print("Pb_bl")
        print(Pb_bl.shape)
        print(Pb_bl[0,:3,:3])"""

        #pa_tl, pb_br = P, P_tl_br
        #pa_tr, pb_bl = P, P_tr_bl

        """
        print(pa_tl.shape)
        print(pb_br.shape)
        print(pa_tr.shape)
        print(pb_bl.shape)

        print(pa_tl.permute(0, 3, 1, 2).view(batch_size,3,-1).shape)
        print(pb_br.permute(0, 3, 1, 2).view(batch_size,3,-1).shape)
        print(pa_tr.permute(0, 3, 1, 2).view(batch_size,3,-1).shape)
        print(pb_bl.permute(0, 3, 1, 2).view(batch_size,3,-1).shape)
        """
        #print(D.shape)
        #print(torch.matmul(K_inv[:, :3, :3], pa_tl.permute(0, 3, 1, 2).view(batch_size,3,-1)).shape)
        V = 0
        pa_pb1 = torch.matmul(K_inv[:, :3, :3], pa_tl.permute(0, 3, 1, 2).view(batch_size,3,-1)) - torch.matmul(K_inv[:, :3, :3], pb_br.permute(0, 3, 1, 2).view(batch_size,3,-1))
        pa_pb2 = torch.matmul(K_inv[:, :3, :3], pa_tr.permute(0, 3, 1, 2).view(batch_size,3,-1)) - torch.matmul(K_inv[:, :3, :3], pb_bl.permute(0, 3, 1, 2).view(batch_size,3,-1))
        V = D * pa_pb1.view(batch_size,3,height,width).permute(0,2,3,1) - D * pa_pb2.view(batch_size,3,height,width).permute(0,2,3,1)
        #V = torch.abs(V)
        print(V)
        #¿print(N_hat.shape)
        orth_loss = torch.einsum('bijk,bijk->bij', N_hat.permute(0, 2, 3, 1), V)
               
        #print (orth_loss.shape)

        return orth_loss.sum()


    
    def get_ilumination_invariant_loss(self, pred, target):
        features_p = get_ilumination_invariant_features(pred)
        features_t = get_ilumination_invariant_features(target)
        """abs_diff = torch.abs(features_t - features_p)
        l1_loss = abs_diff.mean(1, True)"""

        ssim_loss = self.ssim(features_p, features_t).mean(1, True)
        #ii_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return ssim_loss
    

    def compute_losses(self, inputs, outputs):

        losses = {}
        loss_reprojection = 0
        loss_ilumination_invariant = 0
        total_loss = 0
        #orthonogal_loss = 0
        normal_loss = 0

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

                reprojection_loss_mask = self.compute_loss_masks(rep,rep_identity)
                reprojection_loss_mask_iil = get_feature_oclution_mask(reprojection_loss_mask)
                #Losses
                target = outputs[("color_refined", frame_id, scale)] #Lighting
                pred = outputs[("color", frame_id, scale)]
                loss_reprojection += (self.compute_reprojection_loss(pred, target) * reprojection_loss_mask).sum() / reprojection_loss_mask.sum()
                #Illuminations invariant loss
                #target = inputs[("color", 0, 0)]
                loss_ilumination_invariant += (self.get_ilumination_invariant_loss(pred,target) * reprojection_loss_mask_iil).sum() / reprojection_loss_mask_iil.sum()
                #Normal loss
                normal_loss += (self.norm_loss(outputs[("normal",frame_id)][("normal", scale)],outputs["normal_inputs"][("normal", scale)], rot_from_axisangle(outputs[("axisangle", 0, frame_id)][:, 0].detach()),frame_id) * reprojection_loss_mask).sum() / reprojection_loss_mask.sum()
                
            loss += loss_reprojection / 2.0    
            #Normal loss
            #if self.normal_flag == 1:
            #self.normal_weight = 0.005
            #self.orthogonal_weight = 0.001
            loss += 0.1 * normal_loss / 2.0
            #Orthogonal loss
            loss += 0.5 * self.compute_orth_loss2(outputs[("disp", scale)], outputs["normal_inputs"][("normal", scale)], inputs[("inv_K", scale)])
                
            #Illumination invariant loss
            loss += 0.1 * loss_ilumination_invariant / 2.0
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
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
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
                    #wandb.log({"color_pred_flow{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs[("color_motion", frame_id, s)][j].data)},step=self.step)
                    wandb.log({"color_pred_refined_{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs[("color_refined", frame_id, s)][j].data)},step=self.step)
                    #wandb.log({"normal_{}_{}/{}".format(frame_id, s, j): wandb.Image(self.visualize_normal_image(inputs[("normal",0)][j]))},step=self.step)
                    #wandb.log({"contrast_{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs["c_"+str(frame_id)+"_"+str(s)][j].data)},step=self.step)
            disp = self.colormap(outputs[("disp", s)][j, 0])
            wandb.log({"disp_multi_{}/{}".format(s, j): wandb.Image(disp.transpose(1, 2, 0))},step=self.step)
            wandb.log({"normal_target_{}/{}".format(s, j): wandb.Image(self.norm_to_rgb(outputs["normal_inputs"][("normal", 0)][j].data))},step=self.step)
            #wandb.log({"normal_predicted{}/{}".format(s, j): wandb.Image(self.visualize_normals(outputs["normal"][("normal", 0)][j].data))},step=self.step)
            """f = outputs["mf_"+str(s)+"_"+str(frame_id)][j].data
            flow = self.flow2rgb(f,32)
            flow = torch.from_numpy(flow)
            wandb.log({"motion_flow_{}_{}".format(s,j): wandb.Image(flow)},step=self.step)"""
            """if self.opt.predictive_mask:
                for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                    wandb.log({"predictive_mask_{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...])},self.step)
            elif not self.opt.disable_automasking:
                wandb.log({
                "automask_{}/{}".format(s, j):
                wandb.Image(outputs["identity_selection/{}".format(s)][j][None, ...])}, self.step)"""
                  

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
        #self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        self.opt.load_weights_folder = "/workspace/monodepth/weights/ENDOVIS/iil/models/weights_19"
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
        """
        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")"""

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
