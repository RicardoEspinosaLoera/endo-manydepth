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


wandb.init(project="MySfMLNormal", entity="respinosa")


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
                         "endovis": datasets.SCAREDDataset,
                         "RNNSLAM": datasets.SCAREDDataset}

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
        #outputs["normal_depth"] = calculate_surface_normal_from_depth()
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
                #print("Pixels")
                #print(pix_coords.shape)
                #print(pix_coords)
                #print(outputs[("sample", frame_id, scale)].shape)
                #print(inputs[("color", frame_id, source_scale)].shape)
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
        
        if frame_id < 0:
            rotation_matrix = rotation_matrix.transpose(1, 2)                

        target = target.permute(0,2,3,1)
        
        pred = pred.permute(0,2,3,1)
        batch_size, height, width, channels = pred.shape

        rotated_images = torch.matmul(target.view(batch_size,-1,3), rotation_matrix[:, :3, :3]) 
        
        rotated_images = rotated_images.view(batch_size,height,width,channels)

        abs_diff = torch.abs(pred - rotated_images)
        l1_loss = abs_diff.mean(1, True)
        return l1_loss.sum()

    
    def compute_orth_loss(self, disp, N_hat, K_inv):
        orth_loss = 0
        _, D = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        D_inv = 1.0 / D
        N_hat = torch.nn.functional.normalize(N_hat, dim=1)
        batch_size,channels, height, width  = D.shape
        meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)

        ones = torch.ones(batch_size, 1, height * width)   

        q1 = np.roll(id_coords,(-1), axis=(2))
        q2 = np.roll(id_coords,(-2), axis=(2))
        #patl = np.roll(patl,(1), axis=(1))
        q3 = np.roll(id_coords,(-1), axis=(1))
        q4 = np.roll(id_coords,(-2), axis=(1))
        

        id_coords = torch.from_numpy(id_coords)
        q1 = torch.from_numpy(q1)
        q2 = torch.from_numpy(q2)
        q3 = torch.from_numpy(q3)
        q4 = torch.from_numpy(q4)

        p = [q1,q2,q3,q4,id_coords]
        p_names = ["q1","q2","q3","q4","p"]
        ps = {}
        for idx,p in enumerate(p):
            pix_coords = torch.unsqueeze(torch.stack(
                [p[0].view(-1), p[1].view(-1)], 0), 0)

            pix_coords = pix_coords.repeat(batch_size, 1, 1)
            pix_coords = torch.cat([pix_coords, ones], 1)
            ps[p_names[idx]] = pix_coords
            #print(pix_coords.shape)

        Ds = {}
        d_names = ["Da_q1","Db_q2","Da_q3","Db_q4"]
        for idx,p in enumerate(p_names[:4]):
            #print(p)
            pix_coords = ps[p][:, :2, :] / (ps[p][:, 2, :].unsqueeze(1) + 1e-7)
            pix_coords = pix_coords.view(batch_size, 2, height, width)
            pix_coords = pix_coords.permute(0, 2, 3, 1)
            pix_coords[..., 0] /= width - 1
            pix_coords[..., 1] /= height - 1
            pix_coords = (pix_coords - 0.5) * 2
            Ds[d_names[idx]] = F.grid_sample(D_inv,pix_coords.to(device=K_inv.device),align_corners=True)
            #wandb.log({"depth_grid": wandb.Image(Ds[d_names[idx]][0])},step=self.step)
        
        X_tilde_p = torch.matmul(K_inv[:, :3, :3],ps["p"].to(device=K_inv.device))
        Cpp = torch.einsum('bijk,bijk->bijk', N_hat, X_tilde_p.view(batch_size,3,height,width))
        for idx,p in enumerate(p_names[:4]):
            X_tilde_q = torch.matmul(K_inv[:, :3, :3], ps[p].to(device=K_inv.device))
            Cpq = torch.einsum('bijk,bijk->bijk', N_hat, X_tilde_q.view(batch_size,3,height,width))
            orth_loss += torch.abs(D_inv * Cpq - Ds[d_names[idx]] * Cpp)
        return orth_loss.sum()

    def compute_orth_loss4(self, disp, N_hat, K_inv):
        orth_loss = 0
        _, D = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        D_inv = 1.0 / D
        batch_size,channels, height, width  = D.shape
        y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        y = y.float().unsqueeze(0).unsqueeze(0)
        x = x.float().unsqueeze(0).unsqueeze(0)
        ones = torch.ones( batch_size, 1, height * width).to(device=K_inv.device)
        print("ones",ones.shape)
        magnitude = torch.norm(N_hat, dim=1, keepdim=True)
        magnitude[magnitude == 0] = 1
        N_hat_normalized = N_hat / magnitude

        # Calculate positions of top-left, bottom-right, top-right, and bottom-left pixels
        normal = torch.stack([x,y], dim=-1).to(device=K_inv.device)
        right = torch.stack([torch.clamp(x + 1, min=0, max=width-1), y], dim=-1).to(device=K_inv.device)
        right_right = torch.stack([torch.clamp(x + 2, min=0, max=width-1),y], dim=-1).to(device=K_inv.device)
        bottom = torch.stack([x, torch.clamp(y + 1, min=0, max=height-1)], dim=-1).to(device=K_inv.device)
        bottom_bottom = torch.stack([x, torch.clamp(y + 2, min=0, max=height-1)], dim=-1).to(device=K_inv.device)
        
        normal_flat = normal.view(1,-1, 2).expand(12, -1, -1)
        rigth_flat = right.view(1,-1, 2).expand(12, -1, -1)
        right_right_flat = right_right.view(1,-1, 2).expand(12, -1, -1)
        bottom_flat = bottom.view(1,-1, 2).expand(12, -1, -1)
        bottom_bottom_flat = bottom_bottom.view(1,-1, 2).expand(12, -1, -1)
        print("normal_flat",normal_flat.shape)
        normal_flat = torch.cat([normal_flat.permute(0,2,1), ones], dim=1)
        rigth_flat = torch.cat([rigth_flat.permute(0,2,1), ones], dim=1)
        right_right_flat = torch.cat([right_right_flat.permute(0,2,1), ones], dim=1)
        bottom_flat = torch.cat([bottom_flat.permute(0,2,1), ones], dim=1)
        bottom_bottom_flat = torch.cat([bottom_bottom_flat.permute(0,2,1), ones], dim=1)
        
        padded_depth = torch.nn.functional.pad(D_inv, (1, 1, 2, 2), mode='constant', value=0)

        # Extracting the specific neighbors
        # right right
        right_depth = padded_depth[:, :, :, 1:]  # right
        right_right_depth = padded_depth[:, :, :, 2:]  # right-right
        # bottom and bottom-bottom
        bottom_depth = padded_depth[:, :, 1:, :]  # bottom
        botto_bottom_depth = padded_depth[:, :, 2:, :]  # Bottom-bottom

       
        X_tilde_p = torch.matmul(K_inv[:, :3, :3],normal_flat)

        Cpp = torch.einsum('bijk,bijk->bi', N_hat_normalized,X_tilde_p.view(batch_size,3,height,width))
        movements = [right_flat,right_right_flat,bottom_flat,bottom_bottom_flat]
        depths = [right_depth,right_right_depth,bottom_depth,botto_bottom_depth]

        for idx,m in enumerate(movements):
            X_tilde_q = torch.matmul(K_inv[:, :3, :3], m)
            Cpq = torch.einsum('bijk,bijk->bi', N_hat_normalized,X_tilde_q.view(batch_size,3,height,width))
            orth_loss += torch.abs(D_inv.view(batch_size, 1, -1) * Cpq.unsqueeze(-1) - depths[idx].reshape(batch_size,1,-1) * Cpp.unsqueeze(-1))
           
        
        x = torch.tensor([[-1, 0, 1]]).to(device=K_inv.device).type(torch.cuda.FloatTensor)
        y = torch.tensor([[-1], [0], [1]]).to(device=K_inv.device).type(torch.cuda.FloatTensor)
        gradient_x = F.conv2d(I, x.view(1, 3, 1, 1))
        gradient_y = F.conv2d(I, y.view(1, 3, 1, 1))

        gradient_magitude = torch.sqrt(gradient_x**2 + gradient_y**2)
        gradient_magitude = torch.mean(gradient_magitude)
        # Calculate G(p)
        G_p = torch.exp(-1 * torch.abs(gradient_magitude) / 1**2)
        return torch.sum(G_p*orth_loss)
        
    def compute_orth_loss2(self, disp, N_hat, K_inv):
        orth_loss = 0
        _, D = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        #print(D.shape)
        #D = D.permute(0, 2, 3, 1)
        #N_hat = N_hat.permute(0, 2, 3, 1)
        N_hat_n = torch.nn.functional.normalize(N_hat, dim=1)
        batch_size,channels, height, width  = D.shape
        meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)

        ones = torch.ones(batch_size, 1, height * width)   
        
        patl = np.roll(id_coords,(-1), axis=(2))
        patl = np.roll(patl,(-1), axis=(1))

        #print(id_coords.shape)

        pbbr = np.roll(id_coords,(1), axis=(2))
        pbbr = np.roll(pbbr,(1), axis=(1))

        #print(pbbr)

        patr = np.roll(id_coords,(1), axis=(2))
        patr = np.roll(patr,(-1), axis=(1))

        #print(patr)

        pabl = np.roll(id_coords,(-1), axis=(2))
        pabl = np.roll(pabl,(1), axis=(1))

        #print(pabl)

        id_coords = torch.from_numpy(id_coords)
        patl = torch.from_numpy(patl)
        pbbr = torch.from_numpy(pbbr)
        patr = torch.from_numpy(patr)
        pbbl = torch.from_numpy(pabl)

        p = [patl,pbbr,patr,pbbl]
        p_names = ["patl","pbbr","patr","pbbl"]
        ps = {}
        for idx,p in enumerate(p):
            pix_coords = torch.unsqueeze(torch.stack(
                [p[0].view(-1), p[1].view(-1)], 0), 0)

            pix_coords = pix_coords.repeat(batch_size, 1, 1)
            pix_coords = torch.cat([pix_coords, ones], 1)
            ps[p_names[idx]] = pix_coords
            #print(pix_coords.shape)
        
        Ds = {}
        d_names = ["Da_tl","Db_br","Da_tr","Db_bl"]
        for idx,p in enumerate(p_names):
            #print(p)
            coords = ps[p][:, :2, :] / (ps[p][:, 2, :].unsqueeze(1) + 1e-7)
            coords = coords.view(batch_size, 2, height, width)
            coords = coords.permute(0, 2, 3, 1)
            coords[..., 0] /= width - 1
            coords[..., 1] /= height - 1
            coords = (coords - 0.5) * 2
            #print(pix_coords)
            Ds[d_names[idx]] = F.grid_sample(D,coords.to(device=K_inv.device),align_corners=True)
            #wandb.log({"depth_grid": wandb.Image(Ds[d_names[idx]][0])},step=self.step)

        pa = torch.matmul(K_inv[:, :3, :3],ps["patl"].to(device=K_inv.device))
        pb = torch.matmul(K_inv[:, :3, :3],ps["pbbr"].to(device=K_inv.device))
        
        
        V = Ds["Da_tl"].view(batch_size, 1, -1) * pa - Ds["Db_br"].view(batch_size, 1, -1) * pb
        pa = torch.matmul(K_inv[:, :3, :3],ps["patr"].to(device=K_inv.device))
        pb = torch.matmul(K_inv[:, :3, :3],ps["pbbl"].to(device=K_inv.device))

        V += Ds["Da_tr"].view(batch_size, 1, -1) * pa - Ds["Db_bl"].view(batch_size, 1, -1) * pb

        orth_loss = torch.einsum('bijk,bijk->bi', N_hat_n, V.view(batch_size,3,height, width))
        
        return torch.mean(torch.sum(orth_loss,dim=1))
    
    def compute_orth_loss3(self, disp, N_hat, K_inv):
        orth_loss = 0
        _, D = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        #D_inv = 1.0 / D
        batch_size,channels, height, width  = D.shape
        # Create coordinate grids
        y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        y = y.float().unsqueeze(0).unsqueeze(0)
        x = x.float().unsqueeze(0).unsqueeze(0)
        
        #ones = torch.ones(12, height * width,1).to(device=K_inv.device)
        ones = torch.ones(batch_size, 1, height * width).to(device=K_inv.device)
        magnitude = torch.norm(N_hat, dim=1, keepdim=True)
        magnitude[magnitude == 0] = 1
        N_hat_normalized = N_hat / magnitude

        # Calculate positions of top-left, bottom-right, top-right, and bottom-left pixels
        """
        top_left = torch.stack([torch.clamp(y + 1.0, min=0, max=height-1), torch.clamp(x + 1.0, min=0, max=width-1)], dim=-1).to(device=K_inv.device)
        bottom_right = torch.stack([torch.clamp(y - 1.0, min=0, max=height-1), torch.clamp(x - 1.0, min=0, max=width-1)], dim=-1).to(device=K_inv.device)
        top_right = torch.stack([torch.clamp(y + 1.0, min=0, max=height-1), torch.clamp(x - 1.0, min=0, max=width-1)], dim=-1).to(device=K_inv.device)
        bottom_left = torch.stack([torch.clamp(y - 1.0, min=0, max=height-1), torch.clamp(x + 1.0, min=0, max=width-1)], dim=-1).to(device=K_inv.device)"""
        
        
        top_left = torch.stack([torch.clamp(x - 0.5, min=0, max=width-1), torch.clamp(y - 0.5, min=0, max=height-1)], dim=-1).to(device=K_inv.device)
        bottom_right = torch.stack([torch.clamp(x + 0.5, min=0, max=width-1), torch.clamp(y + 0.5, min=0, max=height-1)], dim=-1).to(device=K_inv.device)
        top_right = torch.stack([torch.clamp(x + 0.5, min=0, max=width-1), torch.clamp(y - 0.5, min=0, max=height-1)], dim=-1).to(device=K_inv.device)
        bottom_left = torch.stack([torch.clamp(x - 0.5, min=0, max=width-1), torch.clamp(y + 0.5, min=0, max=height-1)], dim=-1).to(device=K_inv.device)
        
        """
        top_left = torch.stack([x - 1, y - 1], dim=-1).to(device=K_inv.device)
        bottom_right = torch.stack([x + 1, y + 1], dim=-1).to(device=K_inv.device)
        top_right = torch.stack([x + 1, y - 1], dim=-1).to(device=K_inv.device)
        bottom_left = torch.stack([x - 1, y + 1 ], dim=-1).to(device=K_inv.device)"""

        """
        top_left = torch.stack([torch.clamp(x + 1.0, min=0, max=width-1), torch.clamp(y + 1.0, min=0, max=height-1)], dim=-1).to(device=K_inv.device)
        bottom_right = torch.stack([torch.clamp(x - 1.0, min=0, max=width-1), torch.clamp(y - 1.0, min=0, max=height-1)], dim=-1).to(device=K_inv.device)
        top_right = torch.stack([torch.clamp(x - 1.0, min=0, max=width-1), torch.clamp(y + 1.0, min=0, max=height-1)], dim=-1).to(device=K_inv.device)
        bottom_left = torch.stack([torch.clamp(x + 1.0, min=0, max=width-1), torch.clamp(y - 1.0, min=0, max=height-1)], dim=-1).to(device=K_inv.device)"""


        #xy = torch.stack([x, y], dim=-1).to(device=K_inv.device)
        #xy = xy.view(1, -1, 2).expand(12, -1, -1)
        #positions_a, positions_b = self.compute_nearby_positions(xy.view(12,2,height,width))
        #print(xy[0])
        # Flatten and concatenate to get pairs of positions
        
        top_left_flat = top_left.view(1,-1, 2).expand(12, -1, -1)
        bottom_right_flat = bottom_right.view(1,-1, 2).expand(12, -1, -1)
        top_right_flat = top_right.view(1,-1, 2).expand(12, -1, -1)
        bottom_left_flat = bottom_left.view(1,-1, 2).expand(12, -1, -1)

        top_left_flat = torch.cat([top_left_flat.permute(0,2,1), ones], dim=1)
        bottom_right_flat = torch.cat([bottom_right_flat.permute(0,2,1), ones], dim=1)
        top_right_flat = torch.cat([top_right_flat.permute(0,2,1), ones], dim=1)
        bottom_left_flat = torch.cat([bottom_left_flat.permute(0,2,1), ones], dim=1)
        
        top_left_flat_ = D.view(batch_size, 1, -1) * top_left_flat
        bottom_right_flat_ = D.view(batch_size, 1, -1) * bottom_right_flat
        top_right_flat_ = D.view(batch_size, 1, -1) * top_right_flat
        bottom_left_flat_ = D.view(batch_size, 1, -1) * bottom_left_flat
        
        #print(top_left_flat_.shape)
        top_left_flat_ = top_left_flat_[:, :2, :] / (top_left_flat_[:, 2, :].unsqueeze(1) + 1e-7)
        top_left_depth = top_left_flat_.view(batch_size,2,height,width).clone()
        top_left_depth = top_left_depth.permute(0, 2, 3, 1)
        top_left_depth[..., 0] /= width - 1
        top_left_depth[..., 1] /= height - 1
        top_left_depth = (top_left_depth - 0.5) * 2

        bottom_right_flat_ = bottom_right_flat_[:, :2, :] / (bottom_right_flat_[:, 2, :].unsqueeze(1) + 1e-7)
        bottom_right_depth = bottom_right_flat_.view(batch_size, 2,height,width).clone()
        bottom_right_depth = bottom_right_depth.permute(0, 2, 3, 1)
        bottom_right_depth[..., 0] /= width - 1
        bottom_right_depth[..., 1] /= height - 1
        bottom_right_depth = (bottom_right_depth - 0.5) * 2

        top_right_flat_ = top_right_flat_[:, :2, :] / (top_right_flat_[:, 2, :].unsqueeze(1) + 1e-7)
        top_right_depth = top_right_flat_.view(batch_size, 2,height,width).clone()
        top_right_depth = top_right_depth.permute(0, 2, 3, 1)
        top_right_depth[..., 0] /= width - 1
        top_right_depth[..., 1] /= height - 1
        top_right_depth = (top_right_depth - 0.5) * 2

        bottom_left_flat_ = bottom_left_flat_[:, :2, :] / (bottom_left_flat_[:, 2, :].unsqueeze(1) + 1e-7)
        bottom_left_depth = bottom_left_flat_.view(batch_size, 2,height,width).clone()
        bottom_left_depth = bottom_left_depth.permute(0, 2, 3, 1)
        bottom_left_depth[..., 0] /= width - 1
        bottom_left_depth[..., 1] /= height - 1
        bottom_left_depth = (bottom_left_depth - 0.5) * 2
        
        #print(top_left_flat)
        """        
        top_left_depth = top_left_flat.permute(0, 2, 1).to(device=K_inv.device) * D.view(batch_size, 1, -1)
        bottom_right_depth = bottom_right_flat.permute(0, 2, 1).to(device=K_inv.device) * D.view(batch_size, 1, -1)
        top_right_depth = top_right_flat.permute(0, 2, 1).to(device=K_inv.device) * D.view(batch_size, 1, -1)
        bottom_left_depth = bottom_left_flat.permute(0, 2, 1).to(device=K_inv.device) * D.view(batch_size, 1, -1)"""
        #print(top_left_flat_.shape)
        D_hat_pa = torch.nn.functional.grid_sample(D, top_left_depth.reshape(batch_size,height,width,2), mode='bilinear', align_corners=False)
        D_hat_pb = torch.nn.functional.grid_sample(D, bottom_right_depth.reshape(batch_size,height,width,2), mode='bilinear', align_corners=False)
        D_hat_pa2 = torch.nn.functional.grid_sample(D, top_right_depth.reshape(batch_size,height,width,2), mode='bilinear', align_corners=False)
        D_hat_pb2 = torch.nn.functional.grid_sample(D, bottom_left_depth.reshape(batch_size,height,width,2), mode='bilinear', align_corners=False)
        #print(D_hat_pa.shape)
        #print(D_hat_pa)
        #D = D.permute(0,2,3,1)
        """
        top_left_depth = D[:,:,top_left_flat[0,:,1].long(), top_left_flat[0,:,0].long()]
        bottom_right_depth = D[:,:,bottom_right_flat[0,:,1].long(), bottom_right_flat[0,:,0].long()]
        top_right_depth = D[:,:,top_right_flat[0,:,1].long(), top_right_flat[0,:,0].long()]
        bottom_left_depth = D[:,:,bottom_left_flat[0,:,1].long(), bottom_left_flat[0,:,0].long()]"""

        #wandb.log({"disp_multi_tl": wandb.Image(top_left_depth[0].view(1,height,width))},step=self.step)
        #wandb.log({"disp_multi_o": wandb.Image(D[0].view(1,height,width))},step=self.step)

        #print(top_left_depth.shape)



        #print(top_left_flat.shape)

        pa_tl = torch.matmul(K_inv[:, :3, :3],top_left_flat.to(device=K_inv.device))
        pb_br = torch.matmul(K_inv[:, :3, :3],bottom_right_flat.to(device=K_inv.device))

        pa_tr = torch.matmul(K_inv[:, :3, :3],top_right_flat.to(device=K_inv.device))
        pb_bl = torch.matmul(K_inv[:, :3, :3],bottom_left_flat.to(device=K_inv.device))
        
        """
        pa_tl = D.view(batch_size, 1, -1) * pa_tl
        pb_br = D.view(batch_size, 1, -1) * pb_br
        pa_tr = D.view(batch_size, 1, -1) * pa_tr
        pb_bl = D.view(batch_size, 1, -1) * pb_bl"""

        """
        # Construct a new depth image using the mean of x and y coordinates
        #top_left_depth = top_left_depth.view(batch_size,3,-1)
        top_left_depth = ((top_left_depth[:, 1, :] + top_left_depth[:, 0, :]) / 2).view(batch_size,1,height,width)
        #bottom_right_depth = bottom_right_depth.view(batch_size,3,-1)
        bottom_right_depth = ((bottom_right_depth[:, 1, :] + bottom_right_depth[:, 0, :]) / 2).view(batch_size,1,height,width)
        #top_right_depth = top_right_depth.view(batch_size,3,-1)
        top_right_depth = ((top_right_depth[:, 1, :] + top_right_depth[:, 0, :]) / 2).view(batch_size,1,height,width)
        #bottom_left_depth = bottom_left_depth.view(batch_size,3,-1)
        bottom_left_depth = ((bottom_left_depth[:, 1, :] + bottom_left_depth[:, 0, :]) / 2).view(batch_size,1,height,width)
        """
        #cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        #print(top_left_flat_.shape)
        V = D_hat_pa.view(batch_size,1,-1) * pa_tl - D_hat_pb.view(batch_size,1,-1) * pb_br
        orth_loss1 = torch.sum(torch.einsum('bijk,bijk->bi', V.view(batch_size,3,height,width),N_hat_normalized.view(batch_size,3,height,width)))
        #orth_loss1 = torch.sum(torch.einsum('bijk,bijk->bi', V.view(batch_size,3,height,width),N_hat_normalized.view(batch_size,3,height,width)))
        #orth_loss1 = V.view(12,3,height,width) * N_hat_normalized
        # Sum over the channel dimension (dimension 1)
        #orth_loss1 = orth_loss1.sum(dim=1)

        V = D_hat_pa2.view(batch_size,1,-1) * pa_tr - D_hat_pb2.view(batch_size,1,-1) * pb_bl
        #orth_loss2 = torch.sum(torch.einsum('bijk,bijk->bi', V.view(batch_size,3,height,width),N_hat_normalized.view(batch_size,3,height,width)))
        orth_loss2 = torch.sum(torch.einsum('bijk,bijk->bi', V.view(batch_size,3,height,width),N_hat_normalized.view(batch_size,3,height,width)))
        #orth_loss2 = V.view(12,3,height,width) * N_hat_normalized
        # Sum over the channel dimension (dimension 1)
        #orth_loss2 = orth_loss2.sum(dim=1)
        
        
        #torch.einsum('ij,ij->', [a, b])
        #orth_loss = torch.einsum('bijk,bijk->', V.view(batch_size,3,height,width),N_hat)
        #V += (top_right_depth.view(12,1,height,width) * pa_tr.view(12,3,height,width)) - (bottom_left_depth.view(12,1,height,width) * pb_bl.view(12,3,height,width))
        #orth_loss = torch.sum(V.view(batch_size,3,height,width) * N_hat_normalized)
        #print(V.shape)
        #orth_loss = torch.einsum('bik,bik->bi', V.view(12, 3, -1),N_hat_normalized.view(12, 3, -1))
        #orth_loss = torch.sum(torch.einsum('bijk,bijk->bi', V.view(batch_size,3,height,width),N_hat_normalized))
        # Compute the dot product for orthogonality
        #orth_loss = torch.sum(V.view(batch_size,3,-1) * N_hat_normalized.view(batch_size,3,-1),dim=1)
        #return -torch.mean(torch.sum(orth_loss,dim=1))
        #print(orth_loss.shape)
        #ol = orth_loss1+orth_loss2
        return torch.mean(orth_loss1+orth_loss2)

    def compute_orth_loss5(self, disp, N_hat, K_inv):
        orth_loss = 0
        _, D = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        #D_inv = 1.0 / D
        batch_size,channels, height, width  = D.shape
        # Create coordinate grids
        y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        y = y.float().unsqueeze(0).unsqueeze(0)
        x = x.float().unsqueeze(0).unsqueeze(0)
        
        ones = torch.ones(12, height * width,1).to(device=K_inv.device)
        
        #ones = torch.ones(batch_size, 1, height * width).to(device=K_inv.device)
        magnitude = torch.norm(N_hat, dim=1, keepdim=True)
        magnitude[magnitude == 0] = 1
        N_hat_normalized = N_hat / magnitude

        top_left = torch.stack([torch.clamp(x - 1, min=0, max=width-1), torch.clamp(y - 1, min=0, max=height-1)], dim=-1).to(device=K_inv.device)
        bottom_right = torch.stack([torch.clamp(x + 1, min=0, max=width-1), torch.clamp(y + 1, min=0, max=height-1)], dim=-1).to(device=K_inv.device)
        top_right = torch.stack([torch.clamp(x + 1, min=0, max=width-1), torch.clamp(y - 1, min=0, max=height-1)], dim=-1).to(device=K_inv.device)
        bottom_left = torch.stack([torch.clamp(x - 1, min=0, max=width-1), torch.clamp(y + 1, min=0, max=height-1)], dim=-1).to(device=K_inv.device)
        
        top_left_flat = top_left.view(1,-1, 2).expand(12, -1, -1)
        bottom_right_flat = bottom_right.view(1,-1, 2).expand(12, -1, -1)
        top_right_flat = top_right.view(1,-1, 2).expand(12, -1, -1)
        bottom_left_flat = bottom_left.view(1,-1, 2).expand(12, -1, -1)

        print(top_left_flat.shape)
        print(ones.shape)

        top_left_flat = torch.cat([top_left_flat.permute(0,2,1), ones], dim=1)
        bottom_right_flat = torch.cat([bottom_right_flat.permute(0,2,1), ones], dim=1)
        top_right_flat = torch.cat([top_right_flat.permute(0,2,1), ones], dim=1)
        bottom_left_flat = torch.cat([bottom_left_flat.permute(0,2,1), ones], dim=1)
        
        padded_depth = torch.nn.functional.pad(D, (1, 1, 1, 1), mode='constant', value=0)

        # Extracting the specific neighbors
        # Top-left and bottom-right
        top_left_depth = padded_depth[:, :, :-2, :-2]  # Top-left
        #print(top_left_depth.shape)
        bottom_right_depth = padded_depth[:, :, 2:, 2:]  # Bottom-right
        # Top-right and bottom-left
        top_right_depth = padded_depth[:, :, :-2, 2:]  # Top-right
        bottom_left_depth = padded_depth[:, :, 2:, :-2]  # Bottom-left


        pa_tl = torch.matmul(K_inv[:, :3, :3],top_left_flat.to(device=K_inv.device))
        pb_br = torch.matmul(K_inv[:, :3, :3],bottom_right_flat.to(device=K_inv.device))

        pa_tr = torch.matmul(K_inv[:, :3, :3],top_right_flat.to(device=K_inv.device))
        pb_bl = torch.matmul(K_inv[:, :3, :3],bottom_left_flat.to(device=K_inv.device))

        V = top_left_depth.reshape(batch_size,1,-1) * pa_tl - bottom_right_depth.reshape(batch_size,1,-1) * pb_br
        orth_loss1 = torch.sum(torch.einsum('bijk,bijk->bi', V.view(batch_size,3,height,width),N_hat_normalized.view(batch_size,3,height,width)))

        V = top_right_depth.reshape(batch_size,1,-1) * pa_tr - bottom_left_depth.reshape(batch_size,1,-1) * pb_bl
        orth_loss2 = torch.sum(torch.einsum('bijk,bijk->bi', V.view(batch_size,3,height,width),N_hat_normalized.view(batch_size,3,height,width)))
        """
        x = torch.tensor([[-1, 0, 1]]).to(device=K_inv.device).type(torch.cuda.FloatTensor)
        y = torch.tensor([[-1], [0], [1]]).to(device=K_inv.device).type(torch.cuda.FloatTensor)
        gradient_x = F.conv2d(I, x.view(1, 3, 1, 1))
        gradient_y = F.conv2d(I, y.view(1, 3, 1, 1))
        #print(gradient_x.shape)
        #print(gradient_y.shape)
        gradient_magitude = torch.sqrt(gradient_x**2 + gradient_y**2)
        gradient_magitude = torch.mean(gradient_magitude)
        # Calculate G(p)
        G_p = torch.exp(-1 * torch.abs(gradient_magitude) / 1**2)"""

        return torch.mean(orth_loss1+orth_loss2)



    
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
        #loss_ilumination_invariant = 0
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
                #reprojection_loss_mask_iil = get_feature_oclution_mask(reprojection_loss_mask)
                #Losses
                target = outputs[("color_refined", frame_id, scale)] #Lighting
                pred = outputs[("color", frame_id, scale)]
                loss_reprojection += (self.compute_reprojection_loss(pred, target) * reprojection_loss_mask).sum() / reprojection_loss_mask.sum()
                #Illuminations invariant loss
                #target = inputs[("color", 0, 0)]
                #loss_ilumination_invariant += (self.get_ilumination_invariant_loss(pred,target) * reprojection_loss_mask_iil).sum() / reprojection_loss_mask_iil.sum()
                #Normal loss
                #normal_loss += (self.norm_loss(outputs[("normal",frame_id)][("normal", scale)],outputs["normal_inputs"][("normal", scale)], rot_from_axisangle(outputs[("axisangle", 0, frame_id)][:, 0].detach()),frame_id) * reprojection_loss_mask).sum() / reprojection_loss_mask.sum()
                
            loss += loss_reprojection / 2.0    
            #Normal loss
            #if self.normal_flag == 1:
            #self.normal_weight = 0.005
            #self.orthogonal_weight = 0.001
            #loss += 0.1 * normal_loss / 2.0
            #Orthogonal loss
            #loss += 0.5 * self.compute_orth_loss2(outputs[("disp", 0)], outputs["normal_inputs"][("normal", 0)], inputs[("inv_K", 0)])
                
            #Illumination invariant loss
            #loss += 0.5 * loss_ilumination_invariant / 2.0
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        
        total_loss /= self.num_scales
        total_loss += 0.5 * self.compute_orth_loss4(outputs[("disp", 0)], outputs["normal_inputs"][("normal", 0)], inputs[("inv_K", 0)])
        #total_loss += 0.6 * self.compute_orth_loss5(outputs[("disp", 0)], outputs["normal_inputs"][("normal", 0)], inputs[("inv_K", 0)])
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
                
            disp = self.colormap(outputs[("disp", s)][j, 0])
            wandb.log({"disp_multi_{}/{}".format(s, j): wandb.Image(disp.transpose(1, 2, 0))},step=self.step)
            wandb.log({"normal_target_{}/{}".format(s, j): wandb.Image(self.norm_to_rgb(outputs["normal_inputs"][("normal", 0)][j].data))},step=self.step)
           
                  

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
        self.opt.load_weights_folder = "/workspace/endo-manydepth/manydepth/weights/scared_ill05/models/weights_19"
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
            
        Returns:
            np.ndarray: Normalized and scaled numpy array suitable for visualization.
        """
        # Ensure the input tensor is on the CPU and in numpy format
        normal_image_np = xyz_image.cpu().numpy()
        #normal_image_np = normal_image_np.transpose(1,2,0)
        # Normalize the normal vectors to unit length
        norm = np.linalg.norm(normal_image_np, axis=0, keepdims=True)
        normal_image_np /= norm

        # Transpose the dimensions to (height, width, channels) for matplotlib
        normal_image_np = np.transpose(normal_image_np, (1, 2, 0))

        # Shift and scale the normal vectors to the [0, 1] range for visualization
        normal_image_np = 0.5 * normal_image_np + 0.5

        #print(normal_image_np.shape)
        #normal_image_np = np.transpose(normal_image_np, (1, 2, 0))
        return normal_image_np

  

        
    def norm_to_rgb(self,norm):
        pred_norm = norm.detach().cpu().permute(1, 2, 0).numpy()  # (H, W, 3)
        norm_rgb = ((pred_norm[...] + 1)) / 2 * 255
        norm_rgb = np.clip(norm_rgb, a_min=0, a_max=255)
        norm_rgb = norm_rgb.astype(np.uint8)
        return norm_rgb
