# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        #print(self.pix_coords.shape)
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        #print(cam_points)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        #print(cam_points.shape)
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)
        #print(cam_points.shape)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        #print(pix_coords.shape)
        #print(pix_coords)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)



def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        """
        Instiantiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the source image
            :param flow: the output from the U-Net
        """
        #print(self.grid)
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:, i, ...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode="border",align_corners=True)

def get_ilumination_invariant_features(img):
    #ENDOVIS dataset
    if(img.shape[1] != 1):
        img_gray = transforms.functional.rgb_to_grayscale(img,1)
    else: 
        img_gray = img
    
    K1 = torch.Tensor([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]]).to(device=img_gray.device)
    K2 = torch.Tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).to(device=img_gray.device)
    K3 = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(device=img_gray.device)
    K4 = torch.Tensor([[2, 1, 0], [1, 0, -1], [0, -1, -2]]).to(device=img_gray.device)
    K5 = torch.Tensor([[1, 0,-1], [2, 0, -2], [1, 0, -1]]).to(device=img_gray.device)
    K6 = torch.Tensor([[0,-1,-2], [1, 0, -1], [2, 1, 0]]).to(device=img_gray.device)
    K7 = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).to(device=img_gray.device)
    K8 = torch.Tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]).to(device=img_gray.device)

    sq_D = torch.zeros_like(img_gray, device=img_gray.device)
    padding = (3 - 1) // 2  # Padding to maintain input size
    M1 = F.conv2d(img_gray, K1.view(1, 1, 3, 3), padding=padding)
    #sq_D += torch.pow(M1,2)
    M2 = F.conv2d(img_gray, K2.view(1, 1, 3, 3), padding=padding)
    #sq_D += torch.pow(M2,2)
    M3 = F.conv2d(img_gray, K3.view(1, 1, 3, 3), padding=padding)
    #sq_D += torch.pow(M3,2)
    M4 = F.conv2d(img_gray, K4.view(1, 1, 3, 3), padding=padding)
    #sq_D += torch.pow(M4,2)
    M5 = F.conv2d(img_gray, K5.view(1, 1, 3, 3), padding=padding)
    #sq_D += torch.pow(M5,2)
    M6 = F.conv2d(img_gray, K6.view(1, 1, 3, 3), padding=padding)
    #sq_D += torch.pow(M6,2)
    M7 = F.conv2d(img_gray, K7.view(1, 1, 3, 3), padding=padding)
    #sq_D += torch.pow(M7,2)
    M8 = F.conv2d(img_gray, K8.view(1, 1, 3, 3), padding=padding)
    #sq_D += torch.pow(M8,2)

    #NormD = torch.sqrt(torch.clamp(sq_D, min=1e-9))


    #t = torch.cat((M1/NormD,M2/NormD,M3/NormD,M4/NormD,M5/NormD,M6/NormD,M7/NormD,M8/NormD), dim = 1)
    t = torch.cat((M1,M2,M3,M4,M5,M6,M7,M8), dim = 1)

    return t      

def get_feature_oclution_mask(img):
    kernel = torch.tensor([[1, 1, 1],[1, 1, 1],[1, 1, 1]]).to(device=img.device).type(torch.cuda.FloatTensor)
    padding = (3 - 1) // 2  # Padding to maintain input size
    o = F.conv2d(img, kernel.view(1, 1, 3, 3), padding=padding)
    t = torch.cat((o,o,o,o,o,o,o,o), dim = 1)
    
    return t



def calculate_surface_normal_from_depth(depth_map, K):
    # Calculate gradients using finite differences
    dz_dx = torch.nn.functional.conv2d(depth_map.unsqueeze(1), torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]).unsqueeze(0).float())
    dz_dy = torch.nn.functional.conv2d(depth_map.unsqueeze(1), torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]).unsqueeze(0).float())

    # Calculate surface normals
    V_p = torch.cat((dz_dx, dz_dy, torch.ones_like(depth_map)), dim=1)

    # Apply K⁻¹ to transform to world coordinates
    V_p = torch.matmul(torch.inverse(K), V_p)

    # Normalize the surface normals
    V_p /= torch.norm(V_p, p=2, dim=1, keepdim=True)

    return V_p