"""
Differentiable volumetric implementation used by pi-GAN generator.
"""

from pickle import GLOBAL
import time
from functools import partial

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from .math_utils_torch import *

rgb_map = torch.tensor([[0, 0, 0], 
                        [0, 85, 255], #skin
                        [0, 170, 255], # 
                        [0, 170, 255], 
                        [0, 255, 0],
                        [0, 255, 0],
                        [0, 255, 85],
                        [0, 255, 170],
                        [0, 255, 170],
                        [170, 255, 0],
                        [255, 0, 0], 
                        [255, 0, 170],
                        [255, 85, 0],
                        [255, 85, 0], 
                        [255, 170, 0],
                        [0, 255, 255], 
                        [255, 0, 170],
                        [170, 255, 255],
                        [255, 0, 255],], dtype=torch.float) / 255.0

def compute_rotation(angles, trans, device):
    """
    Return:
        rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

    Parameters:
        angles           -- torch.tensor, size (B, 3), radian
        trans            -- torch.tensor, size (B, 3)
    """

    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1]).to(device)
    zeros = torch.zeros([batch_size, 1]).to(device)
    x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
    
    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x), 
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([batch_size, 3, 3])
    
    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x

    trans = normalize_vecs(trans)

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    rotation_matrix[:, :3, :3] = rot

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    translation_matrix[:, :3, 3] = trans

    cam2world = translation_matrix @ rotation_matrix
    return cam2world

def fancy_integration_label(rgb_sigma, z_vals, device, return_rgb=False, last_back=False, white_back=False, clamp_mode=None, fill_mode=None):
    """Performs NeRF volumetric rendering."""
    global rgb_map
    batch_size, num_pixels, num_steps, _ = rgb_sigma.shape
    rgbs = rgb_sigma[..., :32]
    sigmas = rgb_sigma[..., 32:33]
    labels = rgb_sigma[..., 33:]

    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2)

    alphas = 1-torch.exp(-deltas * sigmas)
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-10], -2)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights_sum = weights.sum(2)
    if last_back:
        weights[:, :, -1] += (1 - weights_sum)
    rgb_final = torch.sum(weights * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -2)
    label_final = torch.sum(weights * labels, -2)

    if white_back:
        rgb_final = rgb_final + 1-weights_sum

    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor([1., 0, 0], device=rgb_final.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)
    
    if return_rgb:
        rgb_map = rgb_map.repeat(batch_size, 1, 1).to(device)
        label_sigma = sigmas.clone()
        label_sigma[labels.max(dim=-1, keepdim=True)[1]==0] = 0
        labels = labels.reshape(batch_size, num_pixels*num_steps, 19)
        label_rgbs = torch.bmm(labels, rgb_map).reshape(batch_size, num_pixels, num_steps, 3).clamp_(0,1)
        label_alphas = 1-torch.exp(-deltas * label_sigma)
        label_alphas_shifted = torch.cat([torch.ones_like(label_alphas[:, :, :1]), 1-label_alphas + 1e-10], -2)
        label_weights = label_alphas * torch.cumprod(label_alphas_shifted, -2)[:, :, :-1]
        label_rgb_final = torch.sum(label_weights * label_rgbs, -2)
        
        return rgb_final, depth_final, label_final, label_rgb_final, weights
    else:
        return rgb_final, depth_final, label_final, weights

def fancy_integration_label_blend(blend_mode, rgb_sigma, rgb_sigma_zero, z_vals, device, last_back=False, white_back=False, clamp_mode=None, fill_mode=None):
    """Performs NeRF volumetric rendering."""
    global rgb_map
    batch_size, num_pixels, num_steps, _ = rgb_sigma.shape
    rgbs = rgb_sigma[..., :32]
    sigmas = rgb_sigma[..., 32:33]
    labels = rgb_sigma[..., 33:]
    
    rgbs_zero = rgb_sigma_zero[..., :32]
    sigmas_zero = rgb_sigma_zero[..., 32:33]
    labels_zero = rgb_sigma_zero[..., 33:]
    
    # fix non-face rgb feature
    if blend_mode == "background" or blend_mode == "both":
        rgbs[(labels_zero.max(dim=-1, keepdim=True)[1]==0).squeeze(-1), :] = rgbs_zero[(labels_zero.max(dim=-1, keepdim=True)[1]==0).squeeze(-1), :]
        sigmas[(labels_zero.max(dim=-1, keepdim=True)[1]==0).squeeze(-1), :] = sigmas_zero[(labels_zero.max(dim=-1, keepdim=True)[1]==0).squeeze(-1), :]

    if blend_mode == "hair" or blend_mode == "both":
        rgbs[(labels_zero.max(dim=-1, keepdim=True)[1]==17).squeeze(-1), :] = rgbs_zero[(labels_zero.max(dim=-1, keepdim=True)[1]==17).squeeze(-1), :]
        sigmas[(labels_zero.max(dim=-1, keepdim=True)[1]==17).squeeze(-1), :] = sigmas_zero[(labels_zero.max(dim=-1, keepdim=True)[1]==17).squeeze(-1), :]

    
    label_sigma = sigmas.clone()
    label_sigma[labels.max(dim=-1, keepdim=True)[1]==0] = 0
    
    labels = labels.reshape(batch_size, num_pixels*num_steps, 19)

    rgb_map = rgb_map.repeat(batch_size, 1, 1)
    label_rgbs = torch.bmm(labels, rgb_map).reshape(batch_size, num_pixels, num_steps, 3).clamp_(0,1)
    labels = labels.reshape(batch_size, num_pixels, num_steps, 19)
    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2)

    alphas = 1-torch.exp(-deltas * sigmas)
    label_alphas = 1-torch.exp(-deltas * label_sigma)
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-10], -2)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights_sum = weights.sum(2)
    
    label_alphas_shifted = torch.cat([torch.ones_like(label_alphas[:, :, :1]), 1-label_alphas + 1e-10], -2)
    label_weights = label_alphas * torch.cumprod(label_alphas_shifted, -2)[:, :, :-1]


    if last_back:
        weights[:, :, -1] += (1 - weights_sum)

    rgb_final = torch.sum(weights * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -2)
    label_rgb_final = torch.sum(label_weights * label_rgbs, -2)
    label_final = torch.sum(label_weights * labels, -2)

    if white_back:
        rgb_final = rgb_final + 1-weights_sum

    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor([1., 0, 0], device=rgb_final.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)

    return rgb_final, depth_final, label_final, label_rgb_final, weights



def get_initial_rays_trig(n, num_steps, device, fov, resolution, ray_start, ray_end):
    """Returns sample points, z_vals, and ray directions in camera space."""

    W, H = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                          torch.linspace(1, -1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()
    z = -torch.ones_like(x, device=device) / np.tan((2 * math.pi * fov / 360)/2)

    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1))

    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).reshape(1, num_steps, 1).repeat(W*H, 1, 1)
    points = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals

    points = torch.stack(n*[points])
    z_vals = torch.stack(n*[z_vals])
    rays_d_cam = torch.stack(n*[rays_d_cam]).to(device)

    return points, z_vals, rays_d_cam

def get_initial_depth_rays_trig(n, depth, num_steps, device, fov, resolution, ray_start, ray_end):
    """Returns sample points, z_vals, and ray directions in camera space."""
    """depth map (1, 128*128, 1)"""
    W, H = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                          torch.linspace(1, -1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()
    z = -torch.ones_like(x, device=device) / np.tan((2 * math.pi * fov / 360)/2)

    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1))

    # depth z
    z_vals = torch.zeros((n, W*H, num_steps, 1), device=device)
    norm_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).reshape(1, 1, 1, num_steps).repeat(n, W, H, 1)
    # norm_vals = depth_z_vals * (depth_z_vals>=0.9) + norm_z_vals * (depth_z_vals<0.9)
    norm_vals = norm_vals.reshape(n, -1, num_steps, 1)
    norm_points = rays_d_cam.unsqueeze(0).unsqueeze(2).repeat(n, 1, num_steps, 1) * norm_vals

    # compute more samples around depth
    # valid_depth = depth[..., 0] >= 0.88
    # invalid_depth = valid_depth.logical_not()
    depth = depth.reshape(n, W, H, 1) # 1, 128, 128, 1
    norm_z_vals = torch.linspace(ray_start+0.005, ray_end-0.005, num_steps, device=device).reshape(1, 1, 1, num_steps).repeat(n, W, H, 1)
    depth_start = (depth - 0.04).repeat(1, 1, 1, num_steps) # 1, 128, 128, 10
    depth_interval = torch.linspace(0, 0.08, num_steps, device=device).repeat(n, W, H, 1)
    depth_z_vals = depth_start + depth_interval # 1, 128, 128, 10
    depth_z_vals_new = depth_z_vals * (depth_z_vals>=0.88) + norm_z_vals * (depth_z_vals<0.88)
    depth_z_vals_new = depth_z_vals_new.reshape(n, -1, num_steps, 1)
    depth_points = rays_d_cam.unsqueeze(0).unsqueeze(2).repeat(n, 1, num_steps, 1) * depth_z_vals_new
    
    rays_d_cam = torch.stack(n*[rays_d_cam]).to(device)

    return norm_points, depth_points, norm_vals, depth_z_vals_new, rays_d_cam

def perturb_points(points, z_vals, ray_directions, device):
    distance_between_points = z_vals[:,:,1:2,:] - z_vals[:,:,0:1,:]
    offset = (torch.rand(z_vals.shape, device=device)-0.5) * distance_between_points
    z_vals = z_vals + offset

    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals


def transform_sampled_points(points, z_vals, ray_directions, pose, device):
    """Samples a camera position and maps points in camera space to world space."""

    n, num_rays, num_steps, channels = points.shape

    per_points, z_vals = perturb_points(points, z_vals, ray_directions, device)

    camera_origin, pitch, yaw = condition_camera_positions(pose, device)
    forward_vector = normalize_vecs(-camera_origin)

    cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin, device=device)

    points_homogeneous = torch.ones((per_points.shape[0], per_points.shape[1], per_points.shape[2], per_points.shape[3] + 1), device=device)
    points_homogeneous[:, :, :, :3] = per_points
    
    ori_points_homogeneous = torch.ones((points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1), device=device)
    ori_points_homogeneous[:, :, :, :3] = points

    # should be n x 4 x 4 , n x r^2 x num_steps x 4
    transformed_points = torch.bmm(cam2world_matrix, points_homogeneous.reshape(n, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)
    transformed_ori_points = torch.bmm(cam2world_matrix, ori_points_homogeneous.reshape(n, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)
    
    transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3], ray_directions.reshape(n, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, 3)

    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
    homogeneous_origins[:, 3, :] = 1
    transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]

    return transformed_points[..., :3], z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def sample_camera_positions(device, n=1, r=1, horizontal_stddev=0.3, vertical_stddev=0.155, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='gaussian'):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    """

    if mode == 'uniform':
        theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean

    elif mode == 'normal' or mode == 'gaussian':
        theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    elif mode == 'hybrid':
        if random.random() < 0.5:
            theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
            phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
        else:
            theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
            phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    elif mode == 'truncated_gaussian':
        theta = truncated_normal_(torch.zeros((n, 1), device=device)) * horizontal_stddev + horizontal_mean
        phi = truncated_normal_(torch.zeros((n, 1), device=device)) * vertical_stddev + vertical_mean

    elif mode == 'spherical_uniform':
        theta = (torch.rand((n, 1), device=device) - .5) * 2 * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
        v = ((torch.rand((n,1), device=device) - .5) * 2 * v_stddev + v_mean)
        v = torch.clamp(v, 1e-5, 1 - 1e-5)
        phi = torch.arccos(1 - 2 * v)

    else:
        # Just use the mean.
        theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean
        phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)
    output_points[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r*torch.cos(phi)

    return output_points, phi, theta # [B, 3], [B, 1], [B, 1]

def condition_camera_positions(pose, device, r=1):
    batch_size = pose.shape[0]
    phi = pose[:, 0] # pitch
    theta = pose[:, 1] # yaw

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    phi = phi.unsqueeze(-1)
    theta = theta.unsqueeze(-1)

    output_points = torch.zeros((batch_size, 3), device=device)
    output_points[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r*torch.cos(phi)

    return output_points, phi, theta # [B, 3], [B, 1], [B, 1]

def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world


def create_world2cam_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a world2cam matrix."""
    cam2world = create_cam2world_matrix(forward_vector, origin, device=device)
    world2cam = torch.inverse(cam2world)
    return world2cam


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
        # u = torch.linspace(0, 1, N_importance, device=bins.device)
        # u = u.repeat(N_rays, 1)
        # u += (torch.rand(N_rays, N_importance, device=bins.device) - 0.5) / (N_importance - 1)
        # u = torch.clamp(u, 0, 1)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


