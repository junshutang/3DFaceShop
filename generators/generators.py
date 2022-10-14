"""Implicit generator for 3D volumes"""

import random
import torch.nn as nn
import torch
import random
from .volumetric_rendering import *
from stylegan.network import *
from tqdm import tqdm
from torchvision.utils import make_grid

def tensor_to_img(img, **kwargs):
    grid = make_grid(img, **kwargs)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return ndarr[:, :, ::-1]

    
def normalize_coordinate(p, padding=0.1, plane='xz'):
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane =='xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6) # (-0.5, 0.5)
    xy_new = xy_new + 0.5 # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new

def coordinate2index(x, reso, coord_type='2d'):

    x = (x * reso).long()
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index

def norm_img(img, low, high):
    img = img.clamp_(min=low, max=high)
    img = img.sub_(low).div_(max(high - low, 1e-5))
    return img

class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor

def sample_dict(sample_id, sample_exp, sample_gamma, device):
    batch_size = sample_id.shape[0]
    id_coeffs = sample_id[:, :80] * 4.2
    tex_coeffs = sample_id[:, 80:] * 9.0
    exp_coeffs = sample_exp * 4.0
    gammas = sample_gamma * 0.85
    return {
        'id': id_coeffs.to(device),
        'exp': exp_coeffs.to(device),
        'tex': tex_coeffs.to(device),
        'angle': torch.zeros((batch_size, 3)).to(device),
        'gamma': gammas.to(device),
        'trans': torch.zeros((batch_size, 3)).to(device),
    }

class TriplaneImplicitGenerator3d(nn.Module):

    def __init__(self, siren, render_size, img_size,  plane_reso, latent_dim, w_dim, c_dim, device):
        super().__init__()
        self.latent_dim = latent_dim
        self.render_size = render_size
        self.encoder = StyleGAN_Generator(latent_dim, c_dim, w_dim, plane_reso)
        self.decoder = siren(input_dim=32, device=None)
        self.sr = SRModel(render_size, img_size)
        self.epoch = 0
        self.step = 0

    def set_device(self, device):
        self.device = device
        self.encoder.device = device
        self.decoder.device = device

    def generate_plane_features(self, p, triplane, scale=0.185):

        xyz = p
        xyz_new = xyz / (scale + 10e-6) # (-1, 1)
        # f there are outliers out of the range
        if xyz_new.max() >= 1:
            xyz_new[xyz_new >= 1] = 1 - 10e-6
        if xyz_new.min() < -1:
            xyz_new[xyz_new < -1] = -1.0 + 10e-6
        c_plane = {}
        channel_dim = int(triplane.size(1) / 3)
        c_plane['xz'] = triplane[:, 0:channel_dim, :, :] # [3, 32, 256, 256]
        c_plane['xy'] = triplane[:, channel_dim:channel_dim*2, :, :]
        c_plane['yz'] = triplane[:, channel_dim*2:channel_dim*3, :, :]

        xyz_new = xyz_new.unsqueeze(-2) # [B, N, 1, 3]

        cxz = F.grid_sample(c_plane['xz'], xyz_new[:,:,:,[0,2]], padding_mode="border", align_corners=True, mode='bilinear').squeeze(-1)
        cxy = F.grid_sample(c_plane['xy'], xyz_new[:,:,:,[0,1]], padding_mode="border", align_corners=True, mode='bilinear').squeeze(-1)
        cyz = F.grid_sample(c_plane['yz'], xyz_new[:,:,:,[1,2]], padding_mode="border", align_corners=True, mode='bilinear').squeeze(-1)

        c_out = cxz + cxy + cyz

        return c_out

    def forward(self, latent_code, gt_pose, fov, ray_start, ray_end, num_steps, hierarchical_sample, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        gt_pose is labeled pose, swp_pose is swapped pose
        """

        batch_size = latent_code.shape[0]
        img_size = self.render_size
        
        triplane, ws = self.encoder(latent_code, gt_pose) # 256*256*96
        _, pitch, yaw = sample_camera_positions(n=gt_pose.shape[0], r=1, horizontal_stddev=0.3, vertical_stddev=0.155, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, device=self.device, mode='gaussian')
        swp_pose = torch.cat([pitch, yaw], -1)
        render_pose = torch.zeros((batch_size, gt_pose.shape[1]), device=self.device)
        for i in range(len(gt_pose)):
            p = random.random() # [0,1)
            if p < 0.5:
                render_pose[i] = gt_pose[i]
            else:
                render_pose[i] = swp_pose[i]
        
        # Generate initial camera rays and sample points.
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, render_pose, device=self.device)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

        # Tri-plane 256*256*32*3
        transformed_point_feature = self.generate_plane_features(transformed_points, triplane).transpose(1, 2) #[B, 3, 256, 256. 32], [B, num, 3]
        # Model prediction on course points
        coarse_output = self.decoder(transformed_point_feature).reshape(batch_size, img_size * img_size, num_steps, 52)
        # Re-sample fine points alont camera rays, as described in NeRF
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, _, weights = fancy_integration_label(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'])

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                #### Start new importance sampling
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                #### end new importance sampling

            # fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
            fine_point_feature = self.generate_plane_features(fine_points, triplane).transpose(1,2)
            # Model prediction on re-sampled find points
            fine_output = self.decoder(fine_point_feature).reshape(batch_size, img_size * img_size, -1, 52)

            # Combine course and fine points
            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 52))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals

        # Create images with NeRF
        pixels_feature, depth, pixels_label, weights = fancy_integration_label(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'])
        
        sigma = all_outputs[..., 32:33].squeeze(-1)
        sigma_fine = sigma[:, :sigma.shape[1]//2]
        sigma_coarse = sigma[:, sigma.shape[1]//2:]
        
        pixels_feature = pixels_feature.reshape((batch_size, img_size, img_size, 32))
        pixels_feature = pixels_feature.permute(0, 3, 1, 2)

        pixels_label = pixels_label.reshape((batch_size, img_size, img_size, 19)).permute(0, 3, 1, 2)
        
        raw_img = pixels_feature[:,:3,:,:].contiguous() * 2 - 1
        
        final_img = self.sr(pixels_feature, ws)
        
        return raw_img, pixels_label, final_img, sigma, sigma_fine, sigma_coarse, torch.cat([pitch, yaw], -1)
    
    def staged_forward(self, latent_code, render_pose, fov, ray_start, ray_end, num_steps, hierarchical_sample, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        gt_pose is labeled pose, swp_pose is swapped pose
        """

        batch_size = latent_code.shape[0]
        img_size = self.render_size
        triplane, ws = self.encoder(latent_code, render_pose)
        
        # Generate initial camera rays and sample points.
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, render_pose, device=self.device)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

        # Tri-plane 256*256*32*3
        transformed_point_feature = self.generate_plane_features(transformed_points, triplane).transpose(1, 2) #[B, 3, 256, 256. 32], [B, num, 3]
        # Model prediction on course points
        coarse_output = self.decoder(transformed_point_feature).reshape(batch_size, img_size * img_size, num_steps, 52)
        # Re-sample fine points alont camera rays, as described in NeRF
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, _, weights = fancy_integration_label(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'])

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                #### Start new importance sampling
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                #### end new importance sampling

            # fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
            fine_point_feature = self.generate_plane_features(fine_points, triplane).transpose(1,2)
            # Model prediction on re-sampled find points
            fine_output = self.decoder(fine_point_feature).reshape(batch_size, img_size * img_size, -1, 52)

            # Combine course and fine points
            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 52))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals

        # Create images with NeRF
        pixels_feature, depth, pixels_label, weights = fancy_integration_label(all_outputs, all_z_vals, device=self.device, return_rgb=False, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'])

        pixels_feature = pixels_feature.reshape((batch_size, img_size, img_size, 32))
        pixels_feature = pixels_feature.permute(0, 3, 1, 2)
        
        final_img = self.sr(pixels_feature, ws)
        
        return final_img, torch.cat([pitch, yaw], -1)

    def shape_forward(self, latent_code, render_pose, noise_mode, truncation_psi, fov, ray_start, ray_end, num_steps, hierarchical_sample, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        gt_pose is labeled pose, swp_pose is swapped pose
        """

        batch_size = latent_code.shape[0]
        img_size = self.render_size
        front_pose = math.pi/2 * torch.ones((batch_size, 2)).to(latent_code.device)
        triplane, ws = self.encoder(latent_code, front_pose, noise_mode, truncation_psi=truncation_psi)
        # Generate initial camera rays and sample points.
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, render_pose, device=self.device)
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

        # Tri-plane 256*256*32*3
        transformed_point_feature = self.generate_plane_features(transformed_points, triplane).transpose(1, 2) #[B, 3, 256, 256. 32], [B, num, 3]
        # Model prediction on course points
        coarse_output = self.decoder(transformed_point_feature).reshape(batch_size, img_size * img_size, num_steps, 52)
        # Re-sample fine points alont camera rays, as described in NeRF
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, _, weights = fancy_integration_label(coarse_output, z_vals, device=self.device, return_rgb=False, clamp_mode=kwargs['clamp_mode'])

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                #### Start new importance sampling
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                #### end new importance sampling

            # fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
            fine_point_feature = self.generate_plane_features(fine_points, triplane).transpose(1,2)
            # Model prediction on re-sampled find points
            fine_output = self.decoder(fine_point_feature).reshape(batch_size, img_size * img_size, -1, 52)

            # Combine course and fine points
            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 52))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals

        # Create images with NeRF
        pixels_feature, depth, raw_label, pixels_label, weights  = fancy_integration_label(all_outputs, all_z_vals, device=self.device, return_rgb=True, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'])

        raw_label = raw_label.reshape((batch_size, img_size, img_size, 19)).permute(0, 3, 1, 2)
        
        pixels_feature = pixels_feature.reshape((batch_size, img_size, img_size, 32))
        pixels_feature = pixels_feature.permute(0, 3, 1, 2)

        pixels_label = pixels_label.reshape((batch_size, img_size, img_size, 3)).permute(0, 3, 1, 2)
        raw_img = pixels_feature[:,:3,:,:].contiguous() * 2 - 1
        pixels_label = pixels_label.contiguous() * 2 - 1
        
        final_img = self.sr(pixels_feature, ws)

        return raw_img, raw_label, pixels_label, final_img, torch.cat([pitch, yaw], -1)

    def render_forward(self, blend_mode, latent_code, traj, noise_mode, truncation_psi, fov, ray_start, ray_end, num_steps, hierarchical_sample, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        gt_pose is labeled pose, swp_pose is swapped pose
        """

        batch_size = 1
        img_size = self.render_size
        front_pose = math.pi/2 * torch.ones((batch_size, 2)).to(latent_code.device)
        img_frames = []
        for k, pitch, yaw in tqdm(traj):
            fixed_pose = torch.tensor((pitch, yaw)).unsqueeze(0).to(latent_code.device)
            if k==0: 
                triplane_zero, ws = self.encoder(latent_code[k].unsqueeze(0), front_pose, noise_mode, truncation_psi=truncation_psi)
                # Generate initial camera rays and sample points.
                with torch.no_grad():
                    points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
                    transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, fixed_pose, device=self.device)

                    transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
                    transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
                    transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
                    transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

                    if lock_view_dependence:
                        transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                        transformed_ray_directions_expanded[..., -1] = -1

                # Tri-plane 256*256*32*3
                transformed_point_feature = self.generate_plane_features(transformed_points, triplane_zero).transpose(1, 2) #[B, 3, 256, 256. 32], [B, num, 3]
                # Model prediction on course points
                coarse_output = self.decoder(transformed_point_feature).reshape(batch_size, img_size * img_size, num_steps, 52)
                # Re-sample fine points alont camera rays, as described in NeRF
                if hierarchical_sample:
                    with torch.no_grad():
                        transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                        _, _, _, weights = fancy_integration_label(coarse_output, z_vals, device=self.device, return_rgb=False, clamp_mode=kwargs['clamp_mode'])

                        weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                        #### Start new importance sampling
                        z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                        z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                        fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                        num_steps, det=False).detach()
                        fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                        fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                        fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)

                        if lock_view_dependence:
                            transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                            transformed_ray_directions_expanded[..., -1] = -1
                        #### end new importance sampling

                    # fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                    fine_point_feature = self.generate_plane_features(fine_points, triplane_zero).transpose(1,2)
                    # Model prediction on re-sampled find points
                    fine_output = self.decoder(fine_point_feature).reshape(batch_size, img_size * img_size, -1, 52)

                    # Combine course and fine points
                    all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
                    all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                    _, indices = torch.sort(all_z_vals, dim=-2)
                    all_z_vals = torch.gather(all_z_vals, -2, indices)
                    all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 52))
                else:
                    all_outputs = coarse_output
                    all_z_vals = z_vals

                # Create images with NeRF
                pixels_feature, _, _, _, weights  = fancy_integration_label(all_outputs, all_z_vals, device=self.device, return_rgb=True, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'])
                pixels_feature = pixels_feature.reshape((batch_size, img_size, img_size, 32))
                pixels_feature = pixels_feature.permute(0, 3, 1, 2)
                final_img = self.sr(pixels_feature, ws, noise_mode)
            else:
                triplane, ws = self.encoder(latent_code[k].unsqueeze(0), front_pose, noise_mode, truncation_psi=truncation_psi)
                # Generate initial camera rays and sample points.
                with torch.no_grad():
                    points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
                    transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, fixed_pose, device=self.device)

                    transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
                    transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
                    transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
                    transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

                    if lock_view_dependence:
                        transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                        transformed_ray_directions_expanded[..., -1] = -1

                # Tri-plane 256*256*32*3
                transformed_point_feature = self.generate_plane_features(transformed_points, triplane).transpose(1, 2) #[B, 3, 256, 256. 32], [B, num, 3]
                course_point_zero_feature = self.generate_plane_features(transformed_points, triplane_zero).transpose(1, 2) #[B, 3, 256, 256. 32], [B, num, 3]
                    
                # Model prediction on course points
                coarse_output = self.decoder(transformed_point_feature).reshape(batch_size, img_size * img_size, num_steps, 52)
                coarse_zero_output = self.decoder(course_point_zero_feature).reshape(batch_size, img_size * img_size, num_steps, 52)
                
                # Re-sample fine points alont camera rays, as described in NeRF
                if hierarchical_sample:
                    with torch.no_grad():
                        transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                        _, _, _, weights = fancy_integration_label(coarse_output, z_vals, device=self.device, return_rgb=False, clamp_mode=kwargs['clamp_mode'])

                        weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                        #### Start new importance sampling
                        z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                        z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                        fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                        num_steps, det=False).detach()
                        fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                        fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                        fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)

                        if lock_view_dependence:
                            transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                            transformed_ray_directions_expanded[..., -1] = -1
                        #### end new importance sampling

                    # fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                    fine_point_feature = self.generate_plane_features(fine_points, triplane).transpose(1,2)
                    fine_point_zero_feature = self.generate_plane_features(fine_points, triplane_zero).transpose(1,2)
                    
                    # Model prediction on re-sampled find points
                    fine_output = self.decoder(fine_point_feature).reshape(batch_size, img_size * img_size, -1, 52)
                    fine_zero_output = self.decoder(fine_point_zero_feature).reshape(batch_size, img_size * img_size, -1, 52)
                    
                    # Combine course and fine points
                    all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
                    all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                    _, indices = torch.sort(all_z_vals, dim=-2)
                    all_z_vals = torch.gather(all_z_vals, -2, indices)
                    all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 52))
                
                    all_zero_outputs = torch.cat([fine_zero_output, coarse_zero_output], dim = -2)
                    all_zero_outputs = torch.gather(all_zero_outputs, -2, indices.expand(-1, -1, -1, 52))
                
                else:
                    all_outputs = coarse_output
                    all_z_vals = z_vals

                # Create images with NeRF
                pixels_feature, _, _, _, weights = fancy_integration_label_blend(blend_mode, all_outputs, all_zero_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'])
                pixels_feature = pixels_feature.reshape((batch_size, img_size, img_size, 32))
                pixels_feature = pixels_feature.permute(0, 3, 1, 2)
                final_img = self.sr(pixels_feature, ws, noise_mode)
            
            img_frames.append(tensor_to_img(final_img, nrow=4, normalize=True, value_range=(-1,1)))

        return img_frames

        