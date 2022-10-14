"""
Contains code for logging approximate FID scores during training.
If you want to output ground-truth images from the training dataset, you can
run this file as a script.
"""

import os
import shutil
import torch
import copy
import argparse
import math

from torchvision.utils import save_image
from pytorch_fid import fid_score
from tqdm import tqdm
from generators.volumetric_rendering import *

import datasets

def sample_latent_coeff(batch, vae_id, vae_exp, vae_gamma, z_dim, id_dim, exp_dim, gamma_dim, device, pair=None):
    if pair is None:
        latent_z = torch.randn(batch, z_dim, device=device)
        latent_id = torch.randn(batch, id_dim, device=device)
        latent_exp = torch.randn(batch, exp_dim, device=device)
        latent_gamma = torch.randn(batch, gamma_dim, device=device)
    elif pair == 'exp':
        latent_z = torch.randn(1, z_dim, device=device).repeat(batch,1)
        latent_id = torch.randn(1, id_dim, device=device).repeat(batch,1)
        latent_exp = torch.randn(batch, exp_dim, device=device)
        latent_gamma = torch.randn(1, gamma_dim, device=device).repeat(batch,1)
    elif pair == 'id':
        latent_z = torch.randn(1, z_dim, device=device).repeat(batch,1)
        latent_id = torch.randn(batch, id_dim, device=device)
        latent_exp = torch.randn(1, exp_dim, device=device).repeat(batch,1)
        latent_gamma = torch.randn(1, gamma_dim, device=device).repeat(batch,1)
    elif pair == 'gamma':
        latent_z = torch.randn(1, z_dim, device=device).repeat(batch,1)
        latent_id = torch.randn(1, id_dim, device=device).repeat(batch,1)
        latent_exp = torch.randn(1, exp_dim, device=device).repeat(batch,1)
        latent_gamma = torch.randn(batch, gamma_dim, device=device)

    sample_id = vae_id.decode(latent_id)
    sample_exp = vae_exp.decode(latent_exp)
    sample_gamma = vae_gamma.decode(latent_gamma)

    latent = torch.cat([latent_z, sample_id, sample_exp, sample_gamma], dim=1)

    return latent, latent_z, sample_id, sample_exp, sample_gamma


def output_real_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    for i in range(num_imgs//batch_size):
        real_imgs, _ = next(dataloader)
        for img in real_imgs:
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
            img_counter += 1
    
def setup_evaluation(dataset_name, generated_dir, metadata, num_imgs=5000):
    # Only make real images if they haven't been made yet
    real_dir = os.path.join('EvalImages', dataset_name + '_real_images_' + str(metadata['img_size']))
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        dataloader, CHANNELS = datasets.get_dataset(dataset_name, **metadata)
        print('outputting real images...')
        output_real_images(dataloader, num_imgs, real_dir)
        print('...done')
    else:
        print('Real exist!')

    if generated_dir is not None:
        os.makedirs(generated_dir, exist_ok=True)
    
    return real_dir

def output_images(generator, input_metadata, vae_id, vae_exp, vae_gamma, output_dir, num_imgs=5000):
    metadata = copy.deepcopy(input_metadata)
    metadata['img_size'] = input_metadata['img_size']
    metadata['batch_size'] = 4
    z_dim = metadata['z_dim']
    id_dim = metadata['id_dim']
    exp_dim = metadata['exp_dim']
    gamma_dim = metadata['gamma_dim']
    
    generator.eval()
    img_counter = 0
    pbar = tqdm("generating images", total = num_imgs)
    with torch.no_grad():
        while img_counter < num_imgs:
            latent, _, _, _, _ = sample_latent_coeff(metadata['batch_size'], vae_id, vae_exp, vae_gamma, z_dim, id_dim, exp_dim, gamma_dim, generator.device, pair=None)
            _, pitch, yaw = sample_camera_positions(n=metadata['batch_size'], r=1, horizontal_stddev=0.3, vertical_stddev=0.155, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, device=generator.device, mode='truncated_gaussian')
            render_pose = torch.cat([pitch, yaw], -1)
            _, _, _, generated_imgs, _ = generator.shape_forward(latent, render_pose, noise_mode='random', truncation_psi=1, **metadata)
            
            for img in generated_imgs:
                save_image(img, os.path.join(output_dir, f'{img_counter:0>5}.jpg'), normalize=True, value_range=(-1, 1))
                img_counter += 1
                pbar.update(1)
    pbar.close()


def calculate_fid(dataset_name, generated_dir, target_size=256):
    real_dir = os.path.join('../EvalImages', dataset_name + '_real_images_' + str(target_size))
    fid = fid_score.calculate_fid_given_paths([real_dir, generated_dir], 96, 'cuda', 2048)
    torch.cuda.empty_cache()

    return fid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CelebA')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--num_imgs', type=int, default=8000)

    opt = parser.parse_args()

    real_images_dir = setup_evaluation(opt.dataset, None, target_size=opt.img_size, num_imgs=opt.num_imgs)
