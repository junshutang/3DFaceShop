"""Discrimators used in pi-GAN"""

import torch.nn as nn
import torch.nn.functional as F
from stylegan.network import *

def down_up(img, scale):
    down_img = F.interpolate(img, scale_factor=1.0/scale)
    out_img = F.interpolate(down_img, scale_factor=scale, mode='bicubic')
    return out_img

def up(img, scale):
    out_img = F.interpolate(img, scale_factor=scale, mode='bicubic')
    return out_img

class VAEDiscriminator(nn.Module):
    
    def __init__(self, in_channels, latent_dim, ch_dim, depth):
        super().__init__()
        self.input_channel = in_channels
        self.latent_dim = latent_dim
        self.ch_dim = ch_dim
        self.dis_depth = depth
        self.epoch = 0
        self.step = 0

        modules = []
        # Build Encoder
        input_dim = self.input_channel
        for _ in range(self.dis_depth):
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, self.ch_dim),
                    nn.BatchNorm1d(self.ch_dim),
                    nn.LeakyReLU())
            )
            input_dim = self.ch_dim

        self.model = nn.Sequential(*modules)
        self.final_layer = nn.Linear(self.ch_dim, 1)

    def forward(self, x):
        result = self.model(x)
        output = F.sigmoid(self.final_layer(result))
        
        return output

class SDiscriminator(nn.Module):

    def __init__(self, img_resolution, img_channels, latent_dim):
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.discriminator_img = StyleGAN_Discriminator(img_resolution, img_channels, latent_dim, return_latent=True)

    def set_device(self, device):
        self.device = device
        self.discriminator_img.device = device

    def forward(self, cat_image, gen_pose):

        pred_img, pred_img_pose, pred_latent = self.discriminator_img(cat_image, gen_pose)

        return pred_img, pred_img_pose, pred_latent
