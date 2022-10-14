
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from stylegan.network import FullyConnectedLayer

class STYLEMLP_mask(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, lr_mlp=0.01, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.base_network = nn.Sequential(
            FullyConnectedLayer(input_dim, hidden_dim, bias_init=1),
            torch.nn.Softplus(),
        )
        self.image_head = FullyConnectedLayer(hidden_dim, 33, bias_init=1)
        self.mask_head = FullyConnectedLayer(hidden_dim, 19, bias_init=1)

    def forward(self, input, **kwargs):
        batch_size, n_points, dim = input.shape
        x = input.reshape(-1, dim).contiguous()
        x = self.base_network(x)
        x_img = self.image_head(x)
        x_mask = self.mask_head(x)
        x_img = x_img.reshape(batch_size, n_points, -1).contiguous()
        x_mask = x_mask.reshape(batch_size, n_points, -1).contiguous()
        rgb = x_img[:,:,:3]
        other = x_img[:,:,3:32]
        sigma = x_img[:,:,32:33]
        label = torch.softmax(x_mask, dim=-1) # [B, N, 19]
        rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = F.softplus(sigma)
        # label = F.normalize(label, p=1, dim=-1).to(dtype=torch.float)
        # label = (label == label.max(dim=-1, keepdim=True)[0]).to(dtype=torch.float)
        result = torch.cat([rgb, other, sigma, label], dim=-1)
        return result