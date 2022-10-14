import argparse
import math
import os
import glob
import functools
import torch
import numpy as np
import numpy as np
import cv2
import curriculums
from generators import generators
from torchvision.utils import make_grid
from decoders import decoders
from generators import generators
from torch_ema import ExponentialMovingAverage
import scipy.io as scio

def drive_exp_coeff(batch, drive_exp, vae_id, vae_gamma, z_dim, id_dim, gamma_dim, device):

    latent_z = torch.randn(1, z_dim, device=device).repeat(batch,1)
    latent_id = torch.randn(1, id_dim, device=device).repeat(batch,1)
    latent_gamma = torch.randn(1, gamma_dim, device=device).repeat(batch,1)

    sample_id = vae_id.decode(latent_id)
    sample_gamma = vae_gamma.decode(latent_gamma)
    sample_exp = drive_exp[:batch, :] / 4.0

    latent = torch.cat([latent_z, sample_id, sample_exp, sample_gamma], dim=1)

    return latent, latent_z, sample_id, sample_exp, sample_gamma

parser = argparse.ArgumentParser()
# parser.add_argument('--seeds', type=int, default=32)
parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
parser.add_argument('--load_dir', type=str)
parser.add_argument('--output_dir', type=str, default='vids')
parser.add_argument('--ray_step_multiplier', type=int, default=2)
parser.add_argument('--curriculum', type=str, default='CelebA')
parser.add_argument('--psi', type=float, default=0.7)
parser.add_argument('--blend_mode', type=str, default="both")
opt = parser.parse_args()

metadata = getattr(curriculums, opt.curriculum)
metadata = curriculums.extract_metadata(metadata, 0)
metadata['nerf_noise'] = 0
metadata['ray_end'] = 1.15
metadata['num_steps'] = metadata['num_steps'] * opt.ray_step_multiplier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tensor_to_img(img, **kwargs):
    grid = make_grid(img, **kwargs)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return ndarr[:, :, ::-1]

def reversed_cmp(x,y):
    x_ind = int(x.split('-')[-1].split('.')[0])
    y_ind = int(y.split('-')[-1].split('.')[0])
    if x_ind>y_ind:
        return 1
    if x_ind<y_ind:
        return -1
    return 0

video_coeff_dir = "data/obama/mat/*.mat"
mat_path = sorted(glob.glob(video_coeff_dir), key=functools.cmp_to_key(reversed_cmp))

z_dim = metadata['z_dim']
id_dim = metadata['id_dim']
exp_dim = metadata['exp_dim']
gamma_dim = metadata['gamma_dim']

vae_dir = metadata['vae_path']
bfm_dir = metadata['bfm_path']
face_dir = metadata['face_path']
vae_id = torch.load(os.path.join(vae_dir, '14000_vae_id.pth'), map_location=device)
vae_exp = torch.load(os.path.join(vae_dir, '20000_vae_exp.pth'), map_location=device)
vae_gamma = torch.load(os.path.join(vae_dir, '17000_vae_gamma.pth'), map_location=device)

vae_id.eval()
vae_exp.eval()
vae_gamma.eval()
latent_dim = 379

decoder = getattr(decoders, metadata['model'])
generator = getattr(generators, metadata['generator'])(decoder, metadata['render_size'], metadata['img_size'], metadata['plane_reso'], latent_dim, metadata['w_dim'], metadata['c_dim'],device).cuda()
ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
generator.load_state_dict(torch.load(os.path.join(opt.load_dir, 'generator.pth'), map_location=device))
ema.load_state_dict(torch.load(os.path.join(opt.load_dir, 'ema.pth'), map_location=device))
ema.copy_to(generator.parameters())
generator.set_device(device)
generator.eval()

output_dir = os.path.join(opt.output_dir)
os.makedirs(output_dir, exist_ok=True)

drive_id = torch.zeros((len(mat_path), 80)).to(device)
drive_tex = torch.zeros((len(mat_path), 80)).to(device)
drive_exp = torch.zeros((len(mat_path), 64)).to(device)
drive_gamma = torch.zeros((len(mat_path), 27)).to(device)
drive_pose = torch.zeros((len(mat_path), 3)).to(device)
for i, mat_name in enumerate(mat_path):
    mat = scio.loadmat(mat_name)
    drive_id[i] = torch.tensor(mat['id']).squeeze(0).to(device)
    drive_tex[i] = torch.tensor(mat['tex']).squeeze(0).to(device)
    drive_exp[i] = torch.tensor(mat['exp']).squeeze(0).to(device)
    drive_gamma[i] = torch.tensor(mat['gamma']).squeeze(0).to(device)
    drive_pose[i] = torch.tensor(mat['angle']).squeeze(0).to(device)

print("load exp done")
num_frame = 36
trajectory = []
k=0
for t in np.linspace(0, 1, num_frame):
    pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/2
    yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
    trajectory.append((k, pitch, yaw))
    k+=1

# for seed in range(opt.seeds, opt.seeds+1):
for seed in opt.seeds:
    img_frames = []
    fine_name = f"{seed}_interp.mp4"
    torch.manual_seed(seed)
    fixed_latent, _, sample_id, sample_exp, sample_gamma = drive_exp_coeff(num_frame, drive_exp, vae_id, vae_gamma, z_dim, id_dim, gamma_dim, device)
    with torch.no_grad():
        img_frames = generator.render_forward(opt.blend_mode, fixed_latent, trajectory, noise_mode='const', truncation_psi=opt.psi, **metadata)

    height, width, layers = img_frames[0].shape
    fine_video = cv2.VideoWriter(os.path.join(output_dir, fine_name), cv2.VideoWriter_fourcc("m", "p", "4", "v"), 24, (width, height))
    for fine_frame in img_frames:
        fine_video.write(np.array(fine_frame))

    fine_video.release()