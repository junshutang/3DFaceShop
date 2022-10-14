"""Train control. Supports distributed training."""

import argparse
import os
import numpy as np
import math

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from generators import generators
from discriminators import discriminators
from stylegan.network import *
from decoders import decoders
import fid_evaluation
import datasets
import curriculums
from tqdm import tqdm
from stylegan.op import conv2d_gradfix
from torch_ema import ExponentialMovingAverage
import warnings
from skimage import transform as trans
from generators.volumetric_rendering import *
from warp_utils import *
from facenets.bfm import ParametricFaceModel
from facenets.render import MeshRenderer
import facenets.networks as networks
from facenets.hair import BiSeNet
from kornia.geometry import warp_affine


warnings.filterwarnings('ignore')

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    dist.destroy_process_group()

def down_up(img, scale):
    down_img = nn.functional.interpolate(img, scale_factor=1.0/scale)
    out_img = nn.functional.interpolate(down_img, scale_factor=scale, mode='bicubic')
    return out_img

def up(img, scale):
    out_img = nn.functional.interpolate(img, scale_factor=scale, mode='bicubic')
    return out_img

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def norm_img(img, low, high):
    img = img.clamp_(min=low, max=high)
    img = img.sub_(low).div_(max(high - low, 1e-5))
    return img

def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None): # pylint: disable=redefined-builtin
    assert isinstance(input, torch.Tensor)
    if posinf is None:
        posinf = torch.finfo(input.dtype).max
    if neginf is None:
        neginf = torch.finfo(input.dtype).min
    assert nan == 0
    return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)

def resize_n_crop(image, M, dsize=112):
    # image: (b, c, h, w)
    # M   :  (b, 2, 3)
    return warp_affine(image, M, dsize=(dsize, dsize))

# utils for face reconstruction
def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

# utils for face recognition model
def estimate_norm(lm_68p, H):
    # from https://github.com/deepinsight/insightface/blob/c61d3cd208a603dfa4a338bd743b320ce3e94730/recognition/common/face_align.py#L68
    """
    Return:
        trans_m            --numpy.array  (2, 3)
    Parameters:
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        H                  --int/float , image height
    """
    lm = extract_5p(lm_68p)
    lm[:, -1] = H - 1 - lm[:, -1]
    tform = trans.SimilarityTransform()
    src = np.array([
        [97.829911, 127.7792227],
        [181.0531819, 126.3381292],
        [139.5739672, 179.0006314],
        [102.6964111, 216.7013791],
        [179.306845, 215.3732908]], dtype=np.float32)
    src = src * H / 286.0
    tform.estimate(lm, src)
    M = tform.params
    if np.linalg.det(M) == 0:
        M = np.eye(3)

    return M[0:2, :]

def estimate_norm_torch(lm_68p, H):
    lm_68p_ = lm_68p.detach().cpu().numpy()
    M = []
    for i in range(lm_68p_.shape[0]):
        M.append(estimate_norm(lm_68p_[i], H))
    M = torch.tensor(np.array(M), dtype=torch.float32).to(lm_68p.device)
    return M

def setup_model(face_dir, bfm_dir, device):
    camera_d = 10.0
    focal = 1015.0
    center = 112.0
    z_near = 5.0
    z_far = 15.0

    net_recon = networks.define_net_recon(
            net_recon='resnet50', use_last_fc=False, init_path=None
        ).to(device)
    load_path = os.path.join(face_dir, 'recon_model/epoch_20.pth')
    state_dict = torch.load(load_path, map_location=device)
    print('loading the model from %s' % load_path)
    if isinstance(net_recon, torch.nn.DataParallel):
        net_recon = net_recon.module
    net_recon.load_state_dict(state_dict['net_recon'])
    net_recon.eval()
    net_recog = networks.define_net_recog(
                net_recog='r50', pretrained_path=os.path.join(face_dir, 'recog_model/ms1mv3_arcface_r50_fp16/backbone.pth')
                ).to(device)
    net_recog.eval()
    face_model = ParametricFaceModel(
            bfm_folder=bfm_dir, camera_distance=camera_d, focal=focal, center=center,
            is_train=False, default_name='BFM_model_front.mat'
        )
    fov = 2 * np.arctan(center / focal) * 180 / np.pi
    renderer = MeshRenderer(
            rasterize_fov=fov, znear=z_near, zfar=z_far, rasterize_size=int(2 * center)
        )
    return net_recon, net_recog, face_model, renderer

def face_parsing(parser_model, input_img):
    parser_model.eval()
    test_imgs = input_img.clone()
    test_imgs = nn.functional.interpolate(test_imgs, size=(512, 512), mode='bicubic')
    test_imgs = norm_img(test_imgs, -1, 1).clamp_(0,1) # map to (0, 1)
    parsing_map = parser_model(test_imgs)[0]
    face_mask = parsing_map.argmax(dim=1).unsqueeze(1).float()
    face_mask = nn.functional.interpolate(face_mask, size=(128, 128))
    face_mask = face_mask.long()
    
    return face_mask

def coeff2face(face_model, renderer, coef_dict, pose, device, return_ver=False):
    
    face_model.to(device)
    coef_dict['angle'] = pose.to(device)
    pred_vertex, pred_tex, pred_color, pred_lm = face_model.compute_for_render(coef_dict)
    pred_mask, _, pred_face = renderer(pred_vertex, face_model.face_buf, feat=pred_color)
    if return_ver:
        return pred_vertex, pred_face, pred_mask, pred_lm
    else:
        return pred_face, pred_mask, pred_lm

def reconstruct_img(net_recon, face_model, fake_imgs):
    output_coeff = net_recon(fake_imgs)
    pred_coeffs_dict = face_model.split_coeff(output_coeff)
    _, _, _, pred_lm = face_model.compute_for_render(pred_coeffs_dict)
    return pred_lm, pred_coeffs_dict

def setup_dict(co_dict, device):
    id_coeffs = co_dict["id"].squeeze(1).to(device)
    exp_coeffs = co_dict["exp"].squeeze(1).to(device)
    tex_coeffs = co_dict["tex"].squeeze(1).to(device)
    angles = co_dict["angle"].squeeze(1).to(device)
    gammas = co_dict["gamma"].squeeze(1).to(device)
    translations = co_dict["trans"].squeeze(1).to(device)
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }

def rand_sample(mat, randint):
    id_coeffs = mat["id"][randint]
    exp_coeffs = mat["exp"][randint]
    tex_coeffs = mat["tex"][randint]
    angles = mat["angle"][randint]
    gammas = mat["gamma"][randint]
    translations = mat["trans"][randint]
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }

def sample_dict(sample_id, sample_exp, sample_gamma, face_pose, device):
    batch_size = sample_id.shape[0]
    id_coeffs = sample_id[:, :80] * 4.2
    tex_coeffs = sample_id[:, 80:] * 9.0
    exp_coeffs = sample_exp * 4.0
    gammas = sample_gamma * 0.85
    return {
        'id': id_coeffs.to(device),
        'exp': exp_coeffs.to(device),
        'tex': tex_coeffs.to(device),
        'angle': face_pose.to(device),
        'gamma': gammas.to(device),
        'trans': torch.tensor([0, -0.1, 0]).unsqueeze(0).repeat(batch_size, 1).to(device)
    }
    
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

def compute_imitative(net_recon, net_recog, face_model, fake_imgs, pred_face, face_mask, face_coeff, face_lm):
    
    fine_imgs_256 = fake_imgs.clone()
    fine_imgs = fake_imgs.clone()
    render_imgs = pred_face.clone()
    fine_imgs = nn.functional.interpolate(fine_imgs, size=(224, 224), mode='bicubic')
    fine_imgs_256 = nn.functional.interpolate(fine_imgs_256, size=(256, 256), mode='bicubic')
    fine_imgs = norm_img(fine_imgs, -1, 1) # map to (0, 1)
    fine_imgs_256 = norm_img(fine_imgs_256, -1, 1) # map to (0, 1)

    # l2_loss = nn.MSELoss()(fine_imgs * face_mask, face_img)
    l2_loss = torch.sqrt(1e-8 + torch.sum((render_imgs - fine_imgs) ** 2, dim=1, keepdims=True)) * face_mask
    l2_loss = torch.sum(l2_loss) / torch.max(torch.sum(face_mask), torch.tensor(1.0).to(face_mask.device))

    net_recog.eval()
    render_face = fine_imgs * (1-face_mask) + render_imgs * face_mask
    render_face = nn.functional.interpolate(render_face, size=(256, 256), mode='bicubic')
    
    fine_imgs_256 = resize_n_crop(fine_imgs_256, trans_m, dsize=fine_imgs_256.shape[-2]).clamp_(0,1)
    render_face = resize_n_crop(render_face, trans_m, dsize=render_face.shape[-2]).clamp_(0,1)   
    trans_m = estimate_norm_torch(face_lm * 256.0/224, fine_imgs_256.shape[-2])
    render_feat = F.normalize(net_recog(render_face, trans_m), dim=-1, p=2)
    fine_feat = F.normalize(net_recog(fine_imgs_256, trans_m), dim=-1, p=2)
    cosine_d = torch.sum(render_feat * fine_feat, dim=-1)
    id_loss = torch.sum(1 - cosine_d) / cosine_d.shape[0]    
    
    mean_fake = torch.sum(fine_imgs * face_mask, dim=[1,2,3]) / torch.sum(face_mask, dim=[1,2,3])
    mean_render = torch.sum(render_imgs * face_mask, dim=[1,2,3]) / torch.sum(face_mask, dim=[1,2,3])
    skin_color_loss = nn.MSELoss()(mean_fake, mean_render)

    weight = np.ones([68])
    weight[17:27] = 200
    weight[36:37] = 50
    weight[-10:] = 200
    weight = np.expand_dims(weight, 0)
    weight = torch.tensor(weight).to(face_lm.device)
    recon_lm, recon_coeff = reconstruct_img(net_recon, face_model, fine_imgs)
    lm_loss = torch.sum((recon_lm - face_lm)**2, dim=-1) * weight
    lm_loss = torch.sum(lm_loss) / (face_lm.shape[0] * face_lm.shape[1] * 224.0)

    gamma_loss = torch.mean(torch.abs(recon_coeff['gamma'] - face_coeff['gamma']))

    return l2_loss, id_loss, skin_color_loss, lm_loss, gamma_loss

def compute_exp_warp_loss(vgg_model, face_model, renderer, gen_imgs, gen_labels, coeff_dicts, render_poses, device):
    pred_vertex, pred_face, face_mask, pred_lm = coeff2face(face_model, renderer, coeff_dicts, render_poses, device=device, return_ver=True)
    pred_face_diff = pred_vertex[0].unsqueeze(0) - pred_vertex[1].unsqueeze(0)
    pred_face_diff = torch.cat([-pred_face_diff[:,:,1].unsqueeze(-1), pred_face_diff[:,:,0].unsqueeze(-1), torch.zeros_like(pred_face_diff[:,:,0]).unsqueeze(-1)], dim=-1)
    _, _, flow_1to2 = renderer(pred_vertex[1].unsqueeze(0), face_model.face_buf, feat=pred_face_diff)
    flow_1to2 = flow_1to2[:,:2,:,:]
    fine_imgs = gen_imgs.clone()
    fine_imgs_256 = gen_imgs.clone()
    fine_imgs_256 = nn.functional.interpolate(fine_imgs_256, size=(256, 256), mode='bicubic')
    fine_imgs = nn.functional.interpolate(fine_imgs, size=(224, 224), mode='bicubic')
    fine_imgs = norm_img(fine_imgs, -1, 1) # map to (0, 1)
    fine_imgs_256 = norm_img(fine_imgs_256, -1, 1) # map to (0, 1)
    fake_1to2 = dense_image_warp(fine_imgs[0].unsqueeze(0), -flow_1to2 * 224 ).clamp_(0,1)

    loss_mask =  ((face_mask[0].unsqueeze(0) - face_mask[1].unsqueeze(0)) <= 0).float().to(device)
    region_loss = torch.sqrt(1e-8 + torch.sum((fine_imgs[1].unsqueeze(0) - fake_1to2) ** 2, dim=1, keepdims=True)) * loss_mask
    region_loss = torch.sum(region_loss) / torch.max(torch.sum(loss_mask), torch.tensor(1.0).to(device))

    test_imgs = gen_imgs.clone()
    test_imgs = norm_img(test_imgs, -1, 1).clamp_(0,1) # map to (0, 1)
    fine_labels = nn.functional.interpolate(gen_labels, size=(512, 512), mode='bicubic')
    face_mask = fine_labels.argmax(1)

    # mouth style transfer loss
    mouth_mask = ((face_mask==12)+(face_mask==13)).float()
    mouth_mask = mouth_mask.unsqueeze(1)
    style_features = vgg_model(test_imgs)
    style_loss = 0
    for j in range(1):
        style_feature = style_features[j]
        shape = style_feature.shape[-1]
        mouth_mask_ = nn.functional.interpolate(mouth_mask, size=(shape, shape), mode='nearest')
        style_mean = torch.mean(style_feature* mouth_mask_, dim=[1,2,3])
        style_std = torch.var(style_feature* mouth_mask_, dim=[1,2,3])
        style_loss += nn.MSELoss()(style_mean[0], style_mean[1]) + nn.MSELoss()(style_std[0], style_std[1])

    # hair loss
    hair_mask = ((face_mask==17)+(face_mask==16)).float()
    shair_region_mask = ((face_mask==17)+(face_mask==16)+(face_mask==15)+(face_mask==14)+(face_mask==0)).float()
    hair_mask = hair_mask.unsqueeze(1)
    shair_region_mask = shair_region_mask.unsqueeze(1)
    shair_region = test_imgs * shair_region_mask
    hair_loss = torch.mean((hair_mask[0].unsqueeze(0) - hair_mask[1].unsqueeze(0)) ** 2)
    shair_region_loss = torch.sqrt(1e-8 + torch.sum((shair_region[0].unsqueeze(0) - shair_region[1].unsqueeze(0)) ** 2, dim=1, keepdims=True))
    shair_region_loss = 2 * torch.sum(shair_region_loss) / torch.max(torch.sum(shair_region_mask), torch.tensor(1.0).to(device))

    loss = 100 * region_loss + 100 * shair_region_loss + 50 * hair_loss + 50 * style_loss  
    return loss

def compute_gamma_change_loss(net_recon, net_recog, face_model, renderer, gen_imgs, gen_labels, coeff_dicts, render_poses, device):
    fine_imgs = gen_imgs.clone()
    fine_imgs_256 = gen_imgs.clone()
    fine_imgs = nn.functional.interpolate(fine_imgs, size=(224, 224), mode='bicubic')
    fine_imgs_256 = nn.functional.interpolate(fine_imgs_256, size=(256, 256), mode='bicubic')
    fine_imgs = norm_img(fine_imgs, -1, 1).clamp_(0,1) # map to (0, 1)
    fine_imgs_256 = norm_img(fine_imgs_256, -1, 1).clamp_(0,1) # map to (0, 1)
    _, _, pred_lm = coeff2face(face_model, renderer, coeff_dicts, render_poses, device=device)

    net_recog.eval()
    trans_m = estimate_norm_torch(pred_lm * 256.0/224, fine_imgs_256.shape[-2])
    fine_imgs_256 = resize_n_crop(fine_imgs_256, trans_m, dsize=fine_imgs_256.shape[-2]).clamp_(0,1)
    fine_feat = F.normalize(net_recog(fine_imgs_256, trans_m), dim=-1, p=2)
    cosine_d = torch.sum(fine_feat[0].unsqueeze(0) * fine_feat[1].unsqueeze(0)).clamp_(0,1)
    id_loss = torch.sum(1.0 - cosine_d)

    net_recon.eval()
    weight = np.ones([68])
    weight[17:27] = 60
    weight[36:37] = 60
    weight[-8:] = 30
    weight = np.expand_dims(weight, 0)
    weight = torch.tensor(weight).to(device)
    recon_lm, _ = reconstruct_img(net_recon, face_model, fine_imgs)
    lm_loss = torch.sum((recon_lm[0].unsqueeze(0) - recon_lm[1].unsqueeze(0))**2, dim=-1) * weight
    lm_loss = torch.sum(lm_loss) / (68.0 * 224.0) + 1e-6

    test_imgs = gen_imgs.clone()
    test_imgs = norm_img(test_imgs, -1, 1).clamp_(0,1) # map to (0, 1)
    fine_labels = nn.functional.interpolate(gen_labels, size=(512, 512), mode='bicubic')
    face_mask = fine_labels.argmax(1)

    hair_mask = ((face_mask==17)+(face_mask==16)).float()
    shair_region_mask = ((face_mask==17)+(face_mask==16)+(face_mask==15)+(face_mask==14)+(face_mask==0)).float()
    hair_mask = hair_mask.unsqueeze(1)
    shair_region_mask = shair_region_mask.unsqueeze(1)
    shair_region = test_imgs * shair_region_mask
    hair_loss = torch.mean((hair_mask[0].unsqueeze(0) - hair_mask[1].unsqueeze(0)) ** 2)
    region_loss = torch.sqrt(1e-8 + torch.sum((shair_region[0].unsqueeze(0) - shair_region[1].unsqueeze(0)) ** 2, dim=1, keepdims=True))
    region_loss = 2 * torch.sum(region_loss) / torch.max(torch.sum(shair_region_mask), torch.tensor(1.0).to(device))

    
    loss = 100 * region_loss + 50 * hair_loss + 10 * id_loss + 1 * lm_loss
    
    return loss 

def compute_id_change_loss(gen_imgs, gen_labels, device):

    test_imgs = gen_imgs.clone()
    test_imgs = norm_img(test_imgs, -1, 1).clamp_(0,1) # map to (0, 1)
    test_imgs = gen_imgs.clone()
    test_imgs = norm_img(test_imgs, -1, 1).clamp_(0,1) # map to (0, 1)
    fine_labels = nn.functional.interpolate(gen_labels, size=(512, 512), mode='bicubic')
    face_mask = fine_labels.argmax(1).unsqueeze(1)

    bg_region_mask = ((face_mask==0)).float()
    bg_region = test_imgs * bg_region_mask
    region_loss = torch.sqrt(1e-8 + torch.sum((bg_region[0].unsqueeze(0) - bg_region[1].unsqueeze(0)) ** 2, dim=1, keepdims=True))
    region_loss = 2 * torch.sum(region_loss) / torch.max(torch.sum(bg_region_mask), torch.tensor(1.0).to(device))

    loss = 100 * region_loss
    return loss 

# def train_ddp(world_rank, rank, world_size, node, opt):
def train_ddp(rank, world_size, opt):
    world_rank = rank
    
    torch.manual_seed(world_rank)
    dist.init_process_group("nccl", rank=world_rank, world_size=world_size)
    device = torch.device(rank)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)
    # print(metadata)
    decoder = getattr(decoders, metadata['model'])

    z_dim = metadata['z_dim']
    id_dim = metadata['id_dim']
    exp_dim = metadata['exp_dim']
    gamma_dim = metadata['gamma_dim']

    # Setup Pretrained Models
    vae_dir = metadata['vae_path']
    bfm_dir = metadata['bfm_path']
    face_dir = metadata['face_path']
    parsing_dir = metadata['parsing_path']
    vae_id = torch.load(os.path.join(vae_dir, '14000_vae_id.pth'), map_location=device)
    vae_exp = torch.load(os.path.join(vae_dir, '20000_vae_exp.pth'), map_location=device)
    vae_gamma = torch.load(os.path.join(vae_dir, '17000_vae_gamma.pth'), map_location=device)
    vae_id.eval()
    vae_exp.eval()
    vae_gamma.eval()
    ps_model = BiSeNet(n_classes=19).cuda(rank)
    ps_model.load_state_dict(torch.load(os.path.join(parsing_dir, '79999_iter.pth'), map_location=device))
    net_recon, net_recog, face_model, renderer = setup_model(face_dir, bfm_dir, device)

    # Setup loss weight
    adv_lmd = metadata['lmd_adv']
    r1_lmd = metadata['lmd_r1']
    z_lmd = metadata['lmd_z']
    pose_lmd = metadata['lmd_pose']
    l2_lmd = metadata['lmd_l2']
    id_lmd = metadata['lmd_id']
    skin_lmd = metadata['lmd_skin']
    lm_lmd = metadata['lmd_lm']
    gamma_lmd = metadata['lmd_gamma']
    ce_lmd = metadata['lmd_ce']

    # Sample latent feature
    if not opt.second:
        fix_latent, _, fix_sample_id, fix_sample_exp, fix_sample_gamma = sample_latent_coeff(metadata['num_sample'], vae_id, vae_exp, vae_gamma, z_dim, id_dim, exp_dim, gamma_dim, device)
    else:
        fix_exp_pair, _, fix_exp_id_pair, fix_exp_exp_pair, fix_exp_gamma_pair = sample_latent_coeff(4, vae_id, vae_exp, vae_gamma, z_dim, id_dim, exp_dim, gamma_dim, device)
        fix_gamma_pair, _, fix_gamma_id_pair, fix_gamma_exp_pair, fix_gamma_gamma_pair = sample_latent_coeff(4, vae_id, vae_exp, vae_gamma, z_dim, id_dim, exp_dim, gamma_dim, device, pair='gamma')
        fix_latent = torch.cat([fix_exp_pair, fix_gamma_pair], dim=0)
        fix_sample_id = torch.cat([fix_exp_id_pair, fix_gamma_id_pair], dim=0)
        fix_sample_exp = torch.cat([fix_exp_exp_pair, fix_gamma_exp_pair], dim=0)
        fix_sample_gamma = torch.cat([fix_exp_gamma_pair, fix_gamma_gamma_pair], dim=0)

    latent_dim = fix_latent.shape[-1]
    generator = getattr(generators, metadata['generator'])(decoder, metadata['render_size'], metadata['img_size'], metadata['plane_reso'], latent_dim, metadata['w_dim'], metadata['c_dim'],device).cuda(rank)
    discriminator = getattr(discriminators, metadata['discriminator'])(metadata['img_size'], img_channels=6, latent_dim=latent_dim).cuda(rank)
    ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)

    if opt.load_dir != '':
        if opt.load_step != '':
            generator.load_state_dict(torch.load(os.path.join(opt.load_dir, opt.load_step + '_generator.pth'), map_location=device))
            discriminator.load_state_dict(torch.load(os.path.join(opt.load_dir, opt.load_step + '_discriminator.pth'), map_location=device))
            ema.load_state_dict(torch.load(os.path.join(opt.load_dir, opt.load_step + '_ema.pth'), map_location=device))
        else:
            generator.load_state_dict(torch.load(os.path.join(opt.load_dir, 'generator.pth'), map_location=device))
            discriminator.load_state_dict(torch.load(os.path.join(opt.load_dir, 'discriminator.pth'), map_location=device))
            ema.load_state_dict(torch.load(os.path.join(opt.load_dir, 'ema.pth'), map_location=device))
        requires_grad(generator, True)
        requires_grad(discriminator, True)
        
    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    discriminator_ddp = DDP(discriminator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    
    generator = generator_ddp.module
    discriminator = discriminator_ddp.module

    d_reg_ratio = metadata['d_reg_every'] / (metadata['d_reg_every'] + 1)

    optimizer_G = torch.optim.Adam(generator_ddp.parameters(), 
                    lr=metadata['gen_lr'], betas=(0, 0.99))

    optimizer_D = torch.optim.Adam(discriminator_ddp.parameters(), 
                    lr=metadata['disc_lr'] * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

    if opt.load_dir != '':
        if opt.load_step != '':
            optimizer_G.load_state_dict(torch.load(os.path.join(opt.load_dir, opt.load_step + '_optimizer_G.pth'), map_location=device))
            optimizer_D.load_state_dict(torch.load(os.path.join(opt.load_dir, opt.load_step + '_optimizer_D.pth'), map_location=device))
        else:
            optimizer_G.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_G.pth'), map_location=device))
            optimizer_D.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_D.pth'), map_location=device))


    generator_losses = []
    discriminator_losses = []

    if opt.set_step != None:
        generator.step = opt.set_step
        discriminator.step = opt.set_step
    
    generator.set_device(device)
    

    # ----------
    #  Training
    # ----------

    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(discriminator))
        f.write('\n\n')
        f.write(str(curriculum))

    torch.manual_seed(rank)
    dataloader = None
    total_progress_bar = tqdm(total = opt.n_epochs, desc = "Total progress", dynamic_ncols=True)
    total_progress_bar.update(discriminator.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)

    scale_factor = metadata['img_size']/float(metadata['render_size'])
    for _ in range (opt.n_epochs):
        total_progress_bar.update(1)

        metadata = curriculums.extract_metadata(curriculum, discriminator.step)

        if not dataloader or dataloader.batch_size != metadata['batch_size']:
            dataloader, CHANNELS = datasets.get_dataset_distributed(metadata['dataset'],
                                    world_size,
                                    rank,
                                    **metadata)

            step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator.step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator.step)

            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator.step - step_last_upsample))

        for i, (imgs, mat) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            mat = setup_dict(mat, device=device)

            if batch_size != metadata['batch_size']:
                rand_ind = np.random.randint(0, batch_size, size=metadata['batch_size'])
                imgs = imgs[rand_ind]
                mat = rand_sample(mat, rand_ind)
            
            real_imgs = imgs.to(device, non_blocking=True)
            pose = mat['angle']

            fixed_pose = pose[0].unsqueeze(0).repeat(8, 1)
            if discriminator.step % opt.model_save_interval == 0 and world_rank == 0 and discriminator.step != 0 and discriminator.step != 310000:
                torch.save(generator_ddp.module.state_dict(), os.path.join(opt.output_dir, str(discriminator.step) + '_generator.pth'))
                torch.save(discriminator_ddp.module.state_dict(), os.path.join(opt.output_dir, str(discriminator.step) + '_discriminator.pth'))
                torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, str(discriminator.step) + '_optimizer_G.pth'))
                torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, str(discriminator.step) + '_optimizer_D.pth'))
                torch.save(ema.state_dict(), os.path.join(opt.output_dir, str(discriminator.step) + '_ema.pth'))

            batch_size = metadata['batch_size']
            
            if dataloader.batch_size != metadata['batch_size']: break

            generator_ddp.train()
            discriminator_ddp.train()
    
            metadata['nerf_noise'] = max(0, 1. - discriminator.step/5000.)
            
            trans_matrx = torch.Tensor([(-1,0),(0,1)]).to(device)
            gt_pose = math.pi/2 + (pose[:,:2] @ trans_matrx).to(device)
            fixed_pose = math.pi/2 + (fixed_pose[:,:2] @ trans_matrx).to(device)
            
            ################################################################
            # TRAIN DISCRIMINATOR
            with torch.no_grad():
                latent, _, _, _, _ = sample_latent_coeff(real_imgs.shape[0], vae_id, vae_exp, vae_gamma, z_dim, id_dim, exp_dim, gamma_dim, device)
                raw_imgs, raw_labels, gen_imgs, _, _, _, gen_pose = generator_ddp(latent, gt_pose, **metadata)
            cat_gen_img = torch.cat([up(raw_imgs, scale_factor), gen_imgs], dim=1)
            cat_real_img = torch.cat([down_up(real_imgs, scale_factor), real_imgs], dim=1)

            # GAN Loss
            cat_real_img.requires_grad = True
            r_preds, r_pred_pose, r_pred_z = discriminator_ddp(cat_real_img, gt_pose)
            g_preds, _, _ = discriminator_ddp(cat_gen_img.detach(), gen_pose)
            d_loss_pose = torch.nn.MSELoss()(r_pred_pose, gt_pose)
            d_loss_z = torch.nn.MSELoss()(r_pred_z, latent)
            d_loss = torch.nn.functional.softplus(g_preds).mean() + torch.nn.functional.softplus(-r_preds).mean() + d_loss_pose * pose_lmd + d_loss_z * z_lmd
            
            optimizer_D.zero_grad()
            d_loss.backward()
            for param in discriminator_ddp.parameters():
                if param.grad is not None:
                    nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            torch.nn.utils.clip_grad_norm_(discriminator_ddp.parameters(), metadata.get('grad_clip', 0.3))
            optimizer_D.step()
            
            # R1 Regularizor
            d_regularize = i % metadata['d_reg_every'] == 0
            if d_regularize:
                cat_real_img.requires_grad = True
                r_preds, _, _ = discriminator_ddp(cat_real_img, gt_pose)
                r1_loss = d_r1_loss(r_preds, cat_real_img)
                
                optimizer_D.zero_grad()
                (r1_lmd / 2 * r1_loss * metadata['d_reg_every'] + 0 * r_preds[0]).backward()
                for param in discriminator_ddp.parameters():
                    if param.grad is not None:
                        nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                torch.nn.utils.clip_grad_norm_(discriminator_ddp.parameters(), metadata.get('grad_clip', 0.3))
                optimizer_D.step()

            ################################################################
            # TRAIN GENERATOR
            latent, latent_z, sample_id, sample_exp, sample_gamma = sample_latent_coeff(real_imgs.shape[0], vae_id, vae_exp, vae_gamma, z_dim, id_dim, exp_dim, gamma_dim, device)

            raw_imgs, raw_labels, gen_imgs, sigma, sigma_fine, sigma_coase, gen_pose = generator_ddp(latent, gt_pose, **metadata)
            cat_gen_img = torch.cat([up(raw_imgs, scale_factor), gen_imgs], dim=1)
            g_preds, g_pred_pose, g_pred_z = discriminator_ddp(cat_gen_img, gen_pose)

            # render 3DMM face
            face_pose_pitch = (math.pi/2 - gen_pose[:, 0]).unsqueeze(1)
            face_pose_yaw = (gen_pose[:, 1] - math.pi/2).unsqueeze(1)
            face_pose_roll = torch.zeros((batch_size, 1)).to(gen_pose.device)
            face_pose = torch.cat([face_pose_pitch, face_pose_yaw, face_pose_roll], dim=1)

            # render 3DMM
            coeff_dict = sample_dict(sample_id, sample_exp, sample_gamma, face_pose, device=device)
            pred_face, face_mask, face_lm = coeff2face(face_model, renderer, coeff_dict, face_pose, device=device)
            
            # pose loss * 10
            g_loss_pose = torch.nn.MSELoss()(g_pred_pose, gen_pose)
            # latent loss * 1
            g_loss_z = torch.nn.MSELoss()(g_pred_z, latent)

            # adversary loss
            g_loss_adv = torch.nn.functional.softplus(-g_preds).mean()
            
            # smooth loss
            g_loss_smooth = torch.nn.functional.l1_loss(sigma_fine, sigma_coase)
            
            if discriminator.step < opt.warmup1:
                g_loss = g_loss_adv * adv_lmd + g_loss_z * z_lmd + g_loss_pose * pose_lmd
                if world_rank == 0:
                    interior_step_bar.update(1)
                    if i%10 == 0:
                        tqdm.write(f"[Experiment: {opt.output_dir}] [Epoch: {discriminator.epoch}/{opt.n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Adv loss: {g_loss_adv.item()}] [Step: {discriminator.step}] [Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}]")

            elif (discriminator.step >= opt.warmup1) and (discriminator.step < opt.warmup2):
                l2_loss, id_loss, skin_color_loss, lm_loss, gamma_loss = compute_imitative(
                                net_recon, net_recog, face_model, gen_imgs, pred_face, face_mask, coeff_dict, face_lm)
                g_loss_imit = l2_loss * l2_lmd + id_loss * id_lmd + skin_color_loss * skin_lmd + lm_loss * lm_lmd + gamma_loss * gamma_lmd
                g_loss = g_loss_adv * adv_lmd + g_loss_z * z_lmd + g_loss_pose * pose_lmd + g_loss_imit + g_loss_smooth * 0.1
                if world_rank == 0:
                    interior_step_bar.update(1)
                    if i%10 == 0:
                        tqdm.write(f"[Experiment: {opt.output_dir}] [Epoch: {discriminator.epoch}/{opt.n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Adv loss: {g_loss_adv.item()}] [Imi loss: {g_loss_imit.item()}] [Sm loss: {g_loss_smooth.item()}] [Step: {discriminator.step}] [Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}]")
                    
            
            else:
                l2_loss, id_loss, skin_color_loss, lm_loss, gamma_loss = compute_imitative(
                                net_recon, net_recog, face_model, gen_imgs, pred_face, face_mask, coeff_dict, face_lm)
                g_loss_imit = l2_loss * l2_lmd + id_loss * id_lmd + skin_color_loss * skin_lmd + lm_loss * lm_lmd + gamma_loss * gamma_lmd
                gt_labels = face_parsing(ps_model, gen_imgs)
                g_loss_ce = torch.nn.CrossEntropyLoss()(raw_labels, gt_labels.squeeze(1))
                g_loss = g_loss_adv * adv_lmd + g_loss_z * z_lmd + g_loss_pose * pose_lmd + g_loss_imit + g_loss_ce * ce_lmd + g_loss_smooth * 0.5
                if world_rank == 0:
                    interior_step_bar.update(1)
                    if i%10 == 0:
                        tqdm.write(f"[Experiment: {opt.output_dir}] [Epoch: {discriminator.epoch}/{opt.n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Adv loss: {g_loss_adv.item()}] [Imi loss: {g_loss_imit.item()}] [CE loss: {g_loss_ce.item()}] [Sm loss: {g_loss_smooth.item()}] [Step: {discriminator.step}] [Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}]")
                
                
            # disentanglement learning
            if opt.second:
                render_pose = gen_pose[0].repeat(4,1)
                face_pose_pitch = (math.pi/2 - render_pose[:, 0]).unsqueeze(1).to(device)
                face_pose_yaw = (render_pose[:, 1] - math.pi/2).unsqueeze(1).to(device)
                face_pose_roll = torch.zeros((2, 1)).to(device)
                face_pose = torch.cat([face_pose_pitch, face_pose_yaw, face_pose_roll], dim=1)
                
                # generate contrast pair
                latent_id_2 = torch.cat([latent_z[0].unsqueeze(0), sample_id[1].unsqueeze(0), sample_exp[0].unsqueeze(0), sample_gamma[0].unsqueeze(0)], dim=1)
                latent_exp_2 = torch.cat([latent_z[0].unsqueeze(0), sample_id[0].unsqueeze(0), sample_exp[1].unsqueeze(0), sample_gamma[0].unsqueeze(0)], dim=1)
                latent_gamma_2 = torch.cat([latent_z[0].unsqueeze(0), sample_id[0].unsqueeze(0), sample_exp[0].unsqueeze(0), sample_gamma[1].unsqueeze(0)], dim=1)

                latent_contrast_pair = torch.cat([latent_id_2, latent_exp_2, latent_gamma_2], dim=0)
                gen_imgs_contrast, _ = generator_ddp.module.staged_forward(latent_contrast_pair, render_pose, **metadata)

                gen_imgs_id_pair = torch.cat([gen_imgs[0].unsqueeze(0), gen_imgs_contrast[0].unsqueeze(0)], dim=0)
                
                gen_imgs_exp_pair = torch.cat([gen_imgs[0].unsqueeze(0), gen_imgs_contrast[1].unsqueeze(0)], dim=0)
                coeff_dict_exp_pair = sample_dict(sample_id[0].unsqueeze(0).repeat(2,1), sample_exp[0:2], sample_gamma[0].unsqueeze(0).repeat(2,1), face_pose[0].unsqueeze(0).repeat(2,1), device=device)
                
                gen_imgs_gamma_pair = torch.cat([gen_imgs[0].unsqueeze(0), gen_imgs_contrast[2].unsqueeze(0)], dim=0)
                coeff_dict_gamma_pair = sample_dict(sample_id[0].unsqueeze(0).repeat(2,1), sample_exp[0].unsqueeze(0).repeat(2,1), sample_gamma[0:2], face_pose[0].unsqueeze(0).repeat(2,1), device=device)

                id_change_loss = compute_id_change_loss(gen_imgs_id_pair, device)
                exp_change_loss = compute_exp_warp_loss(face_model, renderer, gen_imgs_exp_pair, coeff_dict_exp_pair, face_pose[0].unsqueeze(0).repeat(2,1), device)
                gamma_change_loss = compute_gamma_change_loss(net_recon, net_recog, face_model, renderer, gen_imgs_gamma_pair, coeff_dict_gamma_pair, face_pose[0].unsqueeze(0).repeat(2,1), device)
                
                g_contrast_loss = exp_change_loss + id_change_loss + gamma_change_loss

                g_loss = g_loss * 0.7 + g_contrast_loss

            # overall loss
            g_loss.backward()
            for param in generator_ddp.parameters():
                if param.grad is not None:
                    nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get('grad_clip', 0.3))
            optimizer_G.step()
            optimizer_G.zero_grad()
            ema.update()

            if world_rank == 0:
                if discriminator.step % opt.sample_interval == 0 and discriminator.step !=0 :
                    generator_ddp.eval()
                    ema.store(generator_ddp.parameters())
                    ema.copy_to(generator_ddp.parameters())
                    with torch.no_grad():
                        raw_imgs, raw_rgb_labels, gen_imgs, gen_poses = generator_ddp.module.shape_forward(fix_latent.cuda(rank), fixed_pose, noise_mode='const', truncation_psi=1, **metadata)

                    face_pose_pitch = (math.pi/2 - gen_poses[:, 0]).unsqueeze(1)
                    face_pose_yaw = (gen_poses[:, 1] - math.pi/2).unsqueeze(1)
                    face_pose_roll = torch.zeros((8, 1)).to(gen_poses.device)
                    face_pose = torch.cat([face_pose_pitch, face_pose_yaw, face_pose_roll], dim=1)
                    fix_coeff_dict = sample_dict(fix_sample_id, fix_sample_exp, fix_sample_gamma, face_pose, device=device)
                    face, face_mask, face_lm = coeff2face(face_model, renderer, fix_coeff_dict, face_pose, device=device)
                    face3dmm_vis = face * face_mask

                    save_image(raw_imgs[:], os.path.join(opt.output_dir, f"{discriminator.step}_raw.png"), 
                                nrow=4, normalize=True, range=(-1, 1))
                    save_image(raw_rgb_labels[:], os.path.join(opt.output_dir, f"{discriminator.step}_raw_label.png"), 
                                nrow=4, normalize=True, range=(-1, 1))
                    save_image(gen_imgs[:], os.path.join(opt.output_dir, f"{discriminator.step}_fixed.png"), 
                                nrow=4, normalize=True, range=(-1, 1))
                    save_image(face3dmm_vis[:], os.path.join(opt.output_dir, f"{discriminator.step}_3dmm.png"), 
                                nrow=4, normalize=True)

                    ema.restore(generator_ddp.parameters())
                    torch.cuda.empty_cache()

                if discriminator.step % opt.sample_interval == 0 :
                    torch.save(generator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'generator.pth'))
                    torch.save(discriminator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'discriminator.pth'))
                    torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'optimizer_G.pth'))
                    torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, 'optimizer_D.pth'))
                    torch.save(ema.state_dict(), os.path.join(opt.output_dir, 'ema.pth'))
                    torch.save(generator_losses, os.path.join(opt.output_dir, 'generator.losses'))
                    torch.save(discriminator_losses, os.path.join(opt.output_dir, 'discriminator.losses'))
                    torch.cuda.empty_cache()
                    
            if opt.eval_freq > 0 and (discriminator.step % opt.eval_freq) == 0:    
                generated_dir = os.path.join(opt.output_dir, 'evaluation/generated')
                if rank == 0:
                    fid_evaluation.setup_evaluation(metadata['dataset'], generated_dir, metadata, num_imgs=1000)
                generator_ddp.eval()
                ema.store(generator_ddp.parameters())
                ema.copy_to(generator_ddp.parameters())
                if rank == 0:
                    fid_evaluation.output_images(generator_ddp, metadata, vae_id, vae_exp, vae_gamma, generated_dir, num_imgs=1000)
                ema.restore(generator_ddp.parameters())
                dist.barrier()
                if rank == 0:    
                    fid = fid_evaluation.calculate_fid(metadata['dataset'], generated_dir, metadata['img_size'])
                    with open(os.path.join(opt.output_dir, f'fid.txt'), 'a') as f:
                        f.write(f'\n{discriminator.step}:{fid}')
                    torch.cuda.empty_cache()

            discriminator.step += 1
            generator.step += 1
        discriminator.epoch += 1
        generator.epoch += 1

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--load_step', type=str, default='')
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, default=1000)
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=1000)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--second', action='store_true')
    parser.add_argument('--warmup1', type=int, default=5000)
    parser.add_argument('--warmup2', type=int, default=20000)
    parser.add_argument('--port', type=str, default='12345')

    opt = parser.parse_args()
    print(opt)
    
    os.makedirs(opt.output_dir, exist_ok=True)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = opt.port
    mp.spawn(train_ddp, args=(num_gpus, opt), nprocs=num_gpus, join=True)
