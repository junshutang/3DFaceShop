"""Train VAE-GAN. Supports distributed training."""

import argparse
import os
import numpy as np
import copy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from vaes import vaes
from discriminators import discriminators
import datasets
import curriculums
from tqdm import tqdm
from facenets.bfm import ParametricFaceModel
from facenets.render import MeshRenderer
from torchvision.utils import save_image

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    dist.destroy_process_group()

def process_data(factor, coeff, norm):
    if factor == 'id':
        input_coeff = torch.cat([coeff['id']/norm[1], coeff['tex']/norm[0]], dim=-1)
    elif factor == 'exp':
        input_coeff = coeff['exp']/norm
    elif factor == 'rot':
        input_coeff = coeff['angle']/norm
    elif factor == 'gamma':
        input_coeff = coeff['gamma']/norm
    elif factor == 'trans':
        input_coeff = coeff['trans']/norm
    else:
        raise Exception('invalid factor')
    
    return input_coeff.squeeze(1)

def sample_data(co_dict, device, number):
    id_coeffs = co_dict["id"][:number].squeeze(1).to(device)
    exp_coeffs = co_dict["exp"][:number].squeeze(1).to(device)
    tex_coeffs = co_dict["tex"][:number].squeeze(1).to(device)
    angles = co_dict["angle"][:number].squeeze(1).to(device)
    gammas = co_dict["gamma"][:number].squeeze(1).to(device)
    translations = co_dict["trans"][:number].squeeze(1).to(device)
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }

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

def setup(factor):
    if factor == 'id':
        coeff_dim = 160
        latent_dim = 128
        ch_dim = 512
        encoder_depth = 3
        decoder_depth = 3
        norm = [9.0, 4.2] # tex, id

    elif factor == 'exp':
        coeff_dim = 64
        latent_dim = 32
        ch_dim = 256
        encoder_depth = 3
        decoder_depth = 3
        norm = 3.0
    elif factor == 'gamma':
        coeff_dim = 27
        latent_dim = 16
        ch_dim = 128
        encoder_depth = 3
        decoder_depth = 3
        norm = 0.85
    elif factor == 'trans':
        coeff_dim = 3
        latent_dim = 3
        ch_dim = 32
        encoder_depth = 3
        decoder_depth = 3
        norm = 1

    return coeff_dim, latent_dim, ch_dim, encoder_depth, decoder_depth, norm

def setup_face_model(bfm_dir):
    camera_d = 10.0
    focal = 1015.0  # 1015.0
    center = 112.0 # 112.0
    z_near = 5.0
    z_far = 15.0

    face_model = ParametricFaceModel(
            bfm_folder=bfm_dir, camera_distance=camera_d, focal=focal, center=center,
            is_train=False, default_name='BFM_model_front.mat'
        )
        
    fov = 2 * np.arctan(center / focal) * 180 / np.pi
    renderer = MeshRenderer(
            rasterize_fov=fov, znear=z_near, zfar=z_far, rasterize_size=int(2 * center)
        )
    
    return face_model, renderer

def coeff2face(face_model, renderer, coef_dict, device):
    
    face_model.to(device)
    pred_vertex, _, pred_color, _ = face_model.compute_for_render(coef_dict)
    pred_mask, _, pred_face = renderer(pred_vertex, face_model.face_buf, feat=pred_color)
    
    output_vis = pred_face * pred_mask
    return output_vis
    
def train_ddp(rank, world_size, opt):
    torch.manual_seed(0)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(rank)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)
    factor = opt.factor 
    coeff_dim, latent_dim, ch_dim, encoder_depth, decoder_depth, norm = setup(factor)

    if opt.load_dir != '':
        vae = torch.load(os.path.join(opt.load_dir, 'vae_'+factor+'.pth'), map_location=device)
    else:
        vae = getattr(vaes, metadata['vae'])(coeff_dim, latent_dim, ch_dim, encoder_depth, decoder_depth).cuda(rank)
        discriminator = getattr(discriminators, metadata['discriminator'])(coeff_dim, latent_dim, ch_dim, encoder_depth).cuda(rank)
        
    # wandb.init(project='eg3d-vae')
    vae_ddp = DDP(vae, device_ids=[rank], find_unused_parameters=True)
    discriminator_ddp = DDP(discriminator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    vae = vae_ddp.module
    discriminator = discriminator_ddp.module
    optimizer_vae = torch.optim.Adam(vae_ddp.parameters(), 
                    lr=metadata['vae_lr'])
    optimizer_d = torch.optim.Adam(discriminator_ddp.parameters(), 
                    lr=metadata['d_lr'])
    scheduler_vae = optim.lr_scheduler.ExponentialLR(optimizer_vae, gamma=0.95)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.95)
    
    if opt.load_dir != '':
        optimizer_vae.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_vae_'+factor+'.pth')))
        optimizer_d.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_d_'+factor+'.pth')))
        scheduler_vae.load_state_dict(torch.load(os.path.join(opt.load_dir, 'scheculer_vae_'+factor+'.pth')))
        scheduler_d.load_state_dict(torch.load(os.path.join(opt.load_dir, 'scheculer_d_'+factor+'.pth')))
    
    # ----------
    #  Training
    # ----------
    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(vae))

    torch.manual_seed(rank)
    dataloader = None
    total_progress_bar = tqdm(total = opt.n_epochs, desc = "Total progress", dynamic_ncols=True)
    total_progress_bar.update(vae.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)
    face_model, renderer = setup_face_model(metadata['bfm_path'])

    for epoch in range (opt.n_epochs):
        total_progress_bar.update(1)

        metadata = curriculums.extract_metadata(curriculum, vae.step)
        
        if not dataloader or dataloader.batch_size != metadata['batch_size']:

            dataloader, _ = datasets.get_dataset(metadata['dataset'], **metadata)

            step_next_upsample = curriculums.next_upsample_step(curriculum, vae.step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, vae.step)

            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((vae.step - step_last_upsample))

        for i, (mat) in enumerate(dataloader):
            weight = opt.weight

            if vae.step % opt.model_save_interval == 0 and rank == 0 and vae.step !=0:
                torch.save(vae_ddp.module, os.path.join(opt.output_dir, str(vae.step) + '_vae_'+factor+'.pth'))
                torch.save(discriminator_ddp.module, os.path.join(opt.output_dir, str(vae.step) + '_d_'+factor+'.pth'))
                torch.save(optimizer_vae.state_dict(), os.path.join(opt.output_dir, str(vae.step) + '_optimizer_vae_'+factor+'.pth'))
                torch.save(scheduler_vae.state_dict(), os.path.join(opt.output_dir, str(vae.step) + '_scheduler_vae_'+factor+'.pth'))
                torch.save(optimizer_d.state_dict(), os.path.join(opt.output_dir, str(vae.step) + '_optimizer_d_'+factor+'.pth'))
                torch.save(scheduler_d.state_dict(), os.path.join(opt.output_dir, str(vae.step) + '_scheduler_d_'+factor+'.pth'))
            
            metadata = curriculums.extract_metadata(curriculum, vae.step)
            
            mat = setup_dict(mat, device=device)

            if dataloader.batch_size != metadata['batch_size']: break

            vae_ddp.train()
            discriminator_ddp.train()
            input_coeff = process_data(opt.factor, mat, norm).to(device, non_blocking=True)
            
            # train discriminator
            with torch.no_grad():
                preds, _, _, _ = vae_ddp(input_coeff)
            
            input_coeff.requires_grad = True
            r_preds = discriminator_ddp(input_coeff)
            g_preds = discriminator_ddp(preds.detach())

            d_loss = torch.nn.functional.softplus(g_preds).mean() + torch.nn.functional.softplus(-r_preds).mean()
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # train vae
            results = vae_ddp(input_coeff)
            g_preds = discriminator_ddp(results[0])
            train_loss = vae_ddp.module.loss_function(weight, *results)
            recon_data = train_loss['recon']
            kl_loss = train_loss['kld']
            rec_loss = train_loss['rec_loss']
            adv_loss = torch.nn.functional.softplus(-g_preds).mean()
            g_loss = train_loss['loss'] + adv_loss * 5

            optimizer_vae.zero_grad()
            g_loss.backward()
            optimizer_vae.step()
            
            if rank == 0:
                interior_step_bar.update(1)
                if i%10 == 0:
                    tqdm.write(f"[Experiment: {opt.output_dir}] [GPU: {os.environ['CUDA_VISIBLE_DEVICES']}] [Epoch: {vae.epoch}/{opt.n_epochs}] [D loss: {d_loss.item()}] [VAE loss: {g_loss.item()}] [KL loss: {kl_loss.item()}] [Rec loss: {rec_loss.item()}] [Step: {vae.step}] [Batch Size: {metadata['batch_size']}]")
                    
                if i%100 == 0:
                    fixed_coeff = copy.deepcopy(mat)
                    fixed_coeff = sample_data(fixed_coeff, device=device, number=8)
                    ori_vis = coeff2face(face_model, renderer, fixed_coeff, device=device)
                    if factor == 'id':
                        fixed_coeff['id'] = recon_data[:8,:80] * norm[1]
                        fixed_coeff['tex'] = recon_data[:8,80:] * norm[0]
                    else:
                        fixed_coeff[factor] = recon_data[:8] * norm
                    recon_vis = coeff2face(face_model, renderer, fixed_coeff, device=device)
                    com_vis = torch.cat([ori_vis, recon_vis], dim=0)
                    save_image(com_vis, os.path.join(opt.render_dir, f"{epoch}_{i}_recon.png"), nrow=8, normalize=True)

                if vae.step % opt.sample_interval == 0 and vae.step !=0 :
                    torch.save(vae_ddp.module, os.path.join(opt.output_dir, 'vae_'+factor+'.pth'))
                    torch.save(optimizer_vae.state_dict(), os.path.join(opt.output_dir, 'optimizer_vae_'+factor+'.pth'))
                    torch.save(scheduler_vae.state_dict(), os.path.join(opt.output_dir, 'scheduler_vae_'+factor+'.pth'))
            vae.step += 1
        vae.epoch += 1

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='vae')
    parser.add_argument('--render_dir', type=str, default='render')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=2000)
    parser.add_argument('--factor', type=str, default='id')
    parser.add_argument('--weight', type=float, default=0.0025)

    opt = parser.parse_args()
    print(opt)
    
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.render_dir, exist_ok=True)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = opt.port
    mp.spawn(train_ddp, args=(num_gpus, opt), nprocs=num_gpus, join=True)
