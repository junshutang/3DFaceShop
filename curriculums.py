"""
To easily reproduce experiments, and avoid passing several command line arguments, we implemented
a curriculum utility. Parameters can be set in a curriculum dictionary.

Curriculum Schema:

    Numerical keys in the curriculum specify an upsample step. When the current step matches the upsample step,
    the values in the corresponding dict be updated in the curriculum. Common curriculum values specified at upsamples:
        batch_size: Batch Size.
        num_steps: Number of samples along ray.
        img_size: Generated image resolution.
        batch_split: Integer number over which to divide batches and aggregate sequentially. (Used due to memory constraints)
        gen_lr: Generator learnig rate.
        disc_lr: Discriminator learning rate.

    fov: Camera field of view
    ray_start: Near clipping for camera rays.
    ray_end: Far clipping for camera rays.
    fade_steps: Number of steps to fade in new layer on discriminator after upsample.
    h_stddev: Stddev of camera yaw in radians.
    v_stddev: Stddev of camera pitch in radians.
    h_mean:  Mean of camera yaw in radians.
    v_mean: Mean of camera yaw in radians.
    sample_dist: Type of camera pose distribution. (gaussian | spherical_uniform | uniform)
    topk_interval: Interval over which to fade the top k ratio.
    topk_v: Minimum fraction of a batch to keep during top k training.
    betas: Beta parameters for Adam.
    unique_lr: Whether to use reduced LRs for mapping network.
    weight_decay: Weight decay parameter.
    r1_lambda: R1 regularization parameter.
    latent_dim: Latent dim for Siren network  in generator.
    grad_clip: Grad clipping parameter.
    model: Siren architecture used in generator. (SPATIALSIRENBASELINE | TALLSIREN)
    generator: Generator class. (ImplicitGenerator3d)
    discriminator: Discriminator class. (ProgressiveEncoderDiscriminator | ProgressiveDiscriminator)
    dataset: Training dataset. (CelebA | Carla | Cats)
    clamp_mode: Clamping function for Siren density output. (relu | softplus)
    z_dist: Latent vector distributiion. (gaussian | uniform)
    hierarchical_sample: Flag to enable hierarchical_sampling from NeRF algorithm. (Doubles the number of sampled points)
    z_labmda: Weight for experimental latent code positional consistency loss.
    pos_lambda: Weight parameter for experimental positional consistency loss.
    last_back: Flag to fill in background color with last sampled color on ray.
"""


def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step > current_step:
            return curriculum_step
    return float('Inf')

def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage, i.e. the epoch
    # it last upsampled
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step <= current_step:
            return curriculum_step
    return 0

def get_current_step(curriculum, epoch):
    step = 0
    for update_epoch in curriculum['update_epochs']:
        if epoch >= update_epoch:
            step += 1
    return step

def extract_metadata(curriculum, current_step):
    return_dict = {}
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int], reverse=True):
        if curriculum_step <= current_step:
            for key, value in curriculum[curriculum_step].items():
                return_dict[key] = value
            break
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict

FFHQ_512 = {
    0: {'batch_size': 4, 'num_sample': 8, 'num_steps': 48, 'render_size': 128, 'img_size': 512, 'gen_lr': 0.002, 'disc_lr': 0.0025},
    int(500e3): {},
    'dataset_path': 'data/FFHQ/aligned_img/*.png',
    'posepath': 'data/FFHQ/mat/',
    'vae_path': 'checkpoints/vae_ckp/',
    'bfm_path': 'checkpoints/face_ckpt/BFM',
    'face_path': 'checkpoints/face_ckpt/face_ckp/',
    'parsing_path': 'checkpoints/face_ckpt/',
    'fov': 12.6,
    'ray_start': 0.88,
    'ray_end': 1.12,
    
    'z_dim' : 128,
    'id_dim' : 128,
    'exp_dim' : 32,
    'gamma_dim': 16,
    
    'lmd_adv': 1,
    'lmd_r1': 10,
    'lmd_pose': 10,
    'lmd_z': 1,
    'lmd_sm': 1,
    'lmd_l2': 10,
    'lmd_id': 10,
    'lmd_skin': 1,
    'lmd_lm': 10,
    'lmd_gamma': 1000,
    'lmd_ce': 10,
    
    'w_dim': 512, # stylegan
    'c_dim': 2,
    'hidden_dim': 64,
    'plane_reso': 256,
    'plane_channel': 96,
    'grad_clip': 10,
    'd_reg_every': 16,
    'model': 'STYLEMLP_mask',
    'vae': 'VAE',
    'generator': 'TriplaneImplicitGenerator3d',
    'discriminator': 'SDiscriminator',
    'dataset': 'FFHQ_Mat',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'last_back': False,
}


VAE_ALL = {
    0: {'batch_size': 128, 'vae_lr': 0.005, 'd_lr': 0.0025},
    int(20000): {},
    'bfm_path': 'checkpoints/face_ckpt/BFM',
    'ryspath': 'data/RAVDESS/video_01_01_mat/*.mat',
    'ffhqpath': 'data/FFHQ/mat/*.mat',
    'vae': 'VAE',
    'dataset': 'VAE_Mat',
    'discriminator': "VAEDiscriminator",
}
