import argparse
import os
import cv2
import torch
from torch.nn import functional as F
from PIL import Image
import curriculums
import facenets.networks as networks
from facenets.bfm import ParametricFaceModel
import numpy as np
import dlib
from facenets.render import MeshRenderer
from scipy.io import loadmat, savemat
import os.path as osp
from skimage import transform as trans
from kornia.geometry import warp_affine
import warnings
import glob
warnings.filterwarnings("ignore")

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def POS(xp, x):
    npts = xp.shape[1]
    A = np.zeros([2*npts, 8])
    A[0:2*npts-1:2, 0:3] = x.transpose()
    A[0:2*npts-1:2, 3] = 1
    A[1:2*npts:2, 4:7] = x.transpose()
    A[1:2*npts:2, 7] = 1
    b = np.reshape(xp.transpose(), [2*npts, 1])
    k, _, _, _ = np.linalg.lstsq(A, b)
    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx, sTy], axis=0)
    return t, s

def resize_n_crop_img(img, lm, t, s, target_size=224., mask=None):
    w0, h0 = img.size
    scale = s/target_size*224

    w = (w0/scale*95).astype(np.int32)
    h = (h0/scale*95).astype(np.int32)

    left = (w/2 - target_size/2 + float((t[0] - w0/2)*95/scale)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*95/scale)).astype(np.int32)
    below = up + target_size
    img_old = img.resize((target_size, target_size), resample=Image.LANCZOS)
    img = img.resize((w, h), resample=Image.LANCZOS)
    padding_len = max([abs(min(0,left)),abs(min(0,up)),max(right-w,0),max(below-h,0)])
    if padding_len > 0:
        img = np.array(img)
        img = np.pad(img,pad_width=((padding_len,padding_len),(padding_len,padding_len),(0,0)),mode='reflect')
        img = Image.fromarray(img)
    img = img.crop((left+padding_len,up+padding_len,right+padding_len,below+padding_len))

    if mask is not None:
        mask = mask.resize((w, h), resample=Image.BICUBIC)
        mask = mask.crop((left, up, right, below))
    lm_old = lm
    lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                  t[1] + h0/2], axis=1)*95/scale
    lm = lm - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])

    return img, img_old, lm, lm_old,  mask

def nist_prec(x):
    x = (x.clone() - 0.5) * 2 # -1 ~ 1
    x = x[:, :, 12:243, 12:243] # center crop
    x = torch.flip(x,[1]) # RGB -> BGR
    return x

def reconstruct_image(recon_model, face_model, renderer, image):
    input_img = image.clone()
    input_img = F.interpolate(input_img, size=224, mode='bilinear', align_corners=True)
    output_coeff = recon_model(input_img)
    pred_coeffs_dict = face_model.split_coeff(output_coeff)
    pred_vertex, pred_tex, pred_color, pred_lm = face_model.compute_for_render(pred_coeffs_dict)
    pred_mask, _, pred_face = renderer(pred_vertex, face_model.face_buf, feat=pred_color)
    
    return pred_coeffs_dict, pred_face

def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

def align_img(img, lm, lm3D, mask=None, target_size=224., rescale_factor=204.):
    w0, h0 = img.size
    lm5p = lm
    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    # s = rescale_factor/s
    # processing the image
    img_new, img_old, lm_new, lm_old, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
    trans_params = np.array([w0, h0, s, t[0], t[1]])
    return trans_params, img_new, img_old, lm_new, lm_old, mask_new

def read_data(img, lm, lm3d_std, device, to_tensor=True):
    # to RGB 
    im = img
    W,H = im.size
    lm = lm.astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, im_old, lm, lm_old, _ = align_img(im, lm, lm3d_std, target_size=W)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        im_old = torch.tensor(np.array(im_old)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        lm = torch.tensor(lm).unsqueeze(0).to(device)
        lm_old = torch.tensor(lm_old).unsqueeze(0).to(device)
    return im, lm, im_old, lm_old

def detect_image(input_image, detector, predictor, savepath=""):
    image = input_image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        eyel = np.round(np.mean(shape[36:42,:],axis=0)).astype("int")
        eyer = np.round(np.mean(shape[42:48,:],axis=0)).astype("int")
        nose = shape[33]
        mouthl = shape[48]
        mouthr = shape[54]
        results = np.array(((eyel[0],eyel[1]), (eyer[0],eyer[1]), (nose[0],nose[1]), (mouthl[0],mouthl[1]), (mouthr[0],mouthr[1])))
        message = '%d %d\n%d %d\n%d %d\n%d %d\n%d %d\n' % (eyel[0],eyel[1],
            eyer[0],eyer[1],nose[0],nose[1],
            mouthl[0],mouthl[1],mouthr[0],mouthr[1])
        if savepath is not "":
            with open(savepath, 'w') as s_file:
                s_file.write(message)
        return results

def load_lm3d(bfm_folder):
    
    Lm3D = loadmat(osp.join(bfm_folder, 'similarity_Lm3D_all.mat'))
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D

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
    face_model = ParametricFaceModel(
            bfm_folder=bfm_dir, camera_distance=camera_d, focal=focal, center=center,
            is_train=False, default_name='BFM_model_front.mat'
        )
    face_model.to(device)
    fov = 2 * np.arctan(center / focal) * 180 / np.pi
    renderer = MeshRenderer(
            rasterize_fov=fov, znear=z_near, zfar=z_far, rasterize_size=int(2 * center)
        )
    return net_recon, face_model, renderer

def pre_img(facemodel, img_ori, lm_ori, coeff, device):
    input_size = img_ori.shape[-1]
    fixed_coeff = coeff
    fixed_coeff['trans'] = torch.tensor([0, -0.1, 0]).unsqueeze(0).to(device) # fixed translation
    fix_vertex, fix_tex, fix_color, fix_lm = facemodel.compute_for_render(fixed_coeff)
    fix_mask, _, fix_face = renderer(fix_vertex, facemodel.face_buf, feat=fix_color)
    align_m, align_5p = estimate_align_5p_torch(lm_ori, fix_lm * input_size/224.0, H=input_size)
    align_img = warp_affine(img_ori, align_m, dsize=(input_size, input_size), padding_mode='reflection')
    
    return align_img

def estimate_align_5p_torch(pred_5p, fix_68p, H):
    pred_5p_ = pred_5p.detach().cpu().numpy()
    fix_68p_ = fix_68p.detach().cpu().numpy()
    M = []
    align_5p = []
    for i in range(pred_5p_.shape[0]):
        M_, align_5p_ = estimate_align_5p(pred_5p_[i], fix_68p_[i], H)
        M.append(M_)
        align_5p.append(align_5p_)
    M = torch.tensor(np.array(M), dtype=torch.float32).to(pred_5p.device)
    align_5p = torch.tensor(np.array(align_5p), dtype=torch.float32).to(pred_5p.device)
    return M, align_5p

def estimate_align_5p(pred_5p, fix_68p, H):
    fix_5p = extract_5p(fix_68p)
    fix_5p[:, -1] = H - 1 - fix_5p[:, -1]
    pred_5p[:, -1] = H - 1 - pred_5p[:, -1]
    tform = trans.SimilarityTransform()
    tform.estimate(pred_5p, fix_5p)
    M = tform.params
    if np.linalg.det(M) == 0:
        M = np.eye(3)
    pred_5p_homo = np.concatenate((pred_5p, np.ones((5,1))), axis=-1)
    align_5p = M @ pred_5p_homo.transpose()
    align_5p = (align_5p.transpose())[:,:2]
    align_5p[:, -1] = H - 1 - align_5p[:, -1]
    return M[0:2, :], align_5p

def save_coeff(pred_coeffs_dict, name):
    pred_coeffs = {key:pred_coeffs_dict[key].detach().cpu().numpy() for key in pred_coeffs_dict}
    savemat(name, pred_coeffs)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument('--curriculum', type=str, default='FFHQ_512')
    parser.add_argument('--image_dir', type=str, default='data/FFHQ/img/')
    parser.add_argument('--img_output_dir', type=str, default='data/FFHQ/align')
    parser.add_argument('--mat_output_dir', type=str, default='data/FFHQ/mat')

    opt = parser.parse_args()
    metadata = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(metadata, 0)

    bfm_dir = metadata['bfm_path']
    face_dir = metadata['face_path']
    
    img_output_dir = os.path.join(opt.img_output_dir)
    os.makedirs(img_output_dir, exist_ok=True)
    mat_output_dir = os.path.join(opt.mat_output_dir)
    os.makedirs(mat_output_dir, exist_ok=True)

    net_recon, face_model, renderer = setup_model(face_dir, bfm_dir, device)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('checkpoints/face_ckpt/shape_predictor_68_face_landmarks.dat')
    lm3d_std = load_lm3d(bfm_dir) 
    
    all_img_path = sorted(glob.glob(os.path.join(opt.image_dir,'*.png')))
    
    # preprocess
    for i in range(len(all_img_path)):
        img_path = all_img_path[i]
        img_name = img_path.split(os.path.sep)[-1]
        
        image_pil = Image.open(img_path).convert("RGB")
        image_pil = image_pil.resize((512, 512))
        image = np.array(image_pil)
        lm = detect_image(image, detector, predictor)

        im_tensor, lm_tensor, img_old, lm_old = read_data(image_pil, lm, lm3d_std, device)
        pred_coeff, pred_face = reconstruct_image(net_recon, face_model, renderer, im_tensor)
        save_coeff(pred_coeff, os.path.join(mat_output_dir, img_name.replace('.png','.mat')))
        
        image_align = pre_img(face_model, img_old, lm_old, pred_coeff, device)
        
        out_img = np.array(image_align[0].cpu())
        out_img=out_img.transpose((1,2,0))[:,:,::-1]
        out_img=out_img*255
        
        cv2.imwrite(os.path.join(img_output_dir, img_name), out_img)