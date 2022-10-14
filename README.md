# CoPo: Explicitly Controllable 3D-Aware Portrait Generation (Official Pytorch implementation)
![Teaser](imgs/Picture1.png)
### [Project page](https://junshutang.github.io/control/index.html) |   [Paper](https://arxiv.org/pdf/2209.05434) 
<!-- <br> -->
[Junshu Tang](https://junshutang.github.io/),  [Bo Zhang](https://bo-zhang.me/), Binxin Yang, Ting Zhang, [Dong Chen](https://www.microsoft.com/en-us/research/people/doch/), [Lizhuang Ma](https://dmcv.sjtu.edu.cn/), and [Fang Wen](https://www.microsoft.com/en-us/research/people/fangwen/).
<!-- <br> -->


## Abstract
>In contrast to the traditional avatar creation pipeline which is a costly process, contemporary generative approaches directly learn the data distribution from photographs. While plenty of works extend unconditional generative models and achieve some levels of controllability, it is still challenging to ensure multi-view consistency, especially in large poses. In this work, we propose a network that generates 3D-aware portraits while being controllable according to semantic parameters regarding pose, identity, expression and illumination. Our network uses neural scene representation to model 3D-aware portraits, whose generation is guided by a parametric face model that supports explicit control. While the latent disentanglement can be further enhanced by contrasting images with partially different attributes, there still exists noticeable inconsistency in non-face areas, e.g, hair and background, when animating expressions. We solve this by proposing a volume blending strategy in which we form a composite output by blending dynamic and static areas, with two parts segmented from the jointly learned semantic field. Our method outperforms prior arts in extensive experiments, producing realistic portraits with vivid expression in natural lighting when viewed from free viewpoints. It also demonstrates generalization ability to real images as well as out-of-domain data, showing great promise in real applications. 



## Installation

Install dependencies:
```bash
pip install -r requirements.txt
pip install -U git+https://github.com/fadel/pytorch_ema
````
Training requirements
- [Nvdiffrast](https://nvlabs.github.io/nvdiffrast/). We use Nvdiffrast which is a pytorch library that provides high-performance primitive operations for rasterization-based differentiable rendering.
  ```
  git clone https://github.com/NVlabs/nvdiffrast.git
  cd nvdiffrast/
  python setup.py install
  ```
- [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model).  Get access to BFM09 using this [link](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads). After getting the access, download `01_MorphableModel.mat`. In addition, we use an Expression Basis provided by [Guo et al.](https://github.com/Juyong/3DFace). Download the Expression Basis (Exp_Pca.bin) using this [link (google drive)](https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view?usp=sharing). Put them in `checkpoints/face_ckpt/BFM/`

- [Face Reconstruction Model](https://github.com/sicxu/Deep3DFaceRecon_pytorch). We use the network to extract identity, expression, lighting, and pose coefficients. Download the pretrained model `epoch_20.pth` and put it in `checkpoints/face_ckpt/face_ckp/recon_model`

- Face Recognition Model. We use the [ArcFace](https://github.com/deepinsight/insightface) for extracting the deep face feature. Download the pretrained model `ms1mv3_arcface_r50_fp16/backbone.pth` and put it in `checkpoints/face_ckpt/face_ckp/recog_model`

- Face Landmark Detection. Download `shape_predictor_68_face_landmarks.dat` from [Dlib](https://github.com/davisking/dlib) and put it in `checkpoints/face_ckpt/face_ckp/`.

- Face Parsing Network. Download `79999_iter.pth` from [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) and put in `checkpoints/face_ckpt/face_ckp/`.

## Quick Inference Using Pretrained Model
Download the pretrained model from [here](https://drive.google.com/drive/folders/1FHD6_F3RIDIyYfLRpw_ndfxG57dOlnck) and save them in `checkpoints/model`. For pretrained VAE decoder, please download our pretrained model from [here](https://drive.google.com/drive/folders/1gx3vTEXGefx14E7WDOH6B7rTYgV966Zk?usp=sharing) and save them in `checkpoints/vae_ckp/`. We provide a test sequence in [here](https://drive.google.com/drive/folders/1yxwFuMSoVbntRq13QnAGpaBkVutvppX6?usp=sharing). Please download `obama/mat/*.mat` and put them in `data/`. Then run the command.
```
python test.py --curriculum FFHQ_512 --load_dir checkpoints/model/ --output_dir results --blend_mode both  --seeds 41
```


## Train from Scratch
#### 1) Prepair training data

- **[FFHQ](https://github.com/NVlabs/ffhq-dataset).** Download `images1024x1024` and resize to 512x512 resolution and put them in `data/ffhq/img`.

- **Preprocess.** Run the command, modify `aligned_image_path` and `mat_path`. 
    ````
    python preprocess.py --curriculum FFHQ_512 --image_dir data/ffhq/img --img_output_dir aligned_image_path --mat_output_dir mat_path
    ````

- **[RAVDESS](https://zenodo.org/record/1188976#.Y0kQUHZBzmF).** We select 10 videos and sample 400 images from each video, resulting in 96,000 images in total. We extract expression coefficients for each image. You can download these data from [here](https://drive.google.com/drive/folders/1yxwFuMSoVbntRq13QnAGpaBkVutvppX6?usp=sharing).
#### 2) VAE-GAN training
```
python train_vae.py --curriculum VAE_ALL --output_dir results/vae  --render_dir results/render --weight 0.0025 --factor id # id/exp/gamma
```
- You can also download the pretrained VAE decoder from [here](https://drive.google.com/drive/folders/1gx3vTEXGefx14E7WDOH6B7rTYgV966Zk?usp=sharing) and save them in `checkpoints/vae_ckp/`. 

#### 3) Imitation learning
```
python train_control.py --curriculum FFHQ_512 --output_dir train_ffhq_512 --warmup1 5000 --warmup2 20000
```
#### 4) Disentanglement learning
```
python train_control.py --curriculum FFHQ_512 --output_dir train_ffhq_512 --load_dir load_dir --set_step 20001 --warmup1 5000 --warmup2 20000 --second
```

## Citation
If you use this code for your research, please cite our papers.
```
@article{tang2022explicitly,
  title={Explicitly Controllable 3D-Aware Portrait Generation},
  author={Tang, Junshu and Zhang, Bo and Yang, Binxin and Zhang, Ting and Chen, Dong and Ma, Lizhuang and Wen, Fang},
  journal={arXiv preprint arXiv:2209.05434},
  year={2022}
}
```

## Acknowledgments
This code borrows heavily from [PiGAN](https://github.com/marcoamonteiro/pi-GAN), [StyleGAN2](https://github.com/NVlabs/stylegan2) and [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch).