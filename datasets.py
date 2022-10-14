"""Datasets"""

import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import PIL
import random

import scipy.io as scio

class FFHQ_Mat(Dataset):
    """FFHQ Mat Dataset"""

    # def __init__(self, dataset_path, posepath, lmpath, bfmpath, img_size, **kwargs):
    def __init__(self, dataset_path, posepath, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        self.posepath = posepath
        self.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=0)])
        
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name = self.data[index]
        mat = scio.loadmat(os.path.join(self.posepath, image_name.split('/')[-1].replace('png', 'mat')))
        img = PIL.Image.open(image_name)
        img = self.transform(img)
        
        return img, mat
    

class VAE_Mat(Dataset):

    def __init__(self, ryspath, ffhqpath, **kwargs):
        super().__init__()
        self.rydata = glob.glob(ryspath)
        self.ffhqdata = glob.glob(ffhqpath)
        assert len(self.rydata) > 0, "Can't find ryser data; make sure you specify the path to your dataset"
        assert len(self.ffhqdata) > 0, "Can't find  ffhq data; make sure you specify the path to your dataset"

        self.data = self.rydata + self.ffhqdata
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mat_name = self.data[index]
        mat = scio.loadmat(mat_name)
        return mat


def get_dataset(name, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
        num_workers=0
    )
    return dataloader, len(dataset)

def get_dataset_distributed(name, world_size, rank, batch_size, **kwargs):
    dataset = globals()[name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=4,
    )

    return dataloader, len(dataset)
