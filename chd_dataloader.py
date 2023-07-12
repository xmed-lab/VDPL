from ast import Num
import os
from posixpath import split
from traceback import print_tb
import cv2
import math
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from random import sample, random, seed
from utils.data_utils import colorful_spectrum_mix, fourier_transform

# Data directories
Base_dir = '/home/hyaoad/remote'

Domain_A_volume_dir = os.path.join(Base_dir, 'semi_medical/CT_2D/cq')
Domain_A_data_dir = Domain_A_volume_dir + '_slice'

Domain_B_volume_dir = os.path.join(Base_dir, 'semi_medical/CT_2D/gx')
Domain_B_data_dir = Domain_B_volume_dir + '_slice'

Domain_C_volume_dir = os.path.join(Base_dir, 'semi_medical/CT_2D/gz')
Domain_C_data_dir = Domain_C_volume_dir + '_slice'

Domain_D_volume_dir = os.path.join(Base_dir, 'semi_medical/CT_2D/rmyy')
Domain_D_data_dir = Domain_D_volume_dir + '_slice'

Domain_volume_dir = [Domain_A_volume_dir, Domain_B_volume_dir, Domain_C_volume_dir, Domain_D_volume_dir]
Domain_data_dir = [Domain_A_data_dir, Domain_B_data_dir, Domain_C_data_dir, Domain_D_data_dir]
Num_of_volume = [22, 27, 25, 25]

def get_meta_split_data_loaders(cfg, test_vendor='D', image_size=512):
    seed(14)
    return get_data_loader_folder(cfg, Domain_volume_dir, Domain_data_dir, target_domain=test_vendor)

def get_data_loader_folder(cfg, volume_dir, data_dir, target_domain='A'):
    if target_domain == 'A':
        domain_1_volume_dir = volume_dir[1]
        domain_2_volume_dir = volume_dir[2]
        domain_3_volume_dir = volume_dir[3]

        domain_1_data_dir = data_dir[1]
        domain_2_data_dir = data_dir[2]
        domain_3_data_dir = data_dir[3]

        domain_1_num = Num_of_volume[1]
        domain_2_num = Num_of_volume[2]
        domain_3_num = Num_of_volume[3]

        fourier_dir = [data_dir[1], data_dir[2], data_dir[3]]

        test_volume_dir = volume_dir[0]
        test_data_dir = data_dir[0]
        test_num = Num_of_volume[0]

    elif target_domain == 'B':
        domain_1_volume_dir = volume_dir[0]
        domain_2_volume_dir = volume_dir[2]
        domain_3_volume_dir = volume_dir[3]

        domain_1_data_dir = data_dir[0]
        domain_2_data_dir = data_dir[2]
        domain_3_data_dir = data_dir[3]

        domain_1_num = Num_of_volume[0]
        domain_2_num = Num_of_volume[2]
        domain_3_num = Num_of_volume[3]

        fourier_dir = [data_dir[0], data_dir[2], data_dir[3]]

        test_volume_dir = volume_dir[1]
        test_data_dir = data_dir[1]
        test_num = Num_of_volume[1]

    elif target_domain == 'C':
        domain_1_volume_dir = volume_dir[0]
        domain_2_volume_dir = volume_dir[1]
        domain_3_volume_dir = volume_dir[3]

        domain_1_data_dir = data_dir[0]
        domain_2_data_dir = data_dir[1]
        domain_3_data_dir = data_dir[3]

        domain_1_num = Num_of_volume[0]
        domain_2_num = Num_of_volume[1]
        domain_3_num = Num_of_volume[3]

        fourier_dir = [data_dir[0], data_dir[1], data_dir[3]]

        test_volume_dir = volume_dir[2]
        test_data_dir = data_dir[2]
        test_num = Num_of_volume[2]

    elif target_domain == 'D':
        domain_1_volume_dir = volume_dir[0]
        domain_2_volume_dir = volume_dir[1]
        domain_3_volume_dir = volume_dir[2]

        domain_1_data_dir = data_dir[0]
        domain_2_data_dir = data_dir[1]
        domain_3_data_dir = data_dir[2]

        domain_1_num = Num_of_volume[0]
        domain_2_num = Num_of_volume[1]
        domain_3_num = Num_of_volume[2]

        fourier_dir = [data_dir[0], data_dir[1], data_dir[2]]

        test_volume_dir = volume_dir[3]
        test_data_dir = data_dir[3]
        test_num = Num_of_volume[3]
    else:
        raise ValueError('Invalid target domain')
    
    print("loading labeled dateset")
    domain_1_labeled_dataset = BaseDataSets(domain_1_volume_dir, domain_1_data_dir, fourier_dir=fourier_dir, split="train", mode="semi", num_of_volume=domain_1_num, ratio=cfg.ratio)
    domain_2_labeled_dataset = BaseDataSets(domain_2_volume_dir, domain_2_data_dir, fourier_dir=fourier_dir, split="train", mode="semi", num_of_volume=domain_2_num, ratio=cfg.ratio)
    domain_3_labeled_dataset = BaseDataSets(domain_3_volume_dir, domain_3_data_dir, fourier_dir=fourier_dir, split="train", mode="semi", num_of_volume=domain_3_num, ratio=cfg.ratio)

    print("loading unlabeled dateset")
    domain_1_unlabeled_dataset = BaseDataSets(domain_1_volume_dir, domain_1_data_dir, fourier_dir=fourier_dir, split="train", mode="unlabeled", num_of_volume=domain_1_num)
    domain_2_unlabeled_dataset = BaseDataSets(domain_2_volume_dir, domain_2_data_dir, fourier_dir=fourier_dir, split="train", mode="unlabeled", num_of_volume=domain_2_num)
    domain_3_unlabeled_dataset = BaseDataSets(domain_3_volume_dir, domain_3_data_dir, fourier_dir=fourier_dir, split="train", mode="unlabeled", num_of_volume=domain_3_num)

    print("loading test dateset")
    test_dataset = BaseDataSets(test_volume_dir, test_data_dir, split="test", mode="semi", num_of_volume=test_num)

    return domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
              domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
                test_dataset

def make_dataset(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images

def fourier_augmentation(img, tar_img, mode, alpha):
    # transfer image from PIL to numpy
    img = np.array(img)
    tar_img = np.array(tar_img)
    img = img[:,:,np.newaxis]
    tar_img = tar_img[:,:,np.newaxis]

    # the mode comes from the paper "A Fourier-based Framework for Domain Generalization"
    if mode == 'AS':
        # print("using AS mode")
        aug_img, aug_tar_img = fourier_transform(img, tar_img, L=0.01, i=1)
    elif mode == 'AM':
        # print("using AM mode")
        aug_img, aug_tar_img = colorful_spectrum_mix(img, tar_img, alpha=alpha)
    else:
        print("mode name error")

    aug_img = np.squeeze(aug_img)
    aug_tar_img = np.squeeze(aug_tar_img)

    return aug_img

class BaseDataSets(Dataset):
    def __init__(
        self,
        volume_dir=None,
        data_dir=None,
        fourier_dir=None,
        split="train",
        mode="semi",
        num_of_volume=None,
        transform=None,
        ratio=0.2,
    ): 
        self.volume_dir = volume_dir
        self.data_dir = data_dir # for example: ../dataset/
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.num = num_of_volume
        self.fourier_imgs = []
        self.fourier_transform = None
        
        if self.split == "train" and mode == "semi":
            self.ratio = ratio # the ratio of labeled data
            self.num = math.ceil(self.num * self.ratio) # the number of labeled volume
            if self.num == 0: # at least one volume is labeled
                self.num = 1
        else:
            self.ratio = 1.0
            
        if self.split == "train":
            self.transform = A.Compose([
                A.RandomRotate90(),
                A.RandomScale(p=0.8),
                A.Resize(width=512, height=512),
                A.HorizontalFlip(p=0.5),
                # A.ColorJitter(),
                # A.GaussianBlur(blur_limit = 5, sigma_limit = 0)
                #add here 
            ])
            self.fourier_transform = A.Compose([
                A.Resize(width=512, height=512),
                A.HorizontalFlip(p=0.5)
            ])
        elif self.split == "test":
            self.transform = A.Compose([
                A.Resize(width=512, height=512),
            ])
        
        print("num is:", self.num)
        volume_roots = sorted(glob(self.volume_dir + "/*image.nii.gz"))
        labeled_volume_roots = volume_roots[:self.num]
        labeled_volume_names = [os.path.basename(v).split('.')[0] for v in labeled_volume_roots]
        
        data_roots = sorted(make_dataset(self.data_dir))
        data_names = [os.path.basename(v).split('.')[0] for v in data_roots]

        # print("labeled_volume_roots",labeled_volume_names)
        # print("data_roots",data_roots)
        for data_name in data_names:
            for volume_name in labeled_volume_names:
                if volume_name in data_name:
                    self.sample_list.append(data_name)
        
        # for Fourier dirs
        if self.split == "train":
            for num_set in range(len(fourier_dir)):
                data_roots = sorted(make_dataset(fourier_dir[num_set]))
                for num_data in range(len(data_roots)):
                    self.fourier_imgs.append(data_roots[num_data])

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(self.data_dir + '/' + case + '.h5', 'r')
        image = h5f["image"][:]
        label = h5f["label"][:]

        if self.transform:
            trans = self.transform(image=image, mask=label)
            image, label = trans['image'], trans['mask']

        if self.split == 'train':
            fourier_path = sample(self.fourier_imgs, 1)[0]
            h5f_fourier = h5py.File(fourier_path)
            fourier_img = h5f_fourier["image"][:]
            fourier = self.fourier_transform(image=fourier_img)['image']
            fourier = fourier_augmentation(image, fourier, mode="AS", alpha=0.5)
            fourier = torch.unsqueeze(torch.from_numpy(fourier), 0)
            fourier = fourier.type(torch.FloatTensor)
        
        if self.split == "test":
            fourier = torch.tensor([0])

        # label to one-hot label
        w, h = label.shape
        label = label.reshape(-1)
        label = (label - 1) % 11
        oh_label = np.zeros((label.size, 11))
        oh_label[np.arange(label.size), label] = 1
        oh_label = oh_label.reshape([w, h, -1])
        oh_label = oh_label.transpose(2, 0, 1)

        image = torch.unsqueeze(torch.from_numpy(image), 0)
        oh_label = torch.from_numpy(oh_label)
        image = image.type(torch.FloatTensor)
        oh_label = oh_label.type(torch.FloatTensor)

        outputs = {
            "img": image,
            "aug_img": fourier, 
            "mask": oh_label
            }
        outputs["idx"] = idx
        return outputs

if __name__ == '__main__':
    volume_dir = '/home/hyaoad/remote/semi_medical/CT_2D/cq'
    data_dir = '/home/hyaoad/remote/semi_medical/CT_2D/cq' + '_slice'
    fourier_dir = ['/home/hyaoad/remote/semi_medical/CT_2D/cq' + '_slice']

    dataset = BaseDataSets(
        volume_dir=volume_dir,
        data_dir=data_dir,
        fourier_dir=fourier_dir,
        split="train",
        mode="semi",
        num_of_volume=22,
        transform=None,
        ratio=0.1,)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    dataiter = iter(dataloader)
    output = dataiter.next()
    print(output["img"].shape)
    print(output["mask"].shape)
    print(output["aug_img"].shape)
    print(output["mask"].max())
    print(output["mask"].min())
