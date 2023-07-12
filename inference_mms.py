import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
import math
import statistics
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch.distributed as dist
import torchvision.models as models

from network.network import my_net
from utils.utils import get_device, check_accuracy, dice_loss, im_convert, label_to_onehot
from mms_dataloader import get_meta_split_data_loaders
from utils.data_utils import save_image
from utils.dice_loss import dice_coeff

import hydra
from omegaconf import DictConfig, OmegaConf

from accelerate import Accelerator
accelerator = Accelerator(fp16=True, split_batches=True)
device = accelerator.device

def pre_data(cfg, batch_size, num_workers, test_vendor):
    test_vendor = test_vendor

    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
        domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
        test_dataset = get_meta_split_data_loaders(cfg,
            test_vendor=test_vendor)

    val_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])

    label_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])

    # unlabel_dataset = ConcatDataset(
    #     [domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset])
    unlabel_dataset = domain_2_unlabeled_dataset

    print("before length of label_dataset", len(label_dataset))

    label_loader = DataLoader(dataset=label_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=True, pin_memory=False)

    unlabel_loader = DataLoader(dataset=unlabel_dataset, batch_size=batch_size, num_workers=num_workers,
                                shuffle=True, drop_last=True, pin_memory=False)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, drop_last=True, pin_memory=False)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, drop_last=True, pin_memory=False)

    print("after length of label_dataset", len(label_dataset))
    print("length of unlabel_dataset", len(unlabel_dataset))
    print("length of val_dataset", len(val_dataset))
    print("length of test_dataset", len(test_dataset))

    return label_loader, unlabel_loader, test_loader, val_loader, len(label_dataset), len(unlabel_dataset)

def save_once(image, pred, mask, flag, image_slice):
    pred = pred[:,0:3,:,:]
    real_mask = mask[:,0:3,:,:]
    mask = im_convert(real_mask, True)
    image = im_convert(image, False)
    pred = im_convert(pred, True)
    
    save_image(mask,'./pic/'+str(flag)+'/real_mask'+str(image_slice)+'.png')
    save_image(image,'./pic/'+str(flag)+'/image'+str(image_slice)+'.png')
    save_image(pred,'./pic/'+str(flag)+'/pred'+str(image_slice)+'.png')

def draw_many_img(model_path_l, model_path_r, test_loader):
    model_l = torch.load(model_path_l, map_location=device)
    model_r = torch.load(model_path_r, map_location=device)
    model_l = model_l.to(device)
    model_r = model_r.to(device)
    model_l.eval()
    model_r.eval()

    flag = '047'
    tot = 0
    tot_sub = []
    for minibatch in tqdm(test_loader):
        imgs = minibatch['img']
        mask = minibatch['mask']
        path_img = minibatch['path_img']
        imgs = imgs.to(device)
        mask = mask.to(device)
        if path_img[0][-10: -7] == flag:
            image_slice = path_img[0][-7:-4]
            with torch.no_grad():
                logits_l = model_l(imgs)
                logits_r = model_r(imgs)

            sof_l = F.softmax(logits_l, dim=1)
            sof_r = F.softmax(logits_r, dim=1)

            pred = (sof_l + sof_r) / 2
            pred = (pred > 0.5).float()

            save_once(imgs, pred, mask, flag, image_slice)

            # dice score
            tot = dice_coeff(pred[:, 0:3, :, :], mask[:, 0:3, :, :], device).item()

            tot_sub.append(tot)
        else:
            pass

    print('dice is ', sum(tot_sub)/len(tot_sub))

def ini_model(cfg):
    # two models with different init
    model_l = my_net('normalUnet', in_channels=1, num_classes=4)
    model_r = my_net('normalUnet', in_channels=1, num_classes=4)

    model_l = model_l.to(device)
    model_l.device = device

    model_r = model_r.to(device)
    model_r.device = device
    
    return model_l, model_r

def inference_dual(cfg, model_path_l, model_path_r, test_loader):

    #  Initialize model
    model_l, model_r = ini_model(cfg)

    model_l = accelerator.unwrap_model(model_l)
    model_l.load_state_dict(torch.load(model_path_l))

    model_r = accelerator.unwrap_model(model_r)
    model_r.load_state_dict(torch.load(model_path_r))

    model_l = model_l.to(device)
    model_l.eval()
    model_r = model_r.to(device)
    model_r.eval()

    tot = []
    tot_sub = []
    flag = '000'
    record_flag = {}

    for minibatch in tqdm(test_loader):
        imgs = minibatch['img']
        mask = minibatch['mask']
        path_img = minibatch['path_img']
        imgs = imgs.to(device)
        mask = mask.to(device)
        # print(flag)
        # print(path_img[0][-10: -7])
        if path_img[0][-10: -7] != flag:
            score = sum(tot_sub)/len(tot_sub)
            tot.append(score)
            tot_sub = []

            if score <= 0.7:
                record_flag[flag] = score
            flag = path_img[0][-10: -7]

        with torch.no_grad():
            logits_l = model_l(imgs)
            logits_r = model_r(imgs)

        sof_l = F.softmax(logits_l, dim=1)
        sof_r = F.softmax(logits_r, dim=1)
        pred = (sof_l + sof_r) / 2
        pred = (pred > 0.5).float()
        dice = dice_coeff(pred[:, 0:3, :, :], mask[:, 0:3, :, :], device).item()
        tot_sub.append(dice)
    tot.append(sum(tot_sub)/len(tot_sub))

    for i in range(len(tot)):
        tot[i] = tot[i] * 100

    print(tot)
    print(len(tot))
    print(sum(tot)/len(tot))
    print(statistics.stdev(tot))

@hydra.main(config_path='configs', config_name='mms_inference')
def main(cfg):
    print(f"Current working directory : {os.getcwd()}")
    batch_size = 1
    num_workers = 8
    test_vendor = 'D'
    
    model_path_l = '/home/hyaoad/remote/semi_medical/journal_code/journal_semi_dg/outputs/2022-08-02/20-20-54_'+str(test_vendor)+'_mms_AAAI_Unet05/l_'+str(test_vendor)+'_mms_AAAI_Unet05.pt'
    model_path_r = '/home/hyaoad/remote/semi_medical/journal_code/journal_semi_dg/outputs/2022-08-02/20-20-54_'+str(test_vendor)+'_mms_AAAI_Unet05/r_'+str(test_vendor)+'_mms_AAAI_Unet05.pt'

    label_loader, unlabel_loader, test_loader, val_loader, num_label_imgs, num_unsup_imgs = pre_data(
        cfg, batch_size=batch_size, num_workers=num_workers, test_vendor=test_vendor)

    # draw_many_img(model_path_l, model_path_r, test_loader)
    inference_dual(cfg, model_path_l, model_path_r, test_loader)

if __name__ == '__main__':
    main()
