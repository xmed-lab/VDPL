import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from scgm_dataloader import get_meta_split_data_loaders
from utils.data_utils import save_image
from utils.dice_loss import dice_coeff
# from draw_dataloader import OneImageFolder
from utils.hausdorff import hausdorff_distance
import yaml

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

    # new_labeldata_num = len(unlabel_dataset) // len(label_dataset) + 1
    # new_label_dataset = label_dataset
    # for i in range(new_labeldata_num):
    #     new_label_dataset = ConcatDataset([new_label_dataset, label_dataset])
    # label_dataset = new_label_dataset

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

def inference(model_path, test_loader):

    model_l = torch.load(model_path)
    model_l = model_l.to(device)
    model_l.eval()

    test_loss = []
    loss = 0
    dice_loss_lv_l = 0
    dice_loss_myo_l = 0
    dice_loss_rv_l = 0
    dice_loss_bg_l = 0

    for batch in tqdm(test_loader):
        imgs, mask, _ = batch
        imgs = imgs.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            logits = model_l(imgs)

        sof_l = F.softmax(logits, dim=1)
        dice_loss_lv_l = dice_loss(sof_l[:, 0, :, :], mask[:, 0, :, :])
        dice_loss_myo_l = dice_loss(sof_l[:, 1, :, :], mask[:, 1, :, :])
        dice_loss_rv_l = dice_loss(sof_l[:, 2, :, :], mask[:, 2, :, :])
        dice_loss_bg_l = dice_loss(sof_l[:, 3, :, :], mask[:, 3, :, :])
        loss = dice_loss_lv_l + dice_loss_myo_l + dice_loss_rv_l + dice_loss_bg_l

        test_loss.append(loss.item())

    test_dice, test_dice_lv, test_dice_myo, test_dice_rv  = check_accuracy(test_loader, model_l)

    test_loss = sum(test_loss) / len(test_loss)
    print(
        f"[ Test | test_loss = {test_loss:.5f} test_dice = {test_dice:.5f}")

def draw_img(model_path_l, model_path_r, test_loader, domain):
    model_l = torch.load(model_path_l, map_location=device)
    model_r = torch.load(model_path_r, map_location=device)
    model_l = model_l.to(device)
    model_r = model_r.to(device)
    model_l.eval()
    model_r.eval()

    dataiter = iter(test_loader)
    minibatch = dataiter.next()
    imgs = minibatch['img']
    aug_img = minibatch['aug_img']
    mask = minibatch['mask']
    img_path = minibatch['path_img']
    imgs = imgs.to(device)
    aug_img = aug_img.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        logits_l, _ = model_l(imgs)
        logits_r, _ = model_r(imgs)

        sof_l = F.softmax(logits_l, dim=1)
        sof_r = F.softmax(logits_r, dim=1)

    sof_t = (sof_l + sof_r) / 2
    pred = (sof_t > 0.5).float()

    # dice score
    tot = dice_coeff(pred[:, 0, :, :], mask[:, 0, :, :], device).item()

    image = imgs
    # torch.set_printoptions(threshold=np.inf)
    # with open('./test.txt', 'wt') as f:
    #     print(onehot_predmax==mask, file=f)
    pred = pred[:,0,:,:]
    real_mask = mask[:,0,:,:]

    print(img_path[0])
    # image slice
    # print(img_path[0][-7:-4])
    # patient num
    # print(img_path[0][-10: -7])
    print("dice score: ", tot)
    real_mask = im_convert(real_mask, False)
    image = im_convert(image, False)
    pred = im_convert(pred, False)
    save_image(real_mask,'./fpic/scgm_gt'+str(domain)+'.png')
    save_image(image,'./fpic/scgm_image'+str(domain)+'.png')
    save_image(pred,'./fpic/scgm_pred'+str(domain)+'.png')

def save_once(image, pred, mask, flag, image_slice):
    pred = pred[:,0,:,:]
    real_mask = mask[:,0,:,:]
    mask = im_convert(real_mask, False)
    image = im_convert(image, False)
    pred = im_convert(pred, False)
    
    save_image(mask,'/home/hyaoad/remote/semi_medical/journal_code/journal_semi_dg/pic/'+str(flag)+'/real_mask_'+str(image_slice)+'.png')
    save_image(image,'/home/hyaoad/remote/semi_medical/journal_code/journal_semi_dg/pic/'+str(flag)+'/image_'+str(image_slice)+'.png')
    save_image(pred,'/home/hyaoad/remote/semi_medical/journal_code/journal_semi_dg/pic/'+str(flag)+'/pred_'+str(image_slice)+'.png')

def draw_many_img(cfg, model_path_l, model_path_r, test_loader):
    # Initialize model
    model_l, model_r = ini_model(cfg)

    model_l = accelerator.unwrap_model(model_l)
    model_l.load_state_dict(torch.load(model_path_l))

    model_r = accelerator.unwrap_model(model_r)
    model_r.load_state_dict(torch.load(model_path_r))

    model_l = model_l.to(device)
    model_l.eval()
    model_r = model_r.to(device)
    model_r.eval()

    flag = '000'
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
            tot = dice_coeff(pred[:, 0, :, :], mask[:, 0, :, :], device).item()
            tot_sub.append(tot)
        else:
            pass

    print('dice is ', sum(tot_sub)/len(tot_sub))

def ini_model(cfg):
    # two models with different init
    model_l = my_net('normalUnet', in_channels=1, num_classes=2)
    model_r = my_net('normalUnet', in_channels=1, num_classes=2)

    model_l = model_l.to(device)
    model_l.device = device

    model_r = model_r.to(device)
    model_r.device = device
    
    return model_l, model_r

def inference_dual(cfg, model_path_l, model_path_r, test_loader):
    # Initialize model
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
    tot_hsd = []
    tot_sub_hsd = []
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

            tot_hsd.append(sum(tot_sub_hsd)/len(tot_sub_hsd))
            tot_sub_hsd = []

            if score <= 0.9:
                record_flag[flag] = score
            flag = path_img[0][-10: -7]

        with torch.no_grad():
            logits_l = model_l(imgs)
            logits_r = model_r(imgs)

        sof_l = F.softmax(logits_l, dim=1)
        sof_r = F.softmax(logits_r, dim=1)
        pred = (sof_l + sof_r) / 2
        pred = (pred > 0.5).float()
        dice = dice_coeff(pred[:, 0, :, :], mask[:, 0, :, :], device).item()
        hsd = hausdorff_distance(pred[:, 0, :, :], mask[:, 0, :, :])
        tot_sub.append(dice)
        tot_sub_hsd.append(hsd)
    tot.append(sum(tot_sub)/len(tot_sub))
    tot_sub.append(sum(tot_sub_hsd)/len(tot_sub_hsd))

    for i in range(len(tot)):
        tot[i] = tot[i] * 100

    print('All')
    print(tot)
    # with open('/home/hyaoad/remote/semi_medical/journal_code/journal_semi_dg/output.csv', 'w', newline='') as csvfile:
    #     print('writing')
    #     writer = csv.writer(csvfile)
    #     for value in tot:
    #         value = round(value, 2)
    #         writer.writerow([value])
    print('dice')
    print(len(tot))
    print(sum(tot)/len(tot))
    print(statistics.stdev(tot))

    print('hausdorff')
    # print(tot_hsd)
    print(sum(tot_hsd)/len(tot_hsd))
    print(statistics.stdev(tot_hsd))
    print(record_flag)

@hydra.main(config_path='configs', config_name='scgm_inference')
def main(cfg):
    print(f"Current working directory : {os.getcwd()}")
    batch_size = 1
    num_workers = 4
    test_vendor = 'C'
    base_path = hydra.utils.get_original_cwd()
    # model_path_l = base_path + '/model/l_scgm_test_'+str(test_vendor)+'.pt'
    # model_path_r = base_path + '/model/r_scgm_test_'+str(test_vendor)+'.pt'
    # model_path_l = '/home/hyaoad/remote/semi_medical/journal_code/journal_semi_dg/outputs/2022-08-25/20-22-18_C_scgm_AAAI_Unet20/l_C_scgm_AAAI_Unet20.pt'
    # model_path_r = '/home/hyaoad/remote/semi_medical/journal_code/journal_semi_dg/outputs/2022-08-25/20-22-18_C_scgm_AAAI_Unet20/r_C_scgm_AAAI_Unet20.pt'
    
    model_path_l = '/home/hyaoad/remote/semi_medical/journal_code/journal_semi_dg/outputs/2023-06-30/16-55-41_C_scgm_CPS_Unet20/l_C_scgm_CPS_Unet20.pt'
    model_path_r = '/home/hyaoad/remote/semi_medical/journal_code/journal_semi_dg/outputs/2023-06-30/16-55-41_C_scgm_CPS_Unet20/r_C_scgm_CPS_Unet20.pt'
    label_loader, unlabel_loader, test_loader, val_loader, num_label_imgs, num_unsup_imgs = pre_data(
        cfg, batch_size=batch_size, num_workers=num_workers, test_vendor=test_vendor)

    # id = '014123'
    # id = '000003'
    # img_path = '/home/listu/code/semi_medical/mnms_split_2D/data/Labeled/vendorA/'+ id +'.npz'
    # mask_path = '/home/listu/code/semi_medical/mnms_split_2D/mask/Labeled/vendorA/'+ id +'.png'
    # re_path = '/home/listu/code/semi_medical/mnms_split_2D_re/Labeled/vendorA/'+ id +'.npz'
    # fourier_path = '/home/listu/code/semi_medical/mnms_split_2D/data/Labeled/vendorB/center2/000005.npz'
    # one_image_data = OneImageFolder(img_path, mask_path, re_path, fourier_path)
    # one_image_loader = DataLoader(dataset=one_image_data, batch_size=1, shuffle=False, drop_last=True, pin_memory=True)

    # draw_img(model_path_l, model_path_r, test_loader, test_vendor)
    draw_many_img(cfg, model_path_l, model_path_r, test_loader)
    # inference_dual(cfg, model_path_l, model_path_r, test_loader)

if __name__ == '__main__':
    main()
