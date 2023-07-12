import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
import math
import statistics
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch.distributed as dist
import torchvision.models as models

from network.network import my_net
from utils.utils import get_device, check_accuracy, dice_loss, im_convert, label_to_onehot
from chd_dataloader import get_meta_split_data_loaders
from utils.data_utils import save_image
from utils.dice_loss import dice_coeff

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

device = 'cuda'
os.environ['WANDB_MODE'] = 'dryrun'

def pre_data(cfg, batch_size, num_workers, test_vendor):
    test_vendor = test_vendor

    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
        domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
        test_dataset = get_meta_split_data_loaders(cfg=cfg,
            test_vendor=test_vendor, image_size=224)

    val_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])

    label_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])

    unlabel_dataset = domain_2_unlabeled_dataset

    print("before length of label_dataset", len(label_dataset))

    label_loader = DataLoader(dataset=label_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=True, pin_memory=False)

    unlabel_loader = DataLoader(dataset=unlabel_dataset, batch_size=batch_size, num_workers=num_workers,
                                shuffle=True, drop_last=True, pin_memory=False)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, drop_last=True, pin_memory=False)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=True, drop_last=True, pin_memory=False)

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

def ini_model():
    # two models with different init
    model_l = my_net('normalUnet', in_channels=1, num_classes=11)
    model_r = my_net('normalUnet', in_channels=1, num_classes=11)

    model_l = model_l.to(device)
    model_l.device = device

    model_r = model_r.to(device)
    model_r.device = device
    
    return model_l, model_r

def inference_dual(model_path_l, model_path_r, test_loader):
    # Initialize model
    model_l, model_r = ini_model()

    model_l.load_state_dict(torch.load(model_path_l, map_location=device))
    model_l = model_l.to(device)
    model_l.eval()

    model_r.load_state_dict(torch.load(model_path_r, map_location=device))
    model_r = model_r.to(device)
    model_r.eval()
    
    tot_1 = 0
    tot_2 = 0
    tot_3 = 0
    tot_4 = 0
    tot_5 = 0
    tot_6 = 0
    tot_7 = 0
    tot_8 = 0
    tot_9 = 0
    tot_10 = 0

    for minibatch in tqdm(test_loader):
        imgs = minibatch['img']
        mask = minibatch['mask']
        imgs = imgs.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            logits_l = model_l(imgs)
            logits_r = model_r(imgs)

        sof_l = F.softmax(logits_l, dim=1)
        sof_r = F.softmax(logits_r, dim=1)

        pred = (sof_l + sof_r) / 2
        pred = (pred > 0.5).float()

        tot_1 += dice_coeff(pred[:, 0, :, :], mask[:, 0, :, :], device).item()
        tot_2 += dice_coeff(pred[:, 1, :, :], mask[:, 1, :, :], device).item()
        tot_3 += dice_coeff(pred[:, 2, :, :], mask[:, 2, :, :], device).item()
        tot_4 += dice_coeff(pred[:, 3, :, :], mask[:, 3, :, :], device).item()
        tot_5 += dice_coeff(pred[:, 4, :, :], mask[:, 4, :, :], device).item()
        tot_6 += dice_coeff(pred[:, 5, :, :], mask[:, 5, :, :], device).item()
        tot_7 += dice_coeff(pred[:, 6, :, :], mask[:, 6, :, :], device).item()
        tot_8 += dice_coeff(pred[:, 7, :, :], mask[:, 7, :, :], device).item()
        tot_9 += dice_coeff(pred[:, 8, :, :], mask[:, 8, :, :], device).item()
        tot_10 += dice_coeff(pred[:, 9, :, :], mask[:, 9, :, :], device).item()
    
    dice_1 = tot_1/len(test_loader)
    dice_2 = tot_2/len(test_loader)
    dice_3 = tot_3/len(test_loader)
    dice_4 = tot_4/len(test_loader)
    dice_5 = tot_5/len(test_loader)
    dice_6 = tot_6/len(test_loader)
    dice_7 = tot_7/len(test_loader)
    dice_8 = tot_8/len(test_loader)
    dice_9 = tot_9/len(test_loader)
    dice_10 = tot_10/len(test_loader)
    dice = (dice_1 + dice_2 + dice_3 + dice_4 + dice_5 + dice_6 + dice_7 + dice_8 + dice_9 + dice_10) / 10

    print(dice)

def wandb_init(cfg: DictConfig):
    wandb.init(
        project='Journal-semi-chd',
        entity='nekokiku',
        name=cfg.exp_name,
        notes=cfg.exp_desc,
        save_code=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    # safe the final config for reproducing
    OmegaConf.save(config=cfg, f=os.path.join(wandb.run.dir, 'conf.yaml'))

@hydra.main(config_path='configs', config_name='chd_inference', )
def main(cfg):
    wandb_init(cfg)
    batch_size = 1
    num_workers = 8
    test_vendor = 'D'
    
    model_path_l = '/home/hyaoad/remote/semi_medical/journal_code/journal_semi_dg/model/18-52-31_'+test_vendor+'_chd_now_Unet02_fix/l_'+test_vendor+'_chd_now_Unet02_fix.pt'
    model_path_r = '/home/hyaoad/remote/semi_medical/journal_code/journal_semi_dg/model/18-52-31_'+test_vendor+'_chd_now_Unet02_fix/r_'+test_vendor+'_chd_now_Unet02_fix.pt'

    label_loader, unlabel_loader, test_loader, val_loader, num_label_imgs, num_unsup_imgs = pre_data(cfg,
        batch_size=batch_size, num_workers=num_workers, test_vendor=test_vendor)
    
    # draw_many_img(model_path_l, model_path_r, test_loader)
    inference_dual(model_path_l, model_path_r, test_loader)

if __name__ == '__main__':
    main()
