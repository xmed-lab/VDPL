"""
Main file for training model
"""
# import common libraries
import os
import sys
import math
import numpy as np
from tqdm import tqdm
# import libraries about pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch.distributed as dist
import torchvision.models as models
# import from other file
from network.network import my_net
from utils.utils import check_accuracy, check_accuracy_dual, label_to_onehot, fetch_scheduler
from chd_dataloader import get_meta_split_data_loaders
from utils.dice_loss import dice_coeff
import utils.mask_gen as mask_gen
from utils.custom_collate import SegCollate
from network.mixstyle import (
    MixStyle, random_mixstyle, activate_mixstyle, run_with_mixstyle,
    deactivate_mixstyle, crossdomain_mixstyle, run_without_mixstyle
)
# import config and logging
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from accelerate import Accelerator
accelerator = Accelerator(fp16=False, split_batches=True)
device = accelerator.device

# for offline running
# os.environ['WANDB_MODE'] = 'dryrun'

def pre_data(cfg):
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    test_vendor = cfg.test_vendor

    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
        domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
        test_dataset = get_meta_split_data_loaders(cfg, test_vendor=test_vendor, image_size=224)

    val_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])

    label_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])

    unlabel_dataset = ConcatDataset(
        [domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset])

    accelerator.print("before length of label_dataset", len(label_dataset))

    new_labeldata_num = len(unlabel_dataset) // len(label_dataset) + 1
    new_label_dataset = label_dataset
    for i in range(new_labeldata_num):
        new_label_dataset = ConcatDataset([new_label_dataset, label_dataset])
    label_dataset = new_label_dataset

    # For CutMix
    mask_generator = mask_gen.BoxMaskGenerator(prop_range=cfg.cutmix_mask_prop_range, n_boxes=cfg.cutmix_boxmask_n_boxes,
                                               random_aspect_ratio=cfg.cutmix_boxmask_fixed_aspect_ratio,
                                               prop_by_area=cfg.cutmix_boxmask_by_size, 
                                               within_bounds=cfg.cutmix_boxmask_outside_bounds,
                                               invert=cfg.cutmix_boxmask_no_invert)

    add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
        mask_generator
    )
    collate_fn = SegCollate()
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)

    label_loader = DataLoader(dataset=label_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    unlabel_loader_0 = DataLoader(dataset=unlabel_dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=True, drop_last=True, pin_memory=False, collate_fn=mask_collate_fn)

    unlabel_loader_1 = DataLoader(dataset=unlabel_dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=True, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    accelerator.print("after length of label_dataset", len(label_dataset))
    accelerator.print("length of unlabel_dataset", len(unlabel_dataset))
    accelerator.print("length of val_dataset", len(val_dataset))
    accelerator.print("length of test_dataset", len(test_dataset))
    accelerator.print("batch_size", batch_size)

    return label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, len(label_dataset), len(unlabel_dataset)

# Dice loss
def dice_loss(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 0.1  # 1e-12

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    #A_sum = torch.sum(tflat * iflat)
    #B_sum = torch.sum(tflat * tflat)
    loss = ((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth)).mean()

    return 1 - loss

def total_dice_loss(pred, target):

    dice_loss_1 = dice_loss(pred[:, 0, :, :], target[:, 0, :, :])
    dice_loss_2 = dice_loss(pred[:, 1, :, :], target[:, 1, :, :])
    dice_loss_3 = dice_loss(pred[:, 2, :, :], target[:, 2, :, :])
    dice_loss_4 = dice_loss(pred[:, 3, :, :], target[:, 3, :, :])
    dice_loss_5 = dice_loss(pred[:, 4, :, :], target[:, 4, :, :])
    dice_loss_6 = dice_loss(pred[:, 5, :, :], target[:, 5, :, :])
    dice_loss_7 = dice_loss(pred[:, 6, :, :], target[:, 6, :, :])
    dice_loss_8 = dice_loss(pred[:, 7, :, :], target[:, 7, :, :])
    dice_loss_9 = dice_loss(pred[:, 8, :, :], target[:, 8, :, :])
    dice_loss_10 = dice_loss(pred[:, 9, :, :], target[:, 9, :, :])
    dice_loss_bg = dice_loss(pred[:, 10, :, :], target[:, 10, :, :])
    loss = dice_loss_1 + dice_loss_2 + dice_loss_3 + dice_loss_4 + dice_loss_5 + dice_loss_6 + dice_loss_7 + dice_loss_8 + dice_loss_9 + dice_loss_10 + dice_loss_bg

    return loss

def ini_model(cfg):
    restore=cfg.restore
    restore_from=cfg.restore_from
    # two models with different init
    # we need use SyncBatchNorm for DDP
    SyncBN = nn.SyncBatchNorm
    BN2d = nn.BatchNorm2d
    if restore:
        model_l = my_net('normalUnet', in_channels=1, num_classes=11,
                         norm_layer=SyncBN, pretrain_file=cfg.pretrain_file, 
                         pretrain=cfg.pretrain, bn_eps=cfg.bn_eps, 
                         bn_momentum=cfg.bn_momentum,
                         use_ms=True)

        model_r = my_net('normalUnet', in_channels=1, num_classes=11,
                         norm_layer=SyncBN, pretrain_file=cfg.pretrain_file, 
                         pretrain=cfg.pretrain, bn_eps=cfg.bn_eps, 
                         bn_momentum=cfg.bn_momentum,
                         use_ms=True)

        model_path_l = '/home/hyaoad/remote/semi_medical/journal_code/journal_semi_dg/model/' + 'l_' + str(restore_from) + '.pt'
        model_path_r = '/home/hyaoad/remote/semi_medical/journal_code/journal_semi_dg/model/' + 'r_' + str(restore_from) + '.pt'
        model_l.load_state_dict(torch.load(model_path_l))
        model_r.load_state_dict(torch.load(model_path_r))
        accelerator.print("restore from ", model_path_l)
        accelerator.print("restore from ", model_path_r)
    else:
        model_l = my_net('normalUnet', in_channels=1, num_classes=11,
                         norm_layer=SyncBN, pretrain_file=cfg.pretrain_file, 
                         pretrain=cfg.pretrain, bn_eps=cfg.bn_eps, 
                         bn_momentum=cfg.bn_momentum,
                         use_ms=True)

        model_r = my_net('normalUnet', in_channels=1, num_classes=11,
                         norm_layer=SyncBN, pretrain_file=cfg.pretrain_file, 
                         pretrain=cfg.pretrain, bn_eps=cfg.bn_eps, 
                         bn_momentum=cfg.bn_momentum,
                         use_ms=True)

    model_l = model_l.to(device)
    model_l.device = device

    model_r = model_r.to(device)
    model_r.device = device

    return model_l, model_r

def ini_optimizer(model_l, model_r, cfg):
    learning_rate = cfg.learning_rate 
    weight_decay = cfg.weight_decay
    # Initialize two optimizer.
    optimizer_l = torch.optim.AdamW(
        model_l.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_r = torch.optim.AdamW(
        model_r.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return optimizer_l, optimizer_r

def cal_variance(pred, aug_pred):
    kl_distance = nn.KLDivLoss(reduction='none')
    sm = torch.nn.Softmax(dim=1)
    log_sm = torch.nn.LogSoftmax(dim=1)
    variance = torch.sum(kl_distance(
        log_sm(pred), sm(aug_pred)), dim=1)
    exp_variance = torch.exp(-variance)

    return variance, exp_variance

def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(cfg, epoch):
    if cfg.cps_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if cfg.consistency_rampup is None:
            cfg.consistency_rampup = cfg.num_epoch
        return cfg.cps_weight * sigmoid_rampup(epoch, cfg.consistency_rampup)
    else:
        return cfg.cps_weight

def train_one_epoch(model_l, model_r, niters_per_epoch, label_dataloader, unlabel_dataloader_0, unlabel_dataloader_1, optimizer_r, optimizer_l, cross_criterion, epoch, cfg):
    # loss data
    total_loss = []
    total_loss_l = []
    total_loss_r = []
    total_cps_loss = []
    total_con_loss = []
    # tqdm
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(niters_per_epoch),
                file=sys.stdout, bar_format=bar_format, disable=not accelerator.is_local_main_process)

    for idx in pbar:
        minibatch = label_dataloader.__next__()
        unsup_minibatch_0 = unlabel_dataloader_0.__next__()
        unsup_minibatch_1 = unlabel_dataloader_1.__next__()

        imgs = minibatch['img']
        # aug_imgs = minibatch['aug_img']
        mask = minibatch['mask']

        unsup_imgs_0 = unsup_minibatch_0['img']
        unsup_imgs_1 = unsup_minibatch_1['img']

        aug_unsup_imgs_0 = unsup_minibatch_0['aug_img']
        aug_unsup_imgs_1 = unsup_minibatch_1['aug_img']
        mask_params = unsup_minibatch_0['mask_params']

        imgs = imgs.to(device)
        # aug_imgs = aug_imgs.to(device)
        mask_type = torch.long
        mask = mask.to(device=device, dtype=mask_type)

        unsup_imgs_0 = unsup_imgs_0.to(device)
        unsup_imgs_1 = unsup_imgs_1.to(device)
        aug_unsup_imgs_0 = aug_unsup_imgs_0.to(device)
        aug_unsup_imgs_1 = aug_unsup_imgs_1.to(device)
        mask_params = mask_params.to(device)

        batch_mix_masks = mask_params
        # unlabeled mixed images
        unsup_imgs_mixed = unsup_imgs_0 * \
            (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks
        # unlabeled r mixed images
        aug_unsup_imgs_mixed = aug_unsup_imgs_0 * \
            (1 - batch_mix_masks) + aug_unsup_imgs_1 * batch_mix_masks

        # activate mixstyle
        model_l.apply(activate_mixstyle)
        model_r.apply(activate_mixstyle)
        # add uncertainty
        with torch.no_grad():
            logits_u0_tea_1 = model_l(unsup_imgs_0)
            logits_u1_tea_1 = model_l(unsup_imgs_1)
            logits_u0_tea_1 = logits_u0_tea_1.detach()
            logits_u1_tea_1 = logits_u1_tea_1.detach()

            logits_u0_tea_2 = model_r(unsup_imgs_0)
            logits_u1_tea_2 = model_r(unsup_imgs_1)
            logits_u0_tea_2 = logits_u0_tea_2.detach()
            logits_u1_tea_2 = logits_u1_tea_2.detach()

            aug_logits_u0_tea_1 = model_l(aug_unsup_imgs_0)
            aug_logits_u1_tea_1 = model_l(aug_unsup_imgs_1)
            aug_logits_u0_tea_1 = aug_logits_u0_tea_1.detach()
            aug_logits_u1_tea_1 = aug_logits_u1_tea_1.detach()

            aug_logits_u0_tea_2 = model_r(aug_unsup_imgs_0)
            aug_logits_u1_tea_2 = model_r(aug_unsup_imgs_1)
            aug_logits_u0_tea_2 = aug_logits_u0_tea_2.detach()
            aug_logits_u1_tea_2 = aug_logits_u1_tea_2.detach()

        # add softmax
        logits_u0_tea_1 = torch.softmax(logits_u0_tea_1, dim=1)
        aug_logits_u0_tea_1 = torch.softmax(aug_logits_u0_tea_1, dim=1)
        logits_u1_tea_1 = torch.softmax(logits_u1_tea_1, dim=1)
        aug_logits_u1_tea_1 = torch.softmax(aug_logits_u1_tea_1, dim=1)
        logits_u0_tea_2 = torch.softmax(logits_u0_tea_2, dim=1)
        aug_logits_u0_tea_2 = torch.softmax(aug_logits_u0_tea_2, dim=1)
        logits_u1_tea_2 = torch.softmax(logits_u1_tea_2, dim=1)
        aug_logits_u1_tea_2 = torch.softmax(aug_logits_u1_tea_2, dim=1)

        # logits_cons_tea_1 = logits_u0_tea_1 * \
        #     (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
        # _, max_1 = torch.max(logits_cons_tea_1, dim=1)
        # max_1 = max_1.long()

        # aug_logits_cons_tea_1 = aug_logits_u0_tea_1 * \
        #     (1 - batch_mix_masks) + aug_logits_u1_tea_1 * batch_mix_masks
        # _, aug_max_1 = torch.max(aug_logits_cons_tea_1, dim=1)
        # aug_max_1 = aug_max_1.long()

        # logits_cons_tea_2 = logits_u0_tea_2 * \
        #     (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
        # _, max_2 = torch.max(logits_cons_tea_2, dim=1)
        # max_2 = max_2.long()

        # aug_logits_cons_tea_2 = aug_logits_u0_tea_2 * \
        #     (1 - batch_mix_masks) + aug_logits_u1_tea_2 * batch_mix_masks
        # _, aug_max_2 = torch.max(aug_logits_cons_tea_2, dim=1)
        # aug_max_2 = aug_max_2.long()

        # # pseudo label 1
        # prob_un_all_1 = torch.stack([logits_cons_tea_1, aug_logits_cons_tea_1], dim=2)
        # max_un_all_1 = torch.stack([max_1, aug_max_1], dim=1)
        # max_conf_un_each_branch_1, _ = torch.max(prob_un_all_1, dim=1)  # bs, n_branch - 1, h, w, d
        # _, branch_id_un_max_conf_1 = torch.max(max_conf_un_each_branch_1, dim=1,
        #                                                         keepdim=True)  # bs, h, w, d
        # ps_label_1 = torch.gather(max_un_all_1, dim=1, index=branch_id_un_max_conf_1)[:, 0]

        # # pseudo label 2
        # prob_un_all_2 = torch.stack([logits_cons_tea_2, aug_logits_cons_tea_2], dim=2)
        # max_un_all_2 = torch.stack([max_2, aug_max_2], dim=1)
        # max_conf_un_each_branch_2, _ = torch.max(prob_un_all_2, dim=1)  # bs, n_branch - 1, h, w, d
        # _, branch_id_un_max_conf_2 = torch.max(max_conf_un_each_branch_2, dim=1,
        #                                                         keepdim=True)  # bs, h, w, d
        # ps_label_2 = torch.gather(max_un_all_2, dim=1, index=branch_id_un_max_conf_2)[:, 0]

        logits_u0_tea_1 = (logits_u0_tea_1 + aug_logits_u0_tea_1) / 2
        logits_u1_tea_1 = (logits_u1_tea_1 + aug_logits_u1_tea_1) / 2
        logits_u0_tea_2 = (logits_u0_tea_2 + aug_logits_u0_tea_2) / 2
        logits_u1_tea_2 = (logits_u1_tea_2 + aug_logits_u1_tea_2) / 2

        # Mix teacher predictions using same mask
        # It makes no difference whether we do this with logits or probabilities as
        # the mask pixels are either 1 or 0
        logits_cons_tea_1 = logits_u0_tea_1 * \
            (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
        _, ps_label_1 = torch.max(logits_cons_tea_1, dim=1)
        ps_label_1 = ps_label_1.long()

        logits_cons_tea_2 = logits_u0_tea_2 * \
            (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
        _, ps_label_2 = torch.max(logits_cons_tea_2, dim=1)
        ps_label_2 = ps_label_2.long()

        # deactivate mixstyle
        model_l.apply(deactivate_mixstyle)
        model_r.apply(deactivate_mixstyle)
        # Get student_l prediction for mixed image
        logits_cons_stu_1 = model_l(unsup_imgs_mixed)
        aug_logits_cons_stu_1 = model_l(aug_unsup_imgs_mixed)
        var_l, exp_var_l = cal_variance(logits_cons_stu_1, aug_logits_cons_stu_1)

        # Get student_r prediction for mixed image
        logits_cons_stu_2 = model_r(unsup_imgs_mixed)
        aug_logits_cons_stu_2 = model_r(aug_unsup_imgs_mixed)
        var_r, exp_var_r = cal_variance(logits_cons_stu_2, aug_logits_cons_stu_2)

        # ensemble
        logits_cons_stu_1 = (logits_cons_stu_1 + aug_logits_cons_stu_1) / 2
        logits_cons_stu_2 = (logits_cons_stu_2 + aug_logits_cons_stu_2) / 2

        # cps loss
        cps_loss = torch.mean(var_l) + torch.mean(var_r) + torch.mean(exp_var_r * cross_criterion(logits_cons_stu_1, ps_label_2)) + torch.mean(
                        exp_var_l * cross_criterion(logits_cons_stu_2, ps_label_1)) 

        # cps weight
        cps_weight = get_current_consistency_weight(cfg, epoch)
        cps_loss = cps_loss * cps_weight

        # activate mixstyle
        model_l.apply(activate_mixstyle)
        model_r.apply(activate_mixstyle)
        # supervised loss on both models
        pre_sup_l = model_l(imgs)
        pre_sup_r = model_r(imgs)

        SFA_loss = 0

        # dice loss
        sof_l = F.softmax(pre_sup_l, dim=1)
        sof_r = F.softmax(pre_sup_r, dim=1)

        loss_r = total_dice_loss(sof_r, mask)
        loss_l = total_dice_loss(sof_l, mask)

        optimizer_l.zero_grad()
        optimizer_r.zero_grad()

        loss = loss_l + loss_r + cps_loss

        accelerator.backward(loss)
        optimizer_l.step()
        optimizer_r.step()

        total_loss.append(loss.item())
        total_loss_l.append(loss_l.item())
        total_loss_r.append(loss_r.item())
        total_cps_loss.append(cps_loss.item())
        total_con_loss.append(SFA_loss)

    total_loss = sum(total_loss) / len(total_loss)
    total_loss_l = sum(total_loss_l) / len(total_loss_l)
    total_loss_r = sum(total_loss_r) / len(total_loss_r)
    total_cps_loss = sum(total_cps_loss) / len(total_cps_loss)
    total_con_loss = sum(total_con_loss) / len(total_con_loss)

    return model_l, model_r, total_loss, total_loss_l, total_loss_r, total_cps_loss, total_con_loss

# use the function to calculate the valid loss or test loss
def test_dual(model_l, model_r, loader):
    model_l.eval()
    model_r.eval()

    loss = []
    t_loss = 0
    r_loss = 0
    dice_loss_1 = 0
    dice_loss_2 = 0
    dice_loss_3 = 0
    dice_loss_4 = 0
    dice_loss_5 = 0
    dice_loss_6 = 0
    dice_loss_7 = 0
    dice_loss_8 = 0
    dice_loss_9 = 0
    dice_loss_10 = 0
    dice_loss_bg = 0

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

    for batch in tqdm(loader, disable=not accelerator.is_local_main_process):
        imgs = batch['img']
        mask = batch['mask']
        imgs = imgs.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            logits_l = model_l(imgs)
            logits_r = model_r(imgs)
         
        logits_l = accelerator.gather(logits_l)
        logits_r = accelerator.gather(logits_r)
        mask = accelerator.gather(mask)

        sof_l = F.softmax(logits_l, dim=1)
        sof_r = F.softmax(logits_r, dim=1)

        pred = (sof_l + sof_r) / 2
        pred = (pred > 0.5).float()

        # loss
        dice_loss_1 = dice_loss(pred[:, 0, :, :], mask[:, 0, :, :])
        dice_loss_2 = dice_loss(pred[:, 1, :, :], mask[:, 1, :, :])
        dice_loss_3 = dice_loss(pred[:, 2, :, :], mask[:, 2, :, :])
        dice_loss_4 = dice_loss(pred[:, 3, :, :], mask[:, 3, :, :])
        dice_loss_5 = dice_loss(pred[:, 4, :, :], mask[:, 4, :, :])
        dice_loss_6 = dice_loss(pred[:, 5, :, :], mask[:, 5, :, :])
        dice_loss_7 = dice_loss(pred[:, 6, :, :], mask[:, 6, :, :])
        dice_loss_8 = dice_loss(pred[:, 7, :, :], mask[:, 7, :, :])
        dice_loss_9 = dice_loss(pred[:, 8, :, :], mask[:, 8, :, :])
        dice_loss_10 = dice_loss(pred[:, 9, :, :], mask[:, 9, :, :])
        dice_loss_bg = dice_loss(pred[:, 10, :, :], mask[:, 10, :, :])
        t_loss = dice_loss_1 + dice_loss_2 + dice_loss_3 + dice_loss_4 + dice_loss_5 + dice_loss_6 + dice_loss_7 + dice_loss_8 + dice_loss_9 + dice_loss_10 + dice_loss_bg
        loss.append(t_loss.item())

        # dice score
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

    r_loss = sum(loss) / len(loss)

    dice_1 = tot_1/len(loader)
    dice_2 = tot_2/len(loader)
    dice_3 = tot_3/len(loader)
    dice_4 = tot_4/len(loader)
    dice_5 = tot_5/len(loader)
    dice_6 = tot_6/len(loader)
    dice_7 = tot_7/len(loader)
    dice_8 = tot_8/len(loader)
    dice_9 = tot_9/len(loader)
    dice_10 = tot_10/len(loader)
    dice = (dice_1 + dice_2 + dice_3 + dice_4 + dice_5 + dice_6 + dice_7 + dice_8 + dice_9 + dice_10) / 10

    model_l.train()
    model_r.train()
    dice_all = {
        'dice_1': dice_1,
        'dice_2': dice_2,
        'dice_3': dice_3,
        'dice_4': dice_4,
        'dice_5': dice_5,
        'dice_6': dice_6,
        'dice_7': dice_7,
        'dice_8': dice_8,
        'dice_9': dice_9,
        'dice_10': dice_10,
    }
    return r_loss, dice, dice_all

def train(label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, niters_per_epoch, cfg):

    # Initialize model
    model_l, model_r = ini_model(cfg)

    # loss
    cross_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

    # Initialize optimizer.
    optimizer_l, optimizer_r = ini_optimizer(
        model_l, model_r, cfg)

    # Initializa scheduler
    scheduler_l = fetch_scheduler(cfg, optimizer_l)
    scheduler_r = fetch_scheduler(cfg, optimizer_r)

    # using acclerator
    model_l, model_r, optimizer_l, optimizer_r, label_loader, unlabel_loader_0, unlabel_loader_1, scheduler_l, scheduler_r = accelerator.prepare(model_l, model_r, optimizer_l, optimizer_r, label_loader, unlabel_loader_0, unlabel_loader_1, scheduler_l, scheduler_r)

    val_loader, test_loader = accelerator.prepare(val_loader, test_loader)
    best_dice = 0
    best_test_dice = 0
    num_epoch = cfg.num_epoch
    for epoch in range(num_epoch):
        # ---------- Training ----------
        model_l.train()
        model_r.train()

        label_dataloader = iter(label_loader)
        unlabel_dataloader_0 = iter(unlabel_loader_0)
        unlabel_dataloader_1 = iter(unlabel_loader_1)

        # normal images
        model_l, model_r, total_loss, total_loss_l, total_loss_r, total_cps_loss, total_con_loss = train_one_epoch(
            model_l, model_r, niters_per_epoch, label_dataloader, unlabel_dataloader_0, unlabel_dataloader_1, optimizer_r, optimizer_l, cross_criterion, epoch, cfg)
        
        # learning rate decay
        scheduler_l.step()
        scheduler_r.step()
        realLearningRate = scheduler_r.get_last_lr()[0]
        # Print the information.
        accelerator.print(
            f"[ Normal image Train | {epoch + 1:03d}/{num_epoch:03d} ] total_loss = {total_loss:.5f} total_loss_l = {total_loss_l:.5f} total_loss_r = {total_loss_r:.5f} total_cps_loss = {total_cps_loss:.5f}")

        # ---------- Validation ----------
        val_loss, val_dice, val_dice_all = test_dual(
            model_l, model_r, val_loader)
        accelerator.print(
            f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] val_loss = {val_loss:.5f} val_dice = {val_dice:.5f}")

        # ---------- Testing----------
        test_loss, test_dice, test_dice_all = test_dual(
            model_l, model_r, test_loader)
        accelerator.print(
            f"[ Test | {epoch + 1:03d}/{num_epoch:03d} ] test_loss = {test_loss:.5f} test_dice = {test_dice:.5f}")

        # if the model improves, save a checkpoint at this epoch
        if test_dice > best_dice:
            best_dice = test_dice
            best_test_dice = test_dice
            model_name_l = 'l_' + cfg.model_name + '.pt'
            model_name_r = 'r_' + cfg.model_name + '.pt'
            model_path_l = os.path.join(os.getcwd(), model_name_l)
            model_path_r = os.path.join(os.getcwd(), model_name_r)

            accelerator.wait_for_everyone()
            unwrapped_model_l = accelerator.unwrap_model(model_l)
            unwrapped_model_r = accelerator.unwrap_model(model_r)

            accelerator.save(unwrapped_model_l.state_dict(), model_path_l)
            accelerator.save(unwrapped_model_r.state_dict(), model_path_r)
            accelerator.print('saving model with best_dice {:.5f}'.format(best_dice))

        if accelerator.is_local_main_process:
            # val
            wandb.log(step=epoch + 1,
            data={'val/val_dice': val_dice, 'val/val_dice_1': val_dice_all['dice_1'],
            'val/val_dice_2': val_dice_all['dice_2'], 'val/val_dice_3': val_dice_all['dice_3'],
            'val/val_dice_4': val_dice_all['dice_4'], 'val/val_dice_5': val_dice_all['dice_5'],
            'val/val_dice_6': val_dice_all['dice_6'], 'val/val_dice_7': val_dice_all['dice_7'],
            'val/val_dice_8': val_dice_all['dice_8'], 'val/val_dice_9': val_dice_all['dice_9'],
            'val/val_dice_10': val_dice_all['dice_10']})
            # test
            wandb.log(step=epoch + 1,
            data={'test/test_dice': test_dice, 'test/best_dice': best_dice,
            'test/test_dice_1': test_dice_all['dice_1'], 'test/test_dice_2': test_dice_all['dice_2'],
            'test/test_dice_3': test_dice_all['dice_3'], 'test/test_dice_4': test_dice_all['dice_4'],
            'test/test_dice_5': test_dice_all['dice_5'], 'test/test_dice_6': test_dice_all['dice_6'],
            'test/test_dice_7': test_dice_all['dice_7'], 'test/test_dice_8': test_dice_all['dice_8'],
            'test/test_dice_9': test_dice_all['dice_9'], 'test/test_dice_10': test_dice_all['dice_10']})
            # loss
            wandb.log(step=epoch + 1,
            data={'epoch': epoch + 1, 'lr': realLearningRate, 'loss/total_loss': total_loss, 'loss/total_loss_l': total_loss_l,
                  'loss/total_loss_r': total_loss_r, 'loss/total_cps_loss': total_cps_loss,
                   'loss/test_loss': test_loss, 'loss/val_loss': val_loss, 'loss/SFA_loss': total_con_loss })

    accelerator.print('The test dice of {0} is {1}'.format(cfg.test_vendor, best_test_dice))

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

@hydra.main(version_base='1.1', config_path='configs', config_name='chd')
def main(cfg):
    accelerator.print(OmegaConf.to_yaml(cfg))
    # set wandb
    if accelerator.is_local_main_process:
        wandb_init(cfg)

    accelerator.print("loading data")
    label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, num_label_imgs, num_unsup_imgs = pre_data(cfg)
    accelerator.print("training")
    max_samples = num_unsup_imgs
    niters_per_epoch = int(math.ceil(max_samples * 1.0 // cfg.batch_size))
    accelerator.print("max_samples", max_samples)
    accelerator.print("niters_per_epoch", niters_per_epoch)

    if cfg.fourier_augment:
        accelerator.print("Fourier mode")
    else:
        accelerator.print("Normal mode")

    accelerator.print(device)
    train(label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, niters_per_epoch, cfg)

if __name__ == '__main__':
    main()
