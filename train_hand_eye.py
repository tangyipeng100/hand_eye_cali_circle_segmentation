#!/usr/bin/env Python
# coding=utf-8

from Naive_cross_line_net import _512_pointnet
from Naive_cross_line_net import _256_pointnet
from Naive_cross_line_net import _128_pointnet
from Naive_cross_line_net import _64_pointnet
from Naive_cross_line_net import *
import torch
from torch.utils.data import DataLoader
from training_infer_tools.my_lr_schedule import CosineWarmupLr
import argparse
import logging
import os
import copy
import datetime
import time
from pathlib import Path
import numpy as np
import importlib
import distutils.util

from visdom import Visdom
from tqdm import tqdm
from utils.Handeye_datasets import Handeye_datasets
import utils.config as con

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Pointnet64', help='Pointnet512, Pointnet256, Pointnet128, Pointnet64')
    parser.add_argument('--dim', type=int, default=2, help='Input points data dimension, 2')
    parser.add_argument('--num_classes', type=int, default=2, help='Segmentation classes:2')
    parser.add_argument('--normal_method', type=str, default='Whole', help='Loading profiles normalization method, Whole or Min_max')
    parser.add_argument("--model_description", type=str, default='Default_model', help='model description path')
    parser.add_argument("--visdom_env", type=str, default='circle_segmentation', help='visdom environment name')
    parser.add_argument("--epochs", type=int, default=250, help="number of epochs")
    parser.add_argument("--train_path", type=str, default='./data/Standard_sphere_seg_dataset_v1/train_file.txt', help='train file path')
    parser.add_argument("--val_path", type=str, default='./data/Standard_sphere_seg_dataset_v1/val_file.txt', help='validate file path')
    parser.add_argument("--batch_size", type=int, default=36, help='size of each profile batch')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or Adadelta, SGD, RMSprop, Rprop, ASGD, Adamax, SparseAdam, AdamW, LBFGS [default: Adam]')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help='learning rate setting')
    parser.add_argument('--warm_up', type=lambda x:bool(distutils.util.strtobool(x)), default='False', help='warm_up strategy for training')
    parser.add_argument('--decay_rate', type=float, default=0, help='weight decay [default: 1e-4]')
    parser.add_argument('--decay_rate_change', type=lambda x:bool(distutils.util.strtobool(x)), default='False', help='weight decay increase with steps')
    parser.add_argument("--SGD_momentum", type=float, default=0.9, help='SGD_momentum')
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")  # 在CPU线程数足够的情况下，可以设置num_workers=四分之一到半线程数，CPU的利用率最高。
    opt = parser.parse_args()
    return opt


def main(opt):
    if opt.num_classes == 2:
        seg_classes = {0: 'Outlier', 1: 'Circle points'}
    else:
        raise Exception('Error num_classes.')

    def log_string(str):
        logger.info(str)
        print(str)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)  # 默认初始化为kaiming均匀分布，之后可以尝试
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%H-%M-%S'))
    daystr = str(datetime.datetime.now().strftime('%Y_%m_%d'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_dir.joinpath(daystr)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    Time_modelStr = timestr + '_' + opt.model
    experiment_dir = experiment_dir.joinpath(Time_modelStr)
    experiment_dir.mkdir(parents=True, exist_ok=True)


    '''LOG'''
    opt = parse_args()
    logger = logging.getLogger(opt.model)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (experiment_dir, opt.model_description+'_log_file'))#只限制目录和日志运行的模型
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    log_string('PARAMETERS ...')
    log_string(opt)

    train_dataset = Handeye_datasets(opt.train_path, con.dataset_pre, normal_method=opt.normal_method, num_classes=opt.num_classes, dim=opt.dim)
    train_weight = torch.Tensor(train_dataset.labelweights).cuda()
    log_string('Category weight:{}'.format(train_weight))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=False,
        #drop_last=True
    )
    val_dataset =  Handeye_datasets(opt.val_path, con.dataset_pre, normal_method=opt.normal_method, num_classes=opt.num_classes, dim=opt.dim)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=False,
    )

    '''MODEL LOADING    '''
    if opt.model == 'Pointnet512':
        model = _512_pointnet(opt.num_classes, opt.dim)
    elif opt.model == 'Pointnet256':
        model = _256_pointnet(opt.num_classes, opt.dim)
    elif opt.model == 'Pointnet128':
        model = _128_pointnet(opt.num_classes, opt.dim)
    elif opt.model == 'Pointnet64':
        model = _64_pointnet(opt.num_classes, opt.dim)
    else:
        raise Exception('No such model.')

    model = model.apply(weights_init)

    if torch.cuda.is_available():
        model = model.cuda()

    log_string('Net name:%s' % opt.model)
    log_string('Optimizer name:%s \n' % opt.optimizer)

    log_string(model)

    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=opt.decay_rate
        )
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.SGD_momentum, weight_decay=opt.decay_rate)
    elif opt.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=opt.learning_rate, rho=0.9, eps=1e-06, weight_decay=opt.decay_rate)
    elif opt.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.learning_rate, alpha=0.99, eps=1e-08, weight_decay=opt.decay_rate, momentum=opt.SGD_momentum, centered=False)
    elif opt.optimizer == 'Rprop':
        optimizer = torch.optim.Rprop(model.parameters(), lr=opt.learning_rate, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    elif opt.optimizer == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr=opt.learning_rate, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=opt.decay_rate)
    elif opt.optimizer == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=opt.decay_rate)
    elif opt.optimizer == 'SparseAdam':
        #print('para model', list(model.parameters()))
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    elif opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=opt.decay_rate, amsgrad=False)
    elif opt.optimizer == 'LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=opt.learning_rate, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)

    if opt.warm_up:
        scheduler = CosineWarmupLr(optimizer, batches=38, max_epochs=opt.batch_size, base_lr=opt.learning_rate,
                               final_lr=1e-6, warmup_epochs=10, warmup_init_lr=0.0)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    log_string(optimizer)
    log_string(scheduler)

    #loss
    criterion = nn.CrossEntropyLoss(weight=train_weight)
    criterion = criterion.cuda()

    forward_timeL = []
    best_trainLoss = 1000
    best_evalacc = 0
    best_mIOU = 0
    best_shape_ious = []

    viz = Visdom(env=opt.visdom_env)
    viz.line([[0., 0.0]], [0.], win='train & val loss', opts=dict(title='train & val loss', legend=['train', 'val']))
    viz.line([[0.0, 0.0, 0.0]], [0.], win='accuracy_mIOU', opts=dict(title='train_validate acc & mIOU',
                                                            legend=['train', 'val', 'mIOU']))
    train_starttime = datetime.datetime.now()
    for epoch in tqdm(range(opt.epochs), ncols=80):
        train_loss_sum = 0
        train_total_correct = 0
        train_total_points = 0
        train_num_batches = len(train_dataloader)

        for batch_i, (profile_im, profile_label, _, _) in enumerate(train_dataloader):
            profile_im = profile_im.float().permute(0, 2, 1)
            profile_label = profile_label.long().squeeze(2)

            if torch.cuda.is_available():
                profile_im = torch.autograd.Variable(profile_im).cuda()
                profile_label = torch.autograd.Variable(profile_label).cuda()

            model = model.train()
            out = model(profile_im)
            loss = criterion(out, profile_label)  # out:[b, 4, 32000], profile_label:[n, N]

            _, pred = torch.max(out, 1)
            num_correct = np.sum(pred.long().cpu().numpy() == profile_label.long().cpu().numpy())
            train_total_correct += num_correct
            train_total_points += pred.shape[0]*pred.shape[1]

            assert torch.isnan(loss).sum() == 0, print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if opt.warm_up:
                scheduler.step()
            train_loss_sum += loss.detach().cpu().item()

        train_acc = train_total_correct / float(train_total_points)
        log_string('train_acc:%.5f' % train_acc)
        train_loss = train_loss_sum / float(train_num_batches)
        log_string('train_loss:%.5f' % train_loss)
        if train_loss < best_trainLoss:
            best_trainLoss = train_loss
            log_string('Best train loss:%.5f' % best_trainLoss)

        if not opt.warm_up:
            scheduler.step()
        #scheduler.step(train_loss)
        for param_group in optimizer.param_groups:  # 调整optimizer中lr的方法
            log_string('Current lr:%.5f' % param_group['lr'])
            if opt.decay_rate_change and epoch >= 100 and epoch % 50 == 0:
                param_group['weight_decay'] = param_group['weight_decay'] + 0.0005
            log_string('Current weight_decay:%.5f' % param_group['weight_decay'])

        with torch.no_grad():
            val_total_correct = 0
            val_total_points = 0
            total_seen_class = [0 for _ in range(len(seg_classes))]
            total_correct_class = [0 for _ in range(len(seg_classes))]

            if opt.num_classes == 2:
                shape_ious = {'Outlier': [], 'Circle points':[]}
            else:
                raise Exception('Error num_classes.')

            shape_ious_value = []
            val_loss_sum = 0
            forward_start_time = time.time()
            val_num_batches = len(val_dataloader)
            model = model.eval()
            for batch_i, (profile_imv, profile_labelv, _, _) in enumerate(val_dataloader):
                profile_imv = profile_imv.float().permute(0, 2, 1)
                profile_labelv = profile_labelv.long().squeeze(2)

                if torch.cuda.is_available():
                    profile_imv = torch.autograd.Variable(profile_imv).cuda()
                    profile_labelv = torch.autograd.Variable(profile_labelv).cuda()

                out_v = model(profile_imv)
                loss_v = criterion(out_v, profile_labelv)

                val_loss_sum += loss_v.detach().cpu().numpy()
                _, pred_v = torch.max(out_v, 1)
                pred_np = pred_v.long().cpu().numpy()
                label_np = profile_labelv.long().cpu().numpy()
                val_total_correct += np.sum(pred_np == label_np)
                val_total_points += pred_v.shape[0]*pred_v.shape[1]

                for j in range(len(seg_classes)):
                    if np.sum((pred_np == j) | (label_np == j)) != 0:
                        total_correct_class[j] += np.sum((pred_np == j) & (label_np == j))
                        total_seen_class[j] += np.sum((pred_np == j) | (label_np == j))

            val_loss = val_loss_sum / val_num_batches
            log_string('val_loss:%.5f' % val_loss)
            viz.line([[train_loss, val_loss]], [epoch], win='train & val loss', update='append')

            for j in range(len(seg_classes)):
                if total_seen_class[j] != 0:
                    iou = total_correct_class[j] / float(total_seen_class[j])
                    shape_ious_value.append(iou)
                    shape_ious[seg_classes[j]] = iou
                else:
                    shape_ious[seg_classes[j]] = 'None'

            forward_time_cost_100 = (time.time() - forward_start_time)/opt.batch_size/50 * 100
            forward_timeL.append(forward_time_cost_100)

            #scheduler.step(val_loss)
            mIou = np.mean(shape_ious_value)
            log_string('Mean IOU: %.5f' % mIou)
            val_acc = val_total_correct / float(val_total_points)
            log_string('val_acc:%.5f' % val_acc)
            log_string('Forward_100_time:%.5f s' % forward_time_cost_100)
            viz.line([[train_acc, val_acc, mIou]], [epoch], win='accuracy_mIOU', update='append')

            if mIou > best_mIOU:
                best_mIOU = mIou
                best_evalacc = val_acc
                best_shape_ious = copy.deepcopy(shape_ious)
                savepath_model = str(experiment_dir) + '/bestmodel.pkl'
                savepath_dict = str(experiment_dir) + '/bestmodel_params.pkl'
                torch.save(model, savepath_model)
                torch.save(model.state_dict(), savepath_dict)
                log_string('Best mIOU:%.5f' % best_mIOU)
                log_string('Best accuracy:%.5f' % best_evalacc)
                log_string('Model saving success!')

    train_cost_time = ( (datetime.datetime.now()-train_starttime).seconds )/3600.0
    log_string('\nTraining total time:%.5f h' % train_cost_time)
    log_string('Mean 100 profile forward time:%.5f s' % np.mean(forward_timeL))
    log_string('Final best accuracy:%.5f' % best_evalacc)
    log_string('Final best loss:%.5f' % best_trainLoss)
    log_string('Final best mIOU:%.5f' % best_mIOU)
    for key, value in best_shape_ious.items():
        if type(value) != str:
            log_string('%s IOU:%.5f' % (key, value))
        else:
            log_string('%s IOU:%S' % (key, value))

if __name__ == '__main__':
    opt = parse_args()
    main(opt)