# -*- coding: utf-8 -*-

import os
import argparse
from glob import glob
from collections import OrderedDict

import pandas as pd

import joblib

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from Dropoutblock import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset import Dataset

import archs
from metrics import iou_score, Recall_suspect, Precision_certain
import losses
from utils import str2bool, count_params

arch_names = list(archs.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='BayesUNet_spatial',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default=False,
                        type=str2bool)
    parser.add_argument('--dataset', default='neu_seg',
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--size', default=256, type=int,
                        help='image size')
    parser.add_argument('--image-ext', default='bmp',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='bmp',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss_splite',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=300
                        , type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,

                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, model_score, criterion, optimizer, optimizer_score, epoch, scheduler=None):
    losses = AverageMeter()
    score_losses = AverageMeter()
    ious = AverageMeter()

    model.train()
    model_score.train()
    torch.autograd.set_detect_anomaly(True)


    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()
        if args.deepsupervision:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            n = 4
            output = model(input)
            loss_ = criterion(output, target).unsqueeze(1)
            for i in range(n-1):
                output = model(input)
                loss_ = torch.cat((loss_, criterion(output, target).unsqueeze(1)), 1)

            loss_var = torch.var(loss_, dim=1)
            # print('loss_var:', loss_var)
            loss_mean = torch.mean(loss_, dim=1)
            input_score = torch.cat((input, target), 1)
            score = model_score(input_score)
            # print('score', score)
            loss = score * loss_mean
            loss = torch.sum(loss)
            # print('loss', loss)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)


            # loss_var_normal = (loss_var - torch.min(loss_var)) / (torch.max(loss_var) - torch.min(loss_var))
            loss_var_normal = torch.sigmoid(loss_var*10000)
            loss_var_softmax = F.softmax(-loss_var_normal)
            # print(loss_var_softmax)
            loss_score = F.binary_cross_entropy_with_logits(score, loss_var_softmax)
            # print('loss_score', loss_score)
            # loss_score_final = loss_score + loss
            loss_score_final = loss_score

            optimizer_score.zero_grad()


            loss_score_final.backward()
            optimizer.step()
            optimizer_score.step()

            iou = iou_score(output, target)

        losses.update(loss.item(), input.size(0))
        score_losses.update(loss_score_final.item(), input.size(0))
        ious.update(iou, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('score_loss', score_losses.avg),
    ])

    return log


def validate(args, val_loader, model,model_score, criterion):
    losses = AverageMeter()
    ious = AverageMeter()
    new_losses = AverageMeter()
    new_ious = AverageMeter()
    Ps = AverageMeter()
    Rs = AverageMeter()

    def apply_dropout(m):
        if type(m) == nn.Dropout or type(m) == nn.Dropout2d or type(m) == DropBlock_search:
            m.train()

    # switch to evaluate mode
    model.eval()
    model_score.eval()
    model.apply(apply_dropout)

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            n = 16
            loss = 0
            iou = 0

            output_bayes = np.zeros(target.size())
            output_bayes_int = np.zeros(target.size())
            inputs = input
            for k in range(3):
                inputs = torch.cat((inputs, input))
            outputs, features = model(inputs)
            for kk in range(3):
                output_, features = model(inputs)
                outputs = torch.cat((outputs, output_))
            for kkk in range(2):
                inputs = torch.cat((inputs, inputs))
            outputs_score = torch.cat((inputs, outputs), 1)
            scores = model_score(outputs_score)
            results = outputs[0, :, :, :] * (scores[0, :, :].unsqueeze(0))
            for ii in range(n - 1):
                results = torch.cat(
                    (results, outputs[ii + 1, :, :, :] * (scores[ii + 1, :, :].unsqueeze(0))), 0)
            result = torch.sum(results, dim=0)
            result = result.unsqueeze(0).unsqueeze(0)
            iou_new = iou_score(result, target)
            loss_new = criterion(result, target)


            for l in range(n):
                output = outputs[l,:,:,:].unsqueeze(0)
                loss_a = criterion(output, target)
                iou_a = iou_score(output, target)
                loss += loss_a
                iou += iou_a
            loss = loss/n
            iou = iou/n
            for l in range(n):
                output = outputs[l, :, :, :].unsqueeze(0)
                output_mat = torch.sigmoid(output).data.cpu().numpy()
                output_int = output_mat > 0.5
                output_bayes += output_mat  # * (1 / n)
                output_bayes_int += output_int

            certain = output_bayes_int >= 16
            P = Precision_certain(certain, target)
            suspect = output_bayes_int >= 1
            R = Recall_suspect(suspect, target)
            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            new_losses.update(loss_new.item(), input.size(0))
            new_ious.update(iou_new.item(), input.size(0))
            Ps.update(P, input.size(0))
            Rs.update(R, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('P', Ps.avg),
        ('R', Rs.avg),
        ('new_losses', new_losses.avg),
        ('new_ious', new_ious.avg),
    ])

    return log


def main():
    args = parse_args()

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.dataset, args.arch)
        else:
            args.name = '%s_%s_woDS_230926' %(args.dataset, args.arch)
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[args.loss]().cuda()

    cudnn.benchmark = True

    # Data loading code
    img_paths = glob('input/' + args.dataset + '/images/*')
    mask_paths = glob('input/' + args.dataset + '/masks/*')

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)

    # create model
    print("=> creating model %s" %args.arch)
    model = archs.__dict__[args.arch](args)

    model_score = archs.Score(args)

    model = model.cuda()
    model_score = model_score.cuda()

    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr':args.lr},
        {'params':filter(lambda p: p.requires_grad, model_score.parameters()), 'lr':args.lr}
                                ])
        optimizer_score = optim.Adam([
            # {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr':args.lr},
            {'params':filter(lambda p: p.requires_grad, model_score.parameters()), 'lr':1e-4}
                                ])
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    #print(train_dataset[0])
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    #print(val_dataset[0])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        # batch_size=args.batch_size,
        batch_size= 1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, model_score, criterion, optimizer, optimizer_score, epoch)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, model_score, criterion)

        print('loss %.4f - score_loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - best_val_iou %.4f - val_new_loss %.4f - val_new_iou %.4f'
            %(train_log['loss'],train_log['score_loss'], train_log['iou'], val_log['loss'], val_log['iou'], best_iou, val_log['new_losses'], val_log['new_ious']))

        tmp = pd.Series([
            epoch,
            train_log['loss'],
            train_log['score_loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
        ], index=['epoch', 'loss', 'score_loss', 'iou', 'val_loss', 'val_iou'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        trigger += 1
        torch.save({'model': model.state_dict(), 'model_score': model_score.state_dict()}, 'models/%s/model_latest.pth' %args.name)

        if val_log['new_ious'] > best_iou:
            torch.save({'model': model.state_dict(), 'model_score': model_score.state_dict()}, 'models/%s/model.pth' %args.name)
            best_iou = val_log['new_ious']
            print("=> saved best model")
            trigger = 0

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
