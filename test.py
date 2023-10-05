# -*- coding: utf-8 -*-

import time
import os
import argparse
from glob import glob
import warnings
import joblib
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
import torch
from torch.utils.data import DataLoader


from dataset import Dataset

import archs
from metrics import  iou_score, accuracy, F1_score_special
from Dropoutblock import  DropBlock_search

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--name', default='neu_seg_BayesUNet_spatial_woDS_230926',
                        help='model name')
    args = parser.parse_args()

    return args

def apply_dropout(m):
    if type(m) == DropBlock_search:
        m.train()

def main():
    val_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %val_args.name)

    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # create model
    print("=> creating model %s" %args.arch)
    model = archs.__dict__[args.arch](args)
    model = model.cuda()

    model_score = archs.Score(args)
    model_score = model_score.cuda()

    # Data loading code
    img_paths = glob('input/' + args.dataset + '/images/*')
    mask_paths = glob('input/' + args.dataset + '/masks/*')

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    for i in range(len(val_mask_paths)):
        val_mask_paths[i] = 'input/' + args.dataset + '/masks/' + val_img_paths[i].split('\\')[-1]

    model.load_state_dict(torch.load('models/%s/model.pth' %args.name)['model'])
    # model.load_state_dict(torch.load('models/%s/model.pth' % args.name))
    model_score.load_state_dict(torch.load('models/%s/model.pth' %args.name)['model_score'])
    # model.load_state_dict(torch.load('models/%s/model.pth' % args.name))
    model.eval()
    model.apply(apply_dropout)

    # train_dataset = Dataset(args, train_img_paths, train_mask_paths)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     # batch_size=args.batch_size,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=False)


    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        # batch_size=args.batch_size,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    starttime = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            # for ij, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
                input = input.cuda()
                target = target.cuda()
                n = 16

                output_bayes = np.zeros(target.size())
                output_bayes_int = np.zeros(target.size())
                inputs = input
                for k in range(3):
                    inputs = torch.cat((inputs,input))
                outputs, features = model(inputs)
                for kk in range(3):
                    output_, features = model(inputs)
                    outputs = torch.cat((outputs, output_))
                inputs_ = inputs
                for kkk in range(3):
                    inputs_ = torch.cat((inputs, inputs_))
                input_score = torch.cat((inputs_, outputs), 1)
                score = model_score(input_score)
                results = outputs[0,:,:,:] * ((score[0,:,:]).unsqueeze(0))
                result = torch.mean(results, dim=0).unsqueeze(0).unsqueeze(0)
                result = torch.sigmoid(result).data.cpu().numpy()

                for l in range(n):
                    output = outputs[l,:,:,:].unsqueeze(0)
                    output_mat = torch.sigmoid(output).data.cpu().numpy()
                    output_int = output_mat > 0.5
                    output_bayes += output_mat #* score_mat  # * (1 / n)
                    output_bayes_int += output_int

                output_bayes = output_bayes *(1/n)
                certain = output_bayes_int >= 16
                suspect = output_bayes_int >= 1
                uncertain = certain ^ suspect
                bayes_int = output_bayes_int / 16

                img_paths = val_img_paths[1 * i:1 * (i + 1)]

                for i in range(output_bayes.shape[0]):
                    save_output = (output_bayes[i, 0, :, :]* 255).astype('uint8')
                    output_heat = cv2.applyColorMap(save_output, cv2.COLORMAP_JET)
                    imsave('output/%s/' % args.name + os.path.basename(img_paths[i]),
                               (output_bayes[i, 0, :, :] * 255).astype('uint8'))
                    imsave('output/%s/' % args.name + os.path.basename(img_paths[i].split('.')[0]+'_certain'+'.'+img_paths[i].split('.')[1]),
                           (certain[i, 0, :, :] * 255).astype('uint8'))
                    imsave('output/%s/' % args.name + os.path.basename(img_paths[i].split('.')[0] + '_suspect' + '.' + img_paths[i].split('.')[1]),
                           (suspect[i, 0, :, :] * 255).astype('uint8'))
                    imsave('output/%s/' % args.name + os.path.basename(
                        img_paths[i].split('.')[0] + '_uncertain' + '.' + img_paths[i].split('.')[1]),
                           (uncertain[i, 0, :, :] * 255).astype('uint8'))
                    imsave('output/%s/' % args.name + os.path.basename(
                        img_paths[i].split('.')[0] + '_heatmap' + '.' + img_paths[i].split('.')[1]),
                           output_heat)
                    imsave('output/%s/' % args.name + os.path.basename(
                        img_paths[i].split('.')[0] + '_bayes_int' + '.' + img_paths[i].split('.')[1]),
                           (bayes_int[i, 0, :, :] * 255).astype('uint8'))
                    imsave('output/%s/' % args.name + os.path.basename(
                        img_paths[i].split('.')[0] + '_score' + '.' + img_paths[i].split('.')[1]),
                           (result[i, 0, :, :] * 255).astype('uint8'))




    torch.cuda.empty_cache()

    endtime = time.time()
    # IoU
    ious = []
    pas = []
    F1s = []
    for i in tqdm(range(len(val_mask_paths))):
        mask = imread(val_mask_paths[i])
        mask = cv2.resize(mask, (256, 256))
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        pb = imread('output/%s/'%args.name+os.path.basename(val_mask_paths[i]))
        certain = imread('output/%s/'%args.name+os.path.basename(val_mask_paths[i]).split('.')[0] + '_certain' + '.' + os.path.basename(val_mask_paths[i]).split('.')[-1])
        suspect = imread('output/%s/'%args.name+os.path.basename(val_mask_paths[i]).split('.')[0] + '_suspect' + '.' + os.path.basename(val_mask_paths[i]).split('.')[-1])

        mask = mask.astype('float32') / 255
        pb = pb.astype('float32') / 255
        certain = certain.astype('float32') / 255
        suspect = suspect.astype('float32') / 255

        iou = iou_score(pb, mask)
        ious.append(iou)

        pa = accuracy(pb,mask)
        pas.append(pa)

        F1 = F1_score_special(certain, suspect, mask)
        F1s.append(F1)

    print('Time: ',endtime - starttime)
    print('IoU: %.4f, PA: %.4f, F1: %.4f' % (np.mean(ious), np.mean(pas), np.mean(F1s)))

if __name__ == '__main__':
    main()
