import numpy as np
import torch

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def Recal(output, target):
    smooth = 1e-5
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    output_ = output[:, 0, :, :] > 0.5
    for i in range(output.shape()[1]-1):
        _output_ = output[:, i+1, :, :] > 0.5
        output_ = output_ or _output_
    output_ = output_ > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = target_.sum()

    return (intersection + smooth) / (union + smooth)

def Precision(output, target):
    smooth = 1e-5
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    output_ = output[:, 0, :, :] > 0.5
    for i in range(output.shape()[1]-1):
        _output_ = output[:, i+1, :, :] > 0.5
        output_ = output_ * _output_
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = output_.sum()

    return (intersection + smooth) / (union + smooth)


def Recall_suspect(output, target):
    smooth = 1e-5
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    target_ = target > 0.5
    output_ = output > 0.5

    intersection = (output_ & target_).sum()
    union = target_.sum()

    return (intersection + smooth) / (union + smooth)

def Precision_certain(output, target):
    smooth = 1e-5
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    target_ = target > 0.5
    output_ = output > 0.5

    intersection = (output_ & target_).sum()
    union = output_.sum()

    return (intersection + smooth) / (union + smooth)


def accuracy(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    else:
        output = torch.from_numpy(output).view(-1).numpy()
    if torch.is_tensor(target):
        target = target.view(-1).data.cpu().numpy()
    else:
        target = torch.from_numpy(target).view(-1).numpy()
    output = (np.round(output)).astype('int')
    target = (np.round(target)).astype('int')

    return (output == target).sum() / len(output)

def F1_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    output_True = output > 0.5
    target_True = target > 0.5
    intersection = (output_True & target_True).sum()
    A = output_True.sum()
    B = target_True.sum()

    Precision = (intersection + smooth) / (A+ smooth)
    Recall = (intersection+ smooth) / (B+ smooth)

    return 2*(Precision*Recall+smooth)/(Precision+Recall+smooth)

def F1_score_special(certain, suspect, target):
    smooth = 1e-5
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    certain_True = certain > 0.5
    suspect_True = suspect > 0.5
    target_True = target > 0.5
    intersection_certain = (certain_True & target_True).sum()
    intersection_suspect = (suspect_True & target_True).sum()
    certainA = certain_True.sum()
    B = target_True.sum()

    Precision = (intersection_certain + smooth) / (certainA + smooth)
    Recall = (intersection_suspect+ smooth) / (B + smooth)

    return 2*(Precision*Recall+smooth)/(Precision+Recall+smooth)
