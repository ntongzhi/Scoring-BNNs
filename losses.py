import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

class BCEDiceLoss_splite(nn.Module):
    def __init__(self):
        super(BCEDiceLoss_splite, self).__init__()

    def forward(self, input, target):
        bce_sum = F.binary_cross_entropy_with_logits(input[0].unsqueeze(0),target[0].unsqueeze(0)).reshape([1])
        for i in range(input.size(0)-1):
            bce_i = F.binary_cross_entropy_with_logits(input[i+1].unsqueeze(0),target[i+1].unsqueeze(0)).reshape([1])
            bce_sum = torch.cat((bce_sum, bce_i),dim=0)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice
        loss = 0.5 * bce_sum + dice
        return loss.unsqueeze(-1).unsqueeze(-1)