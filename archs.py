# -*- coding: utf-8 -*-
from torch import nn
from torch.nn import functional as F
import torch
from torchsummary import summary
from Dropoutblock import DropBlock_search

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out

class Bayesdown_VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(Bayesdown_VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.dropout1 = DropBlock_search(3,0.5)#, size, batch_size)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.dropout2 = DropBlock_search(3,0.5)#, size, batch_size)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.dropout1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.dropout2(out)
        out = self.act_func(out)

        return out

class Score(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = VGGBlock(args.input_channels+1, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.mlp = nn.Linear(nb_filter[4], 1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0_gap = self.global_average_pooling(x4_0)
        output_reshap = x4_0_gap.view(input.size(0), -1)
        output = self.mlp(output_reshap.clone().detach())
        # output = output.unsqueeze(-1)
        output_tanh = torch.tanh(output.unsqueeze(-1))
        output_sotfmax = F.softmax(output_tanh, dim=0)

        return output_sotfmax

class BayesUNet_spatial(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        bs = args.batch_size

        self.conv0_0 = VGGBlock(args.input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = Bayesdown_VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])#, int(args.size/4), bs)
        self.conv3_0 = Bayesdown_VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])#, int(args.size/8), bs)
        self.conv4_0 = Bayesdown_VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])#, int(args.size/16), bs)

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)

        if self.training:
            return output
        else:
            return output, [x0_0, x1_0, x2_0, x3_0, x4_0, x3_1, x2_2, x1_3, x0_4]

if __name__ == '__main__':
    import argparse
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--input-channels', default=3, type=int,
                            help='input channels')
        parser.add_argument('--size', default=256, type=int,
                            help='image size')
        parser.add_argument('--image-ext', default='bmp',
                            help='image file extension')
        parser.add_argument('--mask-ext', default='bmp',
                            help='mask file extension')
        parser.add_argument('-b', '--batch_size', default=4, type=int,
                            metavar='N', help='mini-batch size (default: 16)')
        args = parser.parse_args()

        return args
    args = parse_args()
    model = BayesUNet_spatial(args).to(device)
    summary(model,(3,256,256))
