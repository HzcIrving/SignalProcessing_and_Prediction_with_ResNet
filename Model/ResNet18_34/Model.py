#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He
# email: 1910646@tongji.edu.cn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Fine-tuneable
# def replace_head(model, num_classes):
#     model[-1][-1] = nn.Linear(512, num_classes)
#     apply_init(model[1], nn.init.kaiming_normal_)

# conv1xk (3x3,stride=1)
def conv1xk(in_channels,out_channels,kernel_size=3,stride=1):
    padding = kernel_size//2
    return nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,
                        stride=stride,padding=padding,bias=False)

# initialization
def init_cnn_1d(m):
    if getattr(m, 'bias', None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d,nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn_1d(l)

# def  splitter(m):
#     return L(m[0][:6], m[0][6:], m[1]).map(params)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv1xk(in_channels, out_channels, kernel_size, stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1xk(out_channels, out_channels, kernel_size)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out

# Resnet1D
class ResNet_explo(nn.Sequential):
    def __init__(self,block,layers,in_channels=64,num_classes=5,kernel_size=3,stride=2,dropout=0.2):
        """

        :param block:
        :param layers:
        :param in_channels:
        :param num_classes:
        :param kernel_size:
        :param stride:
        :param dropout:
        """
        in_block = [] # input_block
        residual_block = [] # residual blocks
        header_block = [] # linear head

        self.in_channels = in_channels
        self.block = block

        in_kernel_size = kernel_size*2 + 1

        in_block.append(conv1xk(1, in_channels, in_kernel_size, stride))
        in_block.append(nn.BatchNorm1d(in_channels))
        in_block.append(nn.ReLU(inplace=True))
        in_block.append(nn.MaxPool1d(kernel_size, stride, kernel_size//3))

        residual_block = self.make_blocks(layers, in_channels, kernel_size, stride)

        header_block.append(nn.AdaptiveAvgPool1d(1))
        header_block.append(nn.Flatten())
        header_block.append(nn.Dropout(dropout))
        header_block.append(nn.Linear(in_channels*2**(len(layers)-1), num_classes))

        super().__init__(nn.Sequential(*in_block,*residual_block),nn.Sequential(*header_block))
        init_cnn_1d(self)

    def make_blocks(self, layers, in_channels, kernel_size, stride):
        return [self.make_layer(self.block, in_channels*2**i, l, kernel_size, stride) for i, l in enumerate(layers)]

    def make_layer(self, block, out_channels, blocks, kernel_size=3, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv1xk (self.in_channels, out_channels, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, kernel_size, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, kernel_size))
        return nn.Sequential(*layers)


# if __name__ == "__main__":
#
#     # resnet 18
    resnet1d18 = {
        "block": ResidualBlock,
        "layers": [2, 2, 2, 2],
        "in_channels": 64,
        "kernel_size": 15,
        "stride": 4,
        "num_classes": 10
    }
#
#     # resnet 34
#     resnet1d34 = {
#         "block": ResidualBlock,
#         "layers": [3, 4, 6, 3],
#         "in_channels": 64,
#         "kernel_size": 15,
#         "stride": 4,
#         "num_classes": 10
#     }
#
#     bs = 8
#     arch = resnet1d18
#     model = ResNet_explo(**arch)
#     inp = torch.randn(bs,1,2048)
#     out = model(inp)
#     print(model)
#     assert len(out) == bs
#     assert len(out[0]) == arch['num_classes']