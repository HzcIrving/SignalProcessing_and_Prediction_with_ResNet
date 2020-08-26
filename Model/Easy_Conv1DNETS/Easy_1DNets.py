#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He
# email: 1910646@tongji.edu.cn

# Conv1D Nets for regression

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class EasyConv1D(nn.Module):
    def __init__(self):
        super(EasyConv1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(32, 40, 5)
        self.conv2_bn = nn.BatchNorm1d(40)
        self.conv3 = nn.Conv1d(40, 64, 4)
        self.conv3_bn = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 3)
        self.conv4_bn = nn.BatchNorm1d(128)

        self.relu = nn.ReLU(inplace=True)
        # self.celu =nn.CELU(inplace=True)

        self.fc1 = nn.Linear(15616, 5000)
        self.fc2 = nn.Linear(5000,5)

    def forward(self,x):

        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv1_bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv2_bn(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv3_bn(x)
        x = self.relu(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    net = EasyConv1D()
    print(net)

