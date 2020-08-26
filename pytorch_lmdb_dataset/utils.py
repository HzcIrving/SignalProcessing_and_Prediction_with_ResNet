# #! /usr/bin/enc python
# # -*- coding: utf-8 -*-
# # author: Irving He
# # email: 1910646@tongji.edu.cn
#
# from torch.utils.data.sampler import SubsetRandomSampler
# from torch.utils.data import DataLoader
# import torch
# import numpy as np
#
# from torch.autograd import Variable
#
#
# def prepare_train_valid_loaders(trainset, valid_size=0.2,
#                                 batch_size=128):
#     '''
#     Split trainset data and prepare DataLoader for training and validation
#
#     Args:
#         trainset (Dataset): data
#         valid_size (float): validation size, defalut=0.2
#         batch_size (int) : batch size, default=128
#     '''
#
#     # obtain training indices that will be used for validation
#     num_train = len(trainset)
#     indices = list(range(num_train))
#     np.random.shuffle(indices)
#     split = int(np.floor(valid_size * num_train)
#     train_idx, valid_idx = indices[split:], indices[:split]
#
#     # define samplers for obtaining training and validation batches
#     train_sampler = SubsetRandomSampler(train_idx)
#     valid_sampler = SubsetRandomSampler(valid_idx)
#
#     # prepare data loaders
#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                                sampler=train_sampler)
#     valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                                sampler=valid_sampler)
#     return train_loader, valid_loader
#
