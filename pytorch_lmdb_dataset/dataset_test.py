#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He
# email: 1910646@tongji.edu.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from torch.utils.data import DataLoader
from dataset import LmdbDataset

import argparse

def dataset_test(lmdb_path,batch_size):
    dataset = LmdbDataset(lmdb_path)
    dataloader = DataLoader(dataset,batch_size,shuffle=False,num_workers=0)

    for i,data in enumerate(dataloader):
        img,label = data
        print(i,img,label)
        print(i,img.shape,label.shape)

def main():
    parser = argparse.ArgumentParser(
        description="lmdb dataset load test"
    )
    parser.add_argument("--lmdb_path",type=str,default=None,
                        help="Dataset path",required=True)
    parser.add_argument("--batch_size",type=int,default=32,
                        help="Mini batch size",required=True)
    args = parser.parse_args()
    dataset_test(args.lmdb_path,args.batch_size)

if __name__ == "__main__":
    main()

