#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He
# email: 1910646@tongji.edu.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from torch.utils.data import Dataset

from proto import tensor_pb2
from proto import utils

class ExplosionDataset(Dataset):
    """Explo Dataset"""
    def __init__(self,lmdb_path):
        super(ExplosionDataset,self).__init__()
        import lmdb
        self.env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
            self.keys = [key for key,_ in txn.cursor()]

    def __getitem__(self, item):
        # 重写
        with self.env.begin(write=False) as txn:
            serialized_str = txn.get(self.keys[index])
        tensor_protos = tensor_pb2.TensorProtos()
        tensor_protos.ParseFromString(serialized_str)
        img = utils.tensor_to_numpy_array(tensor_protos.protos[0])
        label = utils.tensor_to_numpy_array(tensor_protos.protos[1])
        return img, label

    def __len__(self):
        return self.length

