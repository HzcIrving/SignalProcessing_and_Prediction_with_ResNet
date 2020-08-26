#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He
# email: 1910646@tongji.edu.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import lmdb
import numpy as np

from proto import utils
from proto import tensor_pb2

import pandas as pd

import sys
sys.path.append("Datasets_path_csv")

# 爆炸信号四阶趋势项数据s
csv_path = 'E:\Speakin_Workspace\Datasets_path_csv\explo_data.csv'
label_path = 'E:\Speakin_Workspace\Datasets_path_csv\label.csv'

def csv_reader(datasets_csv_path=csv_path,datasets_label_path=label_path):
    origin_data = pd.read_csv(datasets_csv_path)
    origin_label = pd.read_csv(datasets_label_path)
    origin_data = np.array(origin_data)[:, 1:].T
    origin_label = np.array(origin_label)[:, 1:].T
    origin_train = origin_data[0:1000]
    origin_test = origin_data[1000:1100]
    train_label = origin_label[0:1000]
    test_label = origin_label[1000:1100]

    return origin_train,train_label,origin_test,test_label

def create_train_db(output_file,origin_data,origin_label):

    # 原始数据:1000x2048
    # 原始label:11x5

    print(">>> Write database...")
    LMDB_MAP_SIZE = 1 << 40   # MODIFY
    print(LMDB_MAP_SIZE)
    env = lmdb.open(output_file,map_size=int(1e9))

    checksum = 0
    with env.begin(write=True) as txn:
        for j in range(len(origin_data)):
            # -------------------------------------------------------------
            # width = 64
            # height = 32
            # img_data = np.random.rand(3, width, height).astype(np.float32)
            # label = np.asarray(j % 10)
            explo_train_data = np.array(origin_data[j]).astype(np.float32)
            explo_train_label = np.array(origin_label[j])
            img_data = explo_train_data
            label = explo_train_label

            # 1D datasets (explosion datasets)
            # 11*2000*1
            # width =  1
            # height = 2000
            # explosion_data = np.random.rand(width,height).astype(np.float32)
            # -------------------------------------------------------------

            # Create TensorProtos
            tensor_protos = tensor_pb2.TensorProtos()
            img_tensor = utils.numpy_array_to_tensor(img_data)
            tensor_protos.protos.extend([img_tensor])

            label_tensor = utils.numpy_array_to_tensor(label)
            tensor_protos.protos.extend([label_tensor])
            txn.put(
                '{}'.format(j).encode('ascii'),
                tensor_protos.SerializeToString()
            )

            if (j % 10 == 0):
                print("Inserted {} rows".format(j))

def create_test_db(output_file,origin_data,origin_label):

    # 原始数据:500x2000
    # 原始label:11x5

    print(">>> Write database...")
    LMDB_MAP_SIZE = 1 << 40   # MODIFY
    print(LMDB_MAP_SIZE)
    env = lmdb.open(output_file)

    checksum = 0
    with env.begin(write=True) as txn:
        for j in range(len(origin_data)):
            # -------------------------------------------------------------
            # width = 64
            # height = 32
            # img_data = np.random.rand(3, width, height).astype(np.float32)
            # label = np.asarray(j % 10)
            explo_test_data = np.array(origin_data[j]).astype(np.float32)
            explo_test_label = np.array(origin_label[j])
            img_data = explo_test_data
            label = explo_test_label

            # 1D datasets (explosion datasets)
            # 11*2000*1
            # width =  1
            # height = 2000
            # explosion_data = np.random.rand(width,height).astype(np.float32)
            # -------------------------------------------------------------

            # Create TensorProtos
            tensor_protos = tensor_pb2.TensorProtos()
            img_tensor = utils.numpy_array_to_tensor(img_data)
            tensor_protos.protos.extend([img_tensor])

            label_tensor = utils.numpy_array_to_tensor(label)
            tensor_protos.protos.extend([label_tensor])
            txn.put(
                '{}'.format(j).encode('ascii'),
                tensor_protos.SerializeToString()
            )

            if (j % 16 == 0):
                print("Inserted {} rows".format(j))

def main():
    data,label,test_data,test_label = csv_reader()

    parser = argparse.ArgumentParser(
        description="LMDB creation"
    )

    # choose the generation type
    # parser.add_argument("--train_output_file", type=str, default=None,
    #                     help="Path to write the training database to",
    #                     required=True)
    parser.add_argument("--test_output_file", type=str, default=None,
                        help="Path to write the testing database to",
                        required=True)


    args = parser.parse_args()

    # create_train_db(args.train_output_file,data,label)
    create_test_db(args.test_output_file,test_data,test_label)

if __name__ == '__main__':
    main()

    # od,ol = csv_reader()
    # print(type(od[1]))
    # print(od.shape)
    # print(type(od))
    # print(od)
    # print(ol.shape)

    # from lmdb_datasets_generation import *
    # env = initialize()
    # display(env)
    #
    # tensor_protos = tensor_pb2.TensorProtos()

    # pass
    # train,train_label,test,test_label = csv_reader(csv_path,label_path)
    # print(train.shape)
    # print(train_label.shape)
    # print(test.shape)
    # print(test_label.shape)