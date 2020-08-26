#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He
# email: 1910646@tongji.edu.cn

import os
import lmdb
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# datasets_path = "E:\Speakin_Workspace\Datasets"
datasets_path = "E:\\Speakin_Workspace\\pytorch_lmdb_dataset\\train_explo_lmdb"
traindata = "train_data"
trainlabel = "train_label"
max_dbs = 2 # train&label, test&label

# 四阶趋势项数据s
datasets_csv_path = "E:\Speakin_Workspace\Datasets_path_csv\explo_data.csv"
datasets_label_path = "E:\Speakin_Workspace\Datasets_path_csv\label.csv"

def initialize():
    # 数据集初始化
    env = lmdb.open(datasets_path)
    return env

def insert(env, sid, name):
    # 插入
    txn = env.begin(write=True)
    txn.put(str(sid).encode(), name.encode())
    txn.commit()

def delete(env, sid):
    # 删除
    txn = env.begin(write=True)
    txn.delete(str(sid).encode())
    txn.commit()

def update(env, sid, name):
    # 更新
    # 每次 commit() 之后都要用 env.begin() 更新 txn（得到最新的lmdb数据库）
    txn = env.begin(write=True)
    txn.put(str(sid).encode(), name.encode())
    txn.commit()

def search(env, sid):
    # 查找
    txn = env.begin()
    name = txn.get(str(sid).encode())
    return name

def display(env):
    # 遍历显示
    txn = env.begin()
    cur = txn.cursor()
    for key, value in cur:
        print(key, value)

def datatolmdb(dataset_path=datasets_path,map_size=int(1e12)):
    # create the dataset file
    env = lmdb.open(dataset_path,max_dbs=max_dbs)
    # create corresponding dataset
    # train_data = env.open_db(traindata.encode())
    # train_label = env.open_db(trainlabel.encode())
    # test_data = env.open_db(testdata)
    # test_label = env.open_db(testlabel)

    origin_data = pd.read_csv(datasets_csv_path)
    origin_label = pd.read_csv(datasets_label_path)

    columns_data = list(origin_data.columns)
    columns_label = list(origin_label.columns)
    columns_data = columns_data[1:]
    columns_label = columns_label[1:]
    print(columns_data)
    print(columns_label)

    origin_data = np.array(origin_data)
    origin_data = origin_data[:, 1:].T
    print(origin_data.shape)

    origin_label = np.array(origin_label)
    origin_label = origin_label[:, 1:].T
    print(origin_label.shape)

    # put
    with env.begin(write=True) as txn:
        for i in range(11):
            print("Data Current columns:", columns_data[i])
            print("Label Current columns:", columns_label[i])
            print(origin_data[i, :])
            print(origin_label[i, :])
            txn.put(key=str(columns_data[i]).encode(), value=origin_data[i, :].tobytes())
            txn.put(key=str(columns_label[i]).encode(), value=origin_label[i, :].tobytes())
            # txn.commit()

    # env.close()
    return columns_data,columns_label


if __name__  == "__main__":
#     # columns_data,columns_label = datatolmdb()
#     # env = lmdb.open(datasets_path, max_dbs=max_dbs)
    env = initialize()
    display(env)
#
    K = search(env,"Label4")
    print(K)
    print(K.decode())






