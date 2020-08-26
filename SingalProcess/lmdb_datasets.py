#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He
# email: 1910646@tongji.edu.cn

"""
制作lmdb数据集
--- 1. 制作1D的爆炸信号的趋势项数据集
--- --- data: explosion signal; label:
env = lmdb.open()：创建 lmdb 环境
txn = env.begin()：建立事务
txn.put(key, value)：进行插入和修改
txn.delete(key)：进行删除
txn.get(key)：进行查询
txn.cursor()：进行遍历
txn.commit()：提交更改
"""

datasets_path = "E:\Speakin_Workspace\Explosion_Datasets_lmdb"

# 四阶趋势项数据
datasets_csv_path = "E:\Speakin_Workspace\Datasets_path_csv\explo_data.csv"
datasets_label_path = "E:\Speakin_Workspace\Datasets_path_csv\label.csv"

import os
import lmdb
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append(datasets_path)

import numpy as np
import pandas as pd

current_path = os.getcwd()
print("current path:",os.getcwd())

origin_data = pd.read_csv(datasets_csv_path)
origin_label = pd.read_csv(datasets_label_path)
# print(origin_data.head())
# print(origin_label.head())

columns_data = list(origin_data.columns)
columns_label = list(origin_label.columns)
columns_data = columns_data[1:]
columns_label = columns_label[1:]
print(columns_data)
print(columns_label)

env = lmdb.open(datasets_path,max_dbs=4)
train_data = env.open_db("train_data".encode())
train_label = env.open_db("train_label".encode())

txn = env.begin(write=True)

origin_data = np.array(origin_data)
origin_data = origin_data[:,1:].T
print(origin_data.shape)

origin_label = np.array(origin_label)
origin_label = origin_label[:,1:].T
print(origin_label.shape)

# add key & value
for i in range(11):
    print("Data Current columns:",columns_data[i])
    print("Label Current columns:",columns_label[i])
    print(origin_data[i,:])
    print(origin_label[i,:])
    txn.put(key=str(columns_data[i]).encode(),value=origin_data[i,:].tobytes(),db=train_data)
    txn.put(key=str(columns_label[i]).encode(),value=origin_label[i,:].tobytes(),db=train_label)

# txn.commit()
# env.close()







