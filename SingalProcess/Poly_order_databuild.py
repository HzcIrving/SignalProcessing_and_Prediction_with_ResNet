#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He
# email: 1910646@tongji.edu.cn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from Pressure import *
from signal_utils import *

import sys
sys.path.append("E:\Speakin_Workspace\pytorch_lmdb_dataset")
from create_lmdb import *
from dataset import *

import warnings
warnings.filterwarnings("ignore")

plt.style.use(['science','no-latex'])

"""
脚本说明
该脚本用于生成带有趋势项阶数的lmdb标准数据集 
1. 构建数据 ---> .csv 
2. 构建标签 ---> .csv 
3. lmdb标准数据集创建 ---> .mdb 
"""

class Poly_order_datasets_building(object):
    """poly order prediction datasets"""
    def __init__(self):
        pass

    def data_generator(self,data_path,label_path,capacity):
        df_data = pd.DataFrame()
        label_data = pd.DataFrame()


def trending_generator(Pm,percent,times,max_order,capacity):
    """
    :param Pm: max_pressure
    :param percent : 峰值衰减比例
    :param times: time series
    :param max_orders: ..
    :param capacity init value 2000
    2000个样本,最高10阶多项式，则:
    每3阶
    """
    times = np.array(times)
    print("时间序列维度:",times.shape)

    interval = int(capacity/max_order)
    print(interval)
    order = 1 # initial order
    orders_record = []

    poly = np.zeros((capacity,times.shape[0]))
    print(poly.shape)
    for i in range(capacity):
        intercept = np.random.randn()
        for j in range(order):
            coeff = np.random.randn() # -1~+1
            poly[i] += coeff*(np.power(times,(j+1)))
            orders_record.append(order) # 记录当前的order
        poly[i] += intercept
        if (i+1)%interval == 0:
            print("current i:{}, order:{}".format(i,order))
            order += 1

    return poly*percent*Pm,orders_record

if __name__ == "__main__":
    times = np.arange(0.0,1.0,1/60000)
    print(times.shape)
    poly,orders_record= trending_generator(10,0.2,times,10,200)

    print(poly.shape)

    plt.figure()
    plt.plot(orders_record)
    plt.show()








