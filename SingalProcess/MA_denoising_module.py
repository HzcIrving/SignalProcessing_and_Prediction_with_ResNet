#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He
# email: 1910646@tongji.edu.cn

# 指数滑动平均
# vt = beta*vt+(1-beta)theta_t
# vt := vt/(1-beta^t)  解决开始偏差
# Exponential Moving Average --- 只需要记录上一次输出于当前样本值

# 窗口滑动平均滤波wma
# 连续把N个采样值堪称一个队列，队列长度为N
# FIFO原则
# 存在边际效应，边缘点需要重新选取 --- 需要记录M个样本

# 抗周期干扰，平滑度高
# 可滤除零飘
# 可滤除cable noise

# decay factor : gamma 控制模型更新的速度，越大越稳定 --- for ema
# window size : N 控制窗口长度 --- for wma

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from signal_utils import *

class MA(object):
    """
    滑动平均工具
    """
    def __init__(self,beta,window_size,data0=0,data_path=None):
        """
        参数
        :param beta:衰减因子
        :param window_size: WMA的窗口大小，必须为基数
        :param data_path: 数据路径
        :param data0 : 数据的首个参数 --- for EMA
        """
        self.decay = beta
        self.N = window_size
        self._one_minus_alpha = 1 - self.decay
        self._adjustment = 1
        self._s = 0

    def WMA(self,times,signal):
        """
        窗口滑动平均
        """
        smooth_out = np.convolve(signal,np.ones(self.N,dtype=int),'valid')/self.N
        r = np.arange(1,self.N-1,2)
        start = np.cumsum(signal[:self.N-1])[::2]/r
        end = (np.cumsum(signal[:-self.N:-1])[::2]/r)[::-1]
        zero_shifting = np.concatenate((start,smooth_out,end))
        denoising_signal = signal - zero_shifting
        return times, denoising_signal

    def CorrectedEMA(self,times,signal):
        """
        指数滑动平均
        """
        smoothed_out = []
        for data in signal:
            smoothed_out.append(self._s)
            self._update(data)
            self._s = self._value()
        return times,np.array(smoothed_out)

    # Corrected_EMA part
    def _update(self,x):
        self._s += self.decay*(x-self._s)
        self._adjustment *= self._one_minus_alpha

    def _value(self):
        return self._s*1/(1-self._adjustment)

if __name__ == "__main__":
    # 测试
    Fs = 5000
    SNR = 10
    times = np.arange(0,1,1.0/Fs)

    filter = MA(0.999,551)

    plt.figure()
    origin_signal = (1.5*np.sin(2*np.pi*50*times))+0.8*np.cos(2*np.pi*40*times)
    origin_signal,_ = add_white_gnoise(origin_signal,snr=SNR)
    origin_signal += Cable_noise_generator(times,5,50)
    _,origin_signal,_ = add_50hz_to_200Hz_noise(origin_signal,5,times)
    origin_signal += times # zero shifting
    print(origin_signal.shape)
    plt.plot(times,origin_signal,'-')
    plt.plot(times,times,'-r')
    plt.plot(times,np.zeros((len(times),)))

    # plt.figure()
    # times,smoothed_out = filter.WMA(times,origin_signal)
    # plt.plot(times,smoothed_out,'-r',label="WMA Filter")
    # print(smoothed_out.shape)
    # plt.legend()

    plt.figure()
    times,smoothed_out = filter.WMA(times,origin_signal)
    plt.plot(times,smoothed_out,'-g',label="WMA Filter")
    plt.plot(times,np.zeros((len(times),)))
    print(smoothed_out.shape)
    plt.legend()

    # plt.figure()
    # origin_signal += Cable_noise_generator(times,5,50)
    # print(origin_signal.shape)
    # plt.plot(times,origin_signal,'--')
    # times,smoothed_out2 = filter.CorrectedEMA(times,origin_signal)
    # plt.plot(times,smoothed_out2,'-r',label="Corrected_EMA Filter")
    # print(smoothed_out.shape)
    # plt.legend()

    plt.show()





