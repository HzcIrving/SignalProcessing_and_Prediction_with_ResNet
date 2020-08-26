#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He
# email: 1910646@tongji.edu.cn

import numpy as np

class HHT_denosing(object):
    def __init__(self,maxlevel,columns_name,times,datapath=None):
        self.ml = maxlevel
        self.cn = columns_name
        self.times = times
        self.dp = datapath

    def HHT_op(self):
        times = np.array(0,1,1/60000)
        signal = np.sin(2*50*times)
        return times,signal

def add_white_noise(x,snr,sample_points):
    times = np.arange(0,1,1/sample_points)
    noise = np.random.rand(sample_points)
    return times,x+noise,noise