#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He
# email: 1910646@tongji.edu.cn

# 信号可视化工具

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use(['science','no-latex'])

def signal_plot(pure_signal,add_noise_signal,a,b):
    """
    The visualization of signal
    :param pure_signal:  original signal
    :param add_noise_signal:  signal after noise-adding operation
    :param a: poly coeffs a
    :param b: poly coeffs (intercept) b
    :return:
    """
    plt.figure()
    plt.plot(pure_signal,'--',markersize=2,label='original')
    plt.plot(add_noise_signal,'-r.',markersize=1,label='Simulation explosion signal')
    # plt.title()
    plt.title("Trending polynomials, coeffs=" + str(np.round(a, 3)) + ", intercept=" + str(np.round(b, 3)))
    plt.grid()
    plt.legend()




