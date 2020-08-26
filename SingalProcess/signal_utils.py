#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

# 工具包 List
# 1. 可选加入工频噪声
# 2. 可选加入趋势项
    # 趋势项: 信号中存在的线性项和变化缓慢的非线性项成分;
    # 产生原因: 采样时未对原始信号进行适当处理，使其含有周期大于采样时间的极低频成分
    #         环境温度变化引起的零点漂移等。
# 3. 可选加入全频段高斯白噪声

import numpy as np
from scipy.fftpack import fft,rfft,fftshift,ifft
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

# 多项式最小二乘拟合构造趋势项的GT(系数项)
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

import warnings
warnings.filterwarnings("ignore")

plt.style.use(['science','no-latex'])

# def wgn(x, snr):
#     Ps = np.sum(abs(x)**2)/len(x)
#     Pn = Ps/(10**((snr/10)))
#     noise = np.random.randn(len(x)) * np.sqrt(npower)
#     signal_add_noise = x + noise
#     return signal_add_noise

# Global Variable
Fs = 5000 # 采样频率
# num_fft = 100 # 采样点数

def fft_func(signal,sample_points=None):
    if sample_points is None:
        fft_signal = fft(signal)
    else:
        fft_signal = fft(signal,sample_points)
    fft_signal = fftshift(fft_signal)
    fft_signal = fft_signal[fft_signal.size//2:] # postive部分
    return fft_signal

def PowerSpectrum(signal,numfft):
    """功率谱函数"""
    num_fft = numfft
    Y = fft(signal,num_fft)
    Y = np.abs(Y)
    ps = Y**2/num_fft
    ps = 20*np.log10(ps)
    return ps

def add_white_gnoise(x,snr):
    """
    添加全频段高斯白噪声
    :param x:  原始信号
    :param snr:  信噪比
    :param seed:  随机factor
    :return:
    """
    # np.random.seed(seed)
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower/snr
    noise = np.random.randn(len(x))*np.sqrt(npower)
    return x + noise,noise

def add_50_Hz_noise(x,snr,pm,t,mode="Max_P"):
    """
    添加50hz工频噪声(48~52Hz)
    :param x:  原始信号
    :param snr:  信噪比
    :param seed:  随机factor
    mode: 噪声模式---以压力峰峰值为基准"Max_P";
                  ---以信噪比作为基准"SNR"
    :return:
    """
    # np.random.seed(seed)
    random_factor1 = 2 * (2 * np.random.rand() - 1)
    random_factor2 = 2 * (2 * np.random.rand() - 1)
    if mode == "snr":
        snr = 10**(snr/10.0)
        xpower = np.sum(x**2)/len(x)
        npower = xpower/snr
        noise = (np.sin(2*np.pi*(50+random_factor1)*t)+\
                 np.cos(2*np.pi*(50+random_factor2)*t))*np.sqrt(npower)*(np.random.rand()+1)
    elif mode == "Max_P":
        noise = 0.1*pm*(0.2*np.sin(2*np.pi*(50+random_factor1)*t)+\
                 0.8*np.cos(2*np.pi*(50+random_factor2)*t))
    else:
        noise = 0
        # if noise == 0:
        raise Exception("Non-exsitend mode")

    return x + noise,noise

def add_50hz_to_200Hz_noise(x,pm,times,weights=[0.5,0.3,0.2,0.1]):
    """
    条件50hz工频以及倍频噪声(50~200)
    :param x: signal
    :param pm:  峰值
    :param weight: 各倍频权重大小
    :param times: 时间序列
    :return: times,x+noise,noise
    """
    x = np.array(x)
    noise_space = np.zeros((len(weights),x.shape[0]))
    for i,v in enumerate(weights):
        random_factor1 = 2*(2*np.random.rand()-1) # 2~-2
        random_factor2 = 2*(2*np.random.rand()-1)
        random_factor3 = np.random.rand()
        noise_space[i] = 0.1*pm*(random_factor3*np.sin(2*np.pi*(50*(i+1)+random_factor1)*times)+\
                                 (1-random_factor3)*np.cos(2*np.pi*(50*(i+1)+random_factor2)*times))
    for i,noise in enumerate(noise_space):
        x += weights[i]*noise
    return times,x,noise_space

def retrending_signal_with_exp_type(signal,max_Pm,times,coeffs,intercepts,Fs=None):
    """
    添加指数型趋势项噪声
    Fs:采样频率; times: 时间序列; max_Pm: 压力峰峰值
    """
    Value_range = max_Pm*0.15
    # trend_seqs = Value_range*((1-np.exp(-times))+0.2*np.random.randn())
    trend_seqs = 0.004 + 0.956*(times)-0.391*(times**2)+0.065*(times**3) # 该部分系数由多项式插值得到
    trand_seqs_scale = trend_seqs*Value_range
    signal = signal + trand_seqs_scale
    return signal,trand_seqs_scale

def Output_current_order(times,signal,order=2):
    """输出当前多项式拟合相关系数等"""
    datasets_x = times
    datasets_y = signal
    length = len(times)
    datasets_x = np.array(datasets_x).reshape([length,1])
    datasets_y = np.array(datasets_y)
    # 建立多项式特征
    poly_reg = PolynomialFeatures(degree=order)
    X_poly = poly_reg.fit_transform(datasets_x)
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_poly,datasets_y)
    # 查看回归系数
    coeffs = lin_reg.coef_
    intercepts = lin_reg.intercept_
    return poly_reg,lin_reg,coeffs,intercepts

def retrending_signal_with_poly_type(Pm,order,times):
    """
    添加指数型的多项式趋势项
    一元order次趋势项
    """
    # np.random.seed(seed)
    times = np.array(times)
    coeffs = []
    # poly_fit = []
    poly = np.zeros((times.shape))
    for i in range(order):
        coeff = np.round(np.random.randn(),3)
        coeffs.append(coeff)
        poly += coeff*(np.power(times,(i+1)))
    intercept = np.round(-0.1*np.random.rand(),3)
    poly += intercept
    # poly /= np.max(poly) # 归一化
    poly = poly*0.1*Pm # 20%的Pm最终项
    return poly,coeffs,intercept

def Cable_noise_generator(times,pm,nums):
    """随机生成电缆噪声"""
    times = np.array(times)
    Nums = len(times)
    Cable_noise = np.zeros((times.shape))
    for i in range(nums):
        num = np.random.randint(Nums)
        Cable_noise[num] = np.random.rand()*pm
    return Cable_noise

if __name__ == "__main__":
    t = np.arange(0,2,1.0/Fs)

    plt.figure()
    plt.subplot(211)
    origin_signal = 1.5*np.sin(2*np.pi*50*t)+1
    _,add_noise,noise = add_50hz_to_200Hz_noise(origin_signal,2.5,t)
    plt.plot(t,origin_signal,'r')
    plt.plot(t,add_noise,'b')

    plt.subplot(212)
    fft = fft_func(add_noise)
    plt.plot(fft,'r')
    plt.show()


    # plt.figure()
    # plt.subplot(411)
    # origin_signal = 1.5*np.sin(2*np.pi*50*t)+1
    # plt.plot(t,origin_signal,'r')
    #
    # plt.subplot(412)
    # snr = 10 # db 当前信噪比
    # noise_signal,noise = add_white_gnoise(origin_signal,snr)
    # plt.plot(t,noise,'g')
    #
    # plt.subplot(413)
    # p = fft(noise_signal)
    # p = fftshift(p)
    # plt.plot(p)
    #
    # S = 0.7 * np.cos(2 * np.pi * 0.1 * t)
    # print(S.shape)
    # plt.subplot(414)
    # p = fft(S)
    # p = fftshift(p)
    # plt.plot(p)
    # # plt.show()
    #
    # plt.figure()
    # plt.subplot(211)
    # noise_50hz = np.sin(2*np.pi*50*t)+np.cos(2*np.pi*50*t)
    # p = fft_func(noise_50hz,Fs)
    # # p = fft(noise_50hz,200)
    # # p = fftshift(p)
    # # p = p[p.size//2:]
    # plt.plot(p)
    #
    # snr = 10
    # # plt.subplot(212)
    # # signal_noise,noise = add_50_Hz_noise(S,snr,t)
    # # fft_noise = fft_func(noise,Fs)
    # # # plt.plot(fft_noise)
    # # plt.plot(S)
    # # plt.plot(signal_noise)
    #
    # S = 0.7 * np.cos(2 * np.pi * 1 * t)
    # plt.figure()
    # plt.subplot(211)
    # # trend_item = (np.random.rand())*t./**4+(np.random.rand())*t**3+(np.random.rand())*t**2+(np.random.rand())*t+0.1
    # # trend_item = trend_item/np.max(trend_item)
    # trend_item = 1 - np.exp(-t)
    # # noise_s = add_white_gnoise(S,snr=10)
    # noise_50hz_trend = S + trend_item*0.7
    # print(trend_item.shape)
    # plt.plot(trend_item,'.')
    # plt.plot(noise_50hz_trend)
    # plt.plot(S)
    #
    # plt.subplot(212)
    # poly_reg,poly_fit_function,a,b = Output_current_order(t,trend_item,order=3)
    # plt.scatter(t,trend_item)
    # plt.plot(t,poly_fit_function.predict(poly_reg.fit_transform(t.reshape(-1,1))),color='blue')
    # print("coeffs GT:",a)
    # print("intercepts GT:",b)
    #
    # plt.figure()
    # poly,coeffs,intercept = retrending_signal_with_poly_type(6,3,t)
    # print("当前趋势项系数:",coeffs)
    # print("当前趋势项多项式截距:",intercept)
    # plt.scatter(t,poly)
    # # plt.show()
    #
    # plt.figure()
    # # 电缆噪声
    # Cable_noise = Cable_noise_generator(t,1)
    # plt.plot(t,Cable_noise)
    # plt.show()






