#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He
# email: 1910646@tongji.edu.cn

"""
常见小波函数
"Haar" 阶跃函数，比较常见
"dbN" 小波阶数不同则N的数值也不同，阶数为1则为Haar小波，当阶数大于1时，是一种非对称
      小波
“symN" 小波同样保持了dbN的简单性，并且symN小波系具备良好的小波对称性，在滤波长度、
      支集长度还有连续性方面均与dbN小波系相同；
"colfN" 小波系, 相比于拥有更好的对称性
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Signal_Tool_utils import * # FFT

# filter
import scipy
from scipy import signal
from scipy.signal import butter

import matplotlib
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")

import pywt
from sklearn.metrics import mean_squared_error

plt.style.use(['science','no-latex'])

# -- Global Vars --
# SNR bigger, better
# MSE smaller, better
DATA_path = "E:\Speakin_Workspace\Acceleration_signal\original_data\set3(9).txt"
NAMES = ['time','data']
TYPE = 'sym4' # wavelet params
THRESHOLD = 0.1 # threshold for denosing op

# -- Utils --
# 1. dataloader
def dataloader_acc(datapath,columns_name):
    df_data = pd.read_table(DATA_path)
    # add columns name
    df_data.columns = NAMES
    time  = df_data[NAMES[0]]
    signal = df_data[NAMES[1]]
    time = np.array(time)
    signal = np.array(signal)
    return time,signal

# 2. get sample freq
def get_freq(t,t_):
    return 1/(t_-t)

# 3. wavelet denoising
# --------------------------------------
# 3.1 对含噪信号进行尺度小波分解: 通过分析选择合理的小波函数以继特定的分解尺度
#     对含噪信号进行特定尺度的目标层数分解
# 3.2 小波阈值处理: 设定阈值对信号尺度分解出的小波系数进行选择，若小波系数的幅
#     值低于该阈值则认定为是由噪声引起的，则舍弃这个分解系数；
# 3.3 小波系数重构: 一维小波重构分解后的小波系数 ---> 去噪信号
def wavelets_denosing(data,type):
    # create wavelet object
    w = pywt.Wavelet(type)
    # maxlev = pywt.dwt_max_level(len(data),w.dec_len)
    maxlev = 10
    print("Maximum level is" + str(maxlev))

    # decompose into wavelet components, to the level selected
    coeffs = pywt.wavedec(data,type,level=maxlev)

    sigma = (1/0.6745) * maddest(coeffs[-maxlev])
    uthresh = sigma * np.sqrt( 2*np.log( len( data ) ) )
    print("Threshold: ",uthresh,"/g")

    for i in range(1,len(coeffs)):
        plt.subplot(maxlev,1,i)
        plt.plot(coeffs[i],label="Before")
        # coeffs[i] = pywt.threshold(coeffs[i],threshold*max(coeffs[i]))
        coeffs[i] = pywt.threshold(coeffs[i], value=uthresh, mode='hard' )
        plt.plot(coeffs[i],label="After")

    # reconstruct
    datarec = pywt.waverec(coeffs,type)
    return datarec
# --------------------------------------

# 4. dwt composition and denosing
# S = An + Dn + Dn-1 + ... + D1
# --------------------------------------
# 4.1 Perform a discrete wavelet transformation to obtain wavelet coeffs wj(x)
#     on level [j] from signal x which consists of approximation [ca] and detail [cd] coeffs
# 4.2 Estimate the threshold Td at decomposition level j vis (MAD stands for Mean Absolute Deviation)
#     delta = 1/0.6745MAD(|cd|）
#     Td = delta * np.sqrt(2*np.log(n))
# 4.3 Perform a hard threshold op on detailed coeffs and truncate(截短) approximation coefficients

def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def high_pass_filter(x, low_cutoff, sample_rate):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    """

    # nyquist frequency is half the sample rate
    nyquist = 0.5 * sample_rate
    norm_low_cutoff = low_cutoff / nyquist

    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    # scipy version 1.2.0
    #sos = butter(10, low_freq, btype='hp', fs=sample_fs, output='sos')

    # scipy version 1.1.0
    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = signal.sosfilt(sos, x)

    return filtered_sig

def denoise_signal( x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )

    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1/0.6745) * maddest( coeff[-level] )

    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )

    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec( coeff, wavelet, mode='per' )
# --------------------------------------

# 5. -------Evaluation metrics func-------------
def numpy_SNR(origianl_waveform, target_waveform):
    # 单位 dB
    # 信噪比
    signal = np.sum(origianl_waveform ** 2)
    noise = np.sum((origianl_waveform - target_waveform) ** 2)
    snr = 10 * np.log10(signal / noise)
    return snr

def psnr(ref_wav, in_wav):
    # 峰值信噪比
    MSE = np.mean((ref_wav - in_wav) ** 2)
    MAX = np.max(ref_wav)       # 信号的最大平时功率
    return 20 * np.log10(MAX / np.sqrt(MSE))

def mse(ref_wav,in_wav):
    return mean_squared_error(ref_wav,in_wav)

if __name__ == "__main__":
    # test 1 signal plot
    Time,Signal = dataloader_acc(DATA_path,NAMES)
    plt.figure(figsize=(10,6))
    plt.subplot(211) # time-domain
    plt.xlabel("Time/s")
    plt.ylabel("Acc/g")
    plt.tight_layout()
    plt.plot(Time,Signal,'k-')

    # test 2 fft plot
    Fs = get_freq(Time[1],Time[2])
    fft,freq = FFT(Fs,Signal)
    plt.subplot(212)
    plt.xlabel("Freq/hz")
    plt.ylabel("Power")
    plt.tight_layout()
    plt.plot(freq,fft,'b-')

    # test 3 wavelet denosing
    plt.figure(figsize=(10,12))
    data_rec = wavelets_denosing(Signal,TYPE)
    plt.tight_layout()
    plt.legend()

    plt.figure(figsize=(10,6))
    plt.subplot(311) # time-domain
    plt.xlabel("Time/s")
    plt.ylabel("Acc/g")
    plt.tight_layout()
    plt.plot(Time,Signal,'k-')

    plt.subplot(312)
    plt.xlabel("Time/s")
    plt.ylabel("Acc/g")
    plt.plot(Time,data_rec,'r-')
    plt.tight_layout()


    fft1,freq1 = FFT(Fs,data_rec)
    plt.subplot(313)
    plt.xlabel("Freq/hz")
    plt.ylabel("Power")
    plt.tight_layout()
    plt.plot(freq1,fft1,'b-')
    plt.show()

    SNR = numpy_SNR(Signal,data_rec)
    PSNR = psnr(Signal,data_rec)
    MSE = mse(Signal,data_rec)
    print("SNR betweeen the two signals:",SNR)
    print("PSNR between the two signals:",PSNR)
    print("MSE between the two signals:",MSE)


    # test 4 dwt wavelet denosing
    # x_hp = high_pass_filter(Signal, low_cutoff=10000, sample_rate=Fs)
    # x_dn = denoise_signal(x_hp, wavelet='db8', level=13)
    # plt.subplot(311)
    # plt.plot(Signal,label="Original")
    # plt.legend()
    # plt.subplot(312)
    # plt.plot(x_hp,label="high pass filter")
    # plt.legend()
    # plt.subplot(313)
    # plt.plot(x_dn,label="wavelet denosing")
    # plt.legend()
    # plt.show()



