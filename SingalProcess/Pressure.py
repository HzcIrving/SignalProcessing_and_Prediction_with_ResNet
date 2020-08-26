#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
自由场压力信号发生器
冲击波->一次气泡脉冲->二次气泡脉冲->三次气泡脉冲
"""

import matplotlib.pyplot as plt
import pylab as pl
import scipy
import numpy as np
import math
from signal_utils import *
from IIR_utils import *
from Data_utils import *
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

import os

dataset_path = "E:\Speakin_Workspace\Datasets_path_csv"

plt.style.use(['science','no-latex'])

# 全局
PHO = 1.52 # g/cm3
W = 100 # 炸药重量
R_0 = round((np.power(1000000*3/(4*np.pi)*1/PHO,1/3))*0.01,2) # 炸药半径
print(R_0)
R = 8 # 压传与爆炸中心的距离
# THETA = 1.0 # 冲击波衰减时间
H = 20 # 深度
Q = 4.184e6 # J/kg TNT爆热
Fs = 2048
NUM_PTS = 2048 # 采样点
SNR = 10

class PressureSignalGenerator(object):
    """
    自由场压力传感器信号发生器
    """
    def __init__(self,w,r_0,r,depth,q):
        self.w = w
        self.r_0 = r_0
        self.r = r
        self.a = math.pow(self.w, 1 / 3) / self.r
        self.theta = self.calc_theta()
        # self.T = time
        self.h = depth # 爆炸深度
        self.first_boost = (50*np.random.rand()+100)*self.theta

        # 炸药特性
        self.KR = 0
        self.KT = 0

        # 炸药爆热
        self.Q = q

        # 峰值压力
        self.p_m = self.calc_max_pressure()
        print("PM: pm = ",self.p_m,"MPa")
        self.p_m2 = 0.2*self.p_m
        print("PM2: pm2 = ", self.p_m2, "MPa")

        # 第一次气泡压力出现时间
        self.period = round(self.calc_cycle_period(times=1),2)
        self.period_left = round(self.period-0.11*self.period,2)
        self.period_right = round(self.period+0.11*self.period,2)
        print("气泡压力峰值第一次出现时刻: T1 = ",self.period,"s")
        print("气泡产生压力起始时间: T1l = ",self.period_left,"s")
        print("气泡产生压力起始时间: T1r = ", self.period_right,"s")

        self.period_pulse = self.period_right - self.period_left
        print("气泡时间: Tp = ",self.period_pulse,"s")

        # 计算第一个气泡r
        self.Rb1 = self.calc_pulse_r()

        print("气泡半径 Rb1 = ",self.Rb1,"m")

    def calc_max_pressure(self):
        """计算压力峰值"""
        if self.r/self.r_0 <=12 and self.r/self.r_0>=6:
            return 44.1*(math.pow(self.a,1.5))
        elif self.r/self.r_0>12:
            return 54.2*(math.pow(self.a,1.13))

    def calc_theta(self):
        """return信号衰减时间"""
        theta = 1e-4*math.pow(self.w,1/3)*math.pow(self.a,-0.24)
        print("衰减时间: theta=",theta)
        return theta

    def generator_pressure(self,current_time,model_type=0):
        c = 0.368 * self.theta / current_time
        c = c*(1-np.power(current_time/self.period,1.5))
        pt = c * self.p_m
        if current_time <= self.theta:
            if model_type == 0:
                pt = self.p_m* np.exp(-current_time/self.theta)
                # print()
                return pt,self.p_m
            elif model_type == 1:
                pt = self.p_m*(1-current_time/self.theta)*np.exp(-current_time/self.theta)
                return pt,self.p_m
        return pt, self.p_m

    def generator_burble1_pressure(self,current_time,decay=True):
        # self.p_m = self.calc_max_pressure()
        r = self.calc_pulse_Max_r()
        T = self.period
        # Pm = 9.03*np.power(self.w,1/3)/r
        self.Pm2 = 0.25*self.p_m
        pt = 0
        if decay == True:
            if current_time >= T-0.11*T and current_time < T:
                pt = (self.Pm2-0.8*self.Pm2) * np.exp(-(T - current_time) / self.theta)
            if current_time >= T and current_time < T+0.11*T:
                pt = self.Pm2* np.exp(-(current_time-T)/self.theta)
        else:
            if current_time >= T - 0.11 * T and current_time < T:
                pt = self.Pm2 * np.exp(-(T - current_time) / self.theta)
            if current_time >= T and current_time < T + 0.11 * T:
                pt = self.Pm2 * np.exp(-(current_time - T) / self.theta)
        return pt, self.Pm2

    def generator_burble2_pressure(self,current_time):
        # self.p_m = self.calc_max_pressure()
        r = self.calc_pulse_Max_r()
        T = self.period+self.period_pulse
        # Pm = 9.03*np.power(self.w,1/3)/r
        self.Pm3 = 0.25*self.Pm2
        pt = 0
        if current_time >= T-0.11*T and current_time < T:
            pt = self.Pm3*np.exp(-(T-current_time)/self.theta)
        if current_time >= T and current_time < T+0.11*T:
            pt = self.Pm3* np.exp(-(current_time-T)/self.theta)

        self.period_pulse2 = 0.22*T
        return pt, self.Pm3

    # def generator_burble2_pressure(self,current_time):

    def generator_burble3_pressure(self,current_time):
        # self.p_m = self.calc_max_pressure()
        r = self.calc_pulse_Max_r()
        T = self.period+self.period_pulse+self.period_pulse2
        # Pm = 9.03*np.power(self.w,1/3)/r
        self.Pm4 = 0.25*self.Pm3
        # self.period_pulse2 = 0.22*T
        pt = 0
        if current_time >= T-0.11*T and current_time < T:
            pt = self.Pm4*np.exp(-(T-current_time)/self.theta)
        if current_time >= T and current_time < T+0.11*T:
            pt = self.Pm4* np.exp(-(current_time-T)/self.theta)
        return pt, self.Pm4

    def calc_cycle_pressure(self,times=1):
        # 计算脉动压力
        pm = self.p_m
        if times == 1:
            # 第一次脉动压力
            self.p_cycle = (np.random.random()*0.1+0.11)*pm
        return self.p_cycle

    def calc_cycle_period(self,times=1,KR=30,KT=0.295):
        # 第一次脉冲压力信号出现时间
        # KR,KT 对于不同炸药，有不同的估值
        self.KR = KR
        self.KT = KT
        # 计算脉动时间
        if times == 1:
            # 第一次
            self.period = self.KT*(np.power(self.w,1/3)/np.power(1+0.1*self.h,5/6))
            # print("第一次气泡压力峰值出现时间: T1=",self.period)
        return round(self.period,2)

    def calc_cycle_second_period(self):
        pass

    def calc_first_buble_pres(self,current_time):
        # 第一次气泡压力出现
        if current_time >= self.period_left and current_time<=self.period_right:
            r = self.calc_pulse_r()
            pm = 9.03*np.power(self.w,1/3)/r
            print("压力:pm =",pm,"MPa")
            # pm = 0.20*self.p_m

            if current_time <= self.period:
                pt_max = pm * np.exp(-(self.period-current_time)/self.period_pulse)
                delta_pt = pt_max - old_pt
                pt = pt_max - delta_pt

            # elif current_time >= self.period:
            else:
                pt_max = pm*np.exp(-(current_time-self.period)/self.period_pulse)
                # delta_pt = pt_max - old_pt
                pt = pt_max - delta_pt
            return pt, self.p_m

    def calc_pulse_r(self):
        """计算膨胀半径"""
        # 冲击波辐射后爆炸生成物的剩余
        # E = WQ Q为炸药爆热
        # TNT 每1kg的爆热: 4.184e6 J/kg
        Energy = 0.41*self.w*self.Q
        # 静水压力
        P0 = 1000*9.8*self.h  # Pa
        r = 3*Energy/(4*np.pi*P0)
        r = np.power(r,1/3)
        vr = r / self.period_pulse
        # print("最大气泡面积: rmax = ",r,"m")
        # print("气泡收敛速度: vr = ",vr,"m/s")
        return r

    def calc_pulse_Max_r(self):
        KR = 30
        r = self.r_0*KR/np.power((1+0.1*self.h),1/3)
        return r

class GenerateExplosionnoise(object):
    def __init__(self,w,r_0,r,depth,q,Fs,num_pts,snr):
        """
        :param W: the weight of TNT /kg
        :param R_0: the radius of TNT /m
        :param R: the distance between the sensor and the explosion center /m
        :param H: the explosion depth /m
        :param Q: explosion hot /J/kg
        :return:
        """
        self.w = w
        self.r_0 = r_0
        self.r = r
        self.depth = depth
        self.q = q
        self.fs = Fs # sample freq
        self.dt = 1.0/Fs
        self.snr = snr
        self.sample_points = num_pts
        # super().__init__(self, w, r_0, r, depth, q)
        self.duration = []
        self.time = self.dt

        self.pressure = []
        self.pressure2 = []
        self.pressure3 = []
        self.pressure4 = []

        self.pm = 0
        self.pm2 = 0
        self.pm3 = 0
        self.pm4 = 0

    def generate_explosion_noise(self,order):
        """
        带噪声数据生成过程 --- 用于demo 4阶多项式系数回归
        :param order: 趋势项阶数
        :return:
        """
        self.time = self.dt

        PG = PressureSignalGenerator(self.w,self.r_0,self.r,self.depth,self.q)

        # step1 generate original signal
        for i in range(self.sample_points):
            pt, self.pm = PG.generator_pressure(self.time)
            pt2, self.pm2 = PG.generator_burble1_pressure(self.time)
            pt3, self.pm3 = PG.generator_burble2_pressure(self.time)
            pt4, self.pm4 = PG.generator_burble3_pressure(self.time)

            self.pressure.append(pt)
            self.pressure2.append(pt2)
            self.pressure3.append(pt3)
            self.pressure4.append(pt4)

            self.duration.append(self.time)
            self.time += self.dt

        signal_Sum = np.array(self.pressure) + \
                     np.array(self.pressure2) + \
                     np.array(self.pressure3) + \
                     np.array(self.pressure4) #(4000,)

        # self.duration = np.array(self.duration)
        Pure_signal = signal_Sum
        # step2 add white guassian noise
        signal_Sum,_ = add_white_gnoise(signal_Sum,self.snr)
        # step3 add 50hz (48~52) and 100,150,200 noise
        # signal_Sum,_ = add_50_Hz_noise(signal_Sum,self.snr, self.pm2, np.array(self.duration))
        signal_Sum,_ = add_50hz_to_200Hz_noise(signal_Sum,self.pm2,np.array(self.duration))
        # step4 retrending operation (4 order)
        retrending_noise, _,_ = retrending_signal_with_poly_type(self.pm,order,self.duration)
        # step5 get coeff_a,coeff_b
        period = np.arange(0,1,1/2000)
        _,_,coeff_a,coeff_b = Output_current_order(self.duration,retrending_noise,order)
        coeff_a = coeff_a[1:5]
        print(retrending_noise.shape)
        signal_Sum += retrending_noise
        # step5 add cable noise
        # cable_noise = Cable_noise_generator(self.duration,self.pm2)
        # signal_Sum += cable_noise

        return Pure_signal,signal_Sum,coeff_a,coeff_b

    def generate_explosion_noise_for_order_pred(self,max_order=10):
        """产生用于多项式趋势项阶数预测的数据"""



def label_generate(a,b):
    """label格式,前len(a)为多项式系数,最后一个为截距"""
    label = np.zeros((len(a)+1))
    a = np.array(a)
    label[0:len(a)] = a
    label[-1] = b
    # label = label.reshape(len(label),-1)
    # label = label.T
    return label

def poly_data1(time,coeffs,b):
    # for i in coeffs:
    poly = b+coeffs[1]*time+coeffs[2]*np.power(time,2)+ \
           coeffs[3]*np.power(time,3)+coeffs[4]*np.power(time,4)
    return poly

def poly_data2(time,coeffs,b):
    # for i in coeffs:
    poly = b+coeffs[0]*time+coeffs[1]*np.power(time,2)+ \
           coeffs[2]*np.power(time,3)+coeffs[3]*np.power(time,4)
    return poly

def Main_test():
    PG = PressureSignalGenerator(W, R_0, R, H, Q)
    # 采样频率
    Fs = 2048
    dt = 1 / 2048
    time = dt
    times = []
    pressure = []
    pressure2 = []
    pressure3 = []
    pressure4 = []
    pm, pm2, pm3, pm4 = 0, 0, 0, 0
    for i in range(2048):
        pt, pm = PG.generator_pressure(time)
        pt2, pm2 = PG.generator_burble1_pressure(time)
        pt3, pm3 = PG.generator_burble2_pressure(time)
        pt4, pm4 = PG.generator_burble3_pressure(time)
        # pt = pt + 0.0005*pm*(2*np.random.random()-1)

        pressure.append(pt)
        pressure2.append(pt2)
        pressure3.append(pt3)
        pressure4.append(pt4)

        times.append(time)
        time += dt

    # plt.subplot(321)
    # pl.plot(times,pressure,'-ro',markersize=2,label='Pressure/MPa')
    # pl.plot(times,pressure2,'-go',markersize=2,label='Pressure2/MPa')
    # pl.plot(times,pressure3, '-bo',markersize=2,label='Pressure3/MPa')
    # pl.plot(times,pressure4,'-ko',markersize=2,label='Pressure4/MPa')
    # plt.xlabel("$T$/s")
    # plt.ylabel("$p$/MPa")
    # # plt.tight_layout()
    # plt.legend()

    signal_Sum = np.array(pressure) + np.array(pressure2) + np.array(pressure3) + np.array(pressure4)
    plt.subplot(321)
    pl.plot(times, signal_Sum, '-ro', markersize=2, label='Signal/MPa')
    # pl.plot(times,pressure2,'-go',markersize=2,label='Pressure2/MPa')
    plt.xlabel("$T$/s")
    plt.ylabel("$p$/MPa")
    # plt.tight_layout()
    # plt.autoscale(tight=True)
    plt.legend()

    # 加上全频高斯
    snr = 10  # db 信噪比
    add_noise, noise = add_white_gnoise(signal_Sum, snr)
    plt.subplot(323)
    pl.plot(add_noise, '-ro', markersize=2, label='10db wgn')
    pl.plot(signal_Sum, '-g', markersize=2, label='original')
    plt.xlabel("$T$/s")
    plt.ylabel("$p$/MPa")
    # plt.autoscale(tight=True)
    plt.legend()

    plt.subplot(322)
    signal_Sum_f = fft_func(signal_Sum)
    pl.plot(signal_Sum_f, '-ro', markersize=2, label='Original signal')
    # plt.autoscale(tight=True)
    plt.legend()

    # 加上高斯白噪声
    plt.subplot(324)
    add_noise_f = fft_func(add_noise)
    pl.plot(add_noise_f, '-ro', markersize=2, label='add wgn signal')
    # plt.autoscale(tight=True)
    plt.legend()

    plt.subplot(325)
    # 加上50HZ工频
    # add_50hz, _ = add_50_Hz_noise(signal_Sum, snr, pm2, np.array(times)) # without white gaussian noise
    add_50hz, _ = add_50_Hz_noise(add_noise, snr, pm2, np.array(times))
    pl.plot(add_50hz, '-ro', markersize=2, label='50hz noise')
    pl.plot(signal_Sum, '-g', markersize=2, label='original')
    plt.xlabel("$T$/s")
    plt.ylabel("$p$/MPa")
    # plt.legend()
    # plt.autoscale(tight=True)
    plt.legend()

    # 加上50hz工频后的信号
    plt.subplot(326)
    # add_noise_trend = retrending_signal(add_noise,pm,times,Fs)
    # pl.plot(add_noise_trend,'-ro', markersize=2, label='retrending op')
    fft_signal = fft_func(add_50hz)
    pl.plot(fft_signal, '-ro', markersize=2, label='50hz+wgn')
    plt.xlabel("$T$/s")
    plt.ylabel("$p$/MPa")
    # plt.autoscale(tight=True)
    plt.legend()

    # 测试倍频_modification
    plt.figure()
    _,add_50_to_200,noise = add_50hz_to_200Hz_noise(add_noise,pm2,np.array(times))
    pl.plot(add_50_to_200,'--r',label='Nfreq')
    plt.legend()

    plt.figure()
    _,add_50_to_200,noise = add_50hz_to_200Hz_noise(add_noise,pm2,np.array(times))
    for i in range(4):
        pl.plot(noise[i])

    plt.figure()
    _,add_50_to_200,noise = add_50hz_to_200Hz_noise(add_noise,pm2,np.array(times))
    for i in range(4):
        fft_signal = fft_func(noise[i])
        pl.plot(fft_signal)


    # 趋势项 1-exp(-t) +- rand[0,1]
    plt.figure()
    plt.subplot(311)
    trend_item = 1 - np.exp(-np.array(times))
    poly_reg, poly_fit_function, a, b = Output_current_order(times, trend_item, order=3)
    retrend_signal, trend = retrending_signal_with_exp_type(add_50hz, pm, np.array(times), a, b)
    pl.plot(retrend_signal, '-ro', markersize=2, label='50hz+wgn+retrending')
    pl.plot(signal_Sum, '-g', markersize=2, label='original')
    pl.plot(trend, '--', markersize=1, label='trend')
    plt.title("Exponential trend item, coeffs=" + str(np.round(a, 4)) + ", intercept=" + str(np.round(b, 4)))
    plt.xlabel("$T$/s")
    plt.ylabel("$p$/MPa")
    # plt.autoscale(tight=True)
    plt.legend()

    # 趋势项 3阶
    # plt.subplot(312)
    #
    # poly, a, b = retrending_signal_with_poly_type(pm, 2, times)
    # pl.plot(add_50hz + poly, '-ro', markersize=2, label='50hz+wgn+retrending')
    # pl.plot(signal_Sum, '-g', markersize=2, label='original')
    # pl.plot(poly, '--', markersize=1, label='trend')
    # plt.title("3 order poly trend item, coeffs=" + str(np.round(a, 4)) + ", intercept=" + str(np.round(b, 4)))
    # plt.xlabel("$T$/s")
    # plt.ylabel("$p$/MPa")
    # # plt.autoscale(tight=True)
    # plt.legend()

    # 趋势项 4阶
    plt.subplot(313)
    poly, a, b = retrending_signal_with_poly_type(pm, 4, times)

    pl.plot(add_50hz + poly, '-ro', markersize=2, label='50hz+wgn+retrending')
    pl.plot(signal_Sum, '-g', markersize=2, label='original')
    pl.plot(poly, '--', markersize=1, label='trend')
    plt.title("4 order poly trend item, coeffs=" + str(np.round(a, 4)) + ", intercept=" + str(np.round(b, 4)))
    plt.legend()
    plt.xlabel("$T$/s")
    plt.ylabel("$p$/MPa")
    # plt.autoscale(tight=True)

    plt.subplots_adjust(top=0.962, bottom=0.058, left=0.038,
                        right=0.99, hspace=0.321, wspace=0.2)

    # 增加电缆噪声
    plt.figure()
    # 以第一次气泡脉动峰值压力作为基准
    Period = np.arange(0,1,1/Fs)


    _,_,coeffs_fit,inter = Output_current_order(Period,poly,order=4)
    print(coeffs_fit)
    print(inter)
    print(a)
    print(b)
    polydata = poly_data1(Period,coeffs_fit,inter) # 按照拟合系数
    polydata2 = poly_data2(Period,a,b)


    Cable_noise = Cable_noise_generator(times, pm2,50)
    signal_Sum_Sum = add_50hz + poly
    pl.plot(times, signal_Sum_Sum, '-r.', markersize=2, label='50hz+wgn+retrending+cable_noise')
    pl.plot(times, signal_Sum, '-g', markersize=2, label='original')
    pl.plot(times, poly, '--', markersize=1, label='trend')
    pl.plot(times,signal_Sum_Sum-poly,'-yo',markersize=2,label='trending op')
    pl.plot(times,signal_Sum_Sum-polydata,label='inverse op with polyfit ')
    pl.plot(times,signal_Sum_Sum-polydata2,label='inverse op without polyfit')
    plt.title("4 order poly trend item, coeffs=" + str(np.round(a, 4)) + ", intercept=" + str(np.round(b, 4)))
    plt.legend()
    plt.xlabel("$T$/s")
    plt.ylabel("$p$/MPa")
    # plt.show()

    # 去除工频
    # butterworth
    plt.figure()
    Explosionsignal = add_50hz + poly + Cable_noise
    plt.subplot(223)
    plt.title("Before Filter...")
    fft_signal_explosion_signal = fft_func(Explosionsignal)
    plt.plot(fft_signal_explosion_signal,"-r",label="original_signal")
    plt.grid()
    plt.legend()

    FilterMains = IIR2Filter(
        order=20,
        cutoff=[48,],
        filterType='lowpass',  # 带阻
        design='butter',  # butterworth
        # rp=0.001,
        fs=Fs
    )
    ExplosionsignalFilter = np.zeros((Explosionsignal.shape))
    for i in range(len(Explosionsignal)):
        ExplosionsignalFilter[i] = FilterMains.filter(Explosionsignal[i])

    FilterMains = IIR2Filter(
            order=20,
            cutoff=[52,],
            filterType='highpass',  # 带阻
            design='butter',  # butterworth
            # rp=0.001,
            fs=Fs
        )
    ExplosionsignalFilter2 = np.zeros((Explosionsignal.shape))
    for i in range(len(Explosionsignal)):
        ExplosionsignalFilter2[i] = FilterMains.filter(Explosionsignal[i])

    Denoising_signal = ExplosionsignalFilter+ExplosionsignalFilter2

    plt.figure()
    plt.plot(Explosionsignal,'b')
    plt.plot(Denoising_signal,'r')
    plt.title("XB Test")

    plt.figure()
    plt.subplot(311)
    plt.plot(np.abs(fft_func(Explosionsignal)))
    plt.subplot(312)
    plt.plot(np.abs(fft_func(Denoising_signal)))


    plt.figure()
    plt.subplot(221)
    plt.title("Before Filter...")
    plt.plot(times, Explosionsignal, "-r", label="original_signal")
    plt.grid()
    plt.legend()

    plt.subplot(222)
    plt.title("After Filter...")
    plt.plot(times, ExplosionsignalFilter, "-g", label="filter_signal")
    plt.grid()
    plt.legend()
    # plt.show()

    plt.subplot(224)
    plt.title("After Filter...")
    fft_signal_explosion_signal_filter = fft_func(ExplosionsignalFilter)
    plt.plot(fft_signal_explosion_signal_filter,"-r",label="original_signal")
    plt.grid()
    plt.legend()
    plt.show()

def Data_generator(data_path,label_path,capacity):
    ExploGen = GenerateExplosionnoise(W,R_0,R,H,Q,Fs,NUM_PTS,SNR)

    # columns = [0]*capacity
    # index = [0]*capacity
    # data_dict = {}
    # label_dict = {}

    df_data = pd.DataFrame()
    label_data = pd.DataFrame()

    for i in range(capacity):
        origin_signal,real_signal,a,b = ExploGen.generate_explosion_noise(order=4)
        real_signal = np.round(real_signal,4)
        label = label_generate(a,b)
        label = np.round(label,4)
        real_signal = pd.Series(real_signal)
        label = pd.Series(label)
        df_data["Data"+str(i)] = real_signal
        label_data["Label"+str(i)] = label

    df_data.to_csv(data_path,encoding="utf-8")
    label_data.to_csv(label_path,encoding="utf-8")

if __name__ == "__main__":
    # Main_test()
    # ExploGen = GenerateExplosionnoise(W,R_0,R,H,Q,Fs,NUM_PTS,SNR)
    # origin_signal,real_signal,a,b = ExploGen.generate_explosion_noise(order=4)
    # signal_plot(origin_signal,real_signal,a,b)
    # label = label_generate(a,b)
    # # real_signal = real_signal.reshape(-1,len(real_signal))
    # print("The dimension of data: {}".format(real_signal.shape))
    # print("The dimension of label: {}".format(label.shape))
    # print(label)
    # print("The coeffs of retrening item: a =",a)
    # print("The coeffs of retrening item: b =",b)

    # 试写入第一组数据


    path = "E:\Speakin_Workspace\Datasets_path_csv\explo_data.csv"
    path_label = "E:\Speakin_Workspace\Datasets_path_csv\label.csv"

    # data = pd.DataFrame(columns=["Data0"],data=real_signal)
    # data.to_csv(path,encoding='utf-8')
    # data = pd.DataFrame(columns=["Data0"],data=label)
    # data.to_csv(path_label,encoding='utf-8')

    # --- uncomment to generate data ---
    # Data_generator(path,path_label,capacity=2101)
    # --- --- --- --- --- --- --- --- --

    # df = pd.read_csv(path)
    # plt.plot(df["Data0"])
    # plt.plot(df["Data1"])
    # plt.plot(df["Data2"])
    # plt.plot(df["Data4"])
    # plt.show()

    Main_test()
















