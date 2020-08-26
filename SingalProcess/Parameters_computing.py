#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

import matplotlib.pyplot as plt
import pylab as pl
import scipy
import numpy as np
import math

plt.style.use(['science','no-latex'])


PHO = 1.52 # g/cm3
W = 1000.0 # 炸药重量
R_0 = round((np.power(1000000*3/(4*np.pi)*1/PHO,1/3))*0.01,2) # 炸药半径
print(R_0)
R = 8 # 压传与爆炸中心的距离
print(R/R_0)
# THETA = 1.0 # 冲击波衰减时间
H = 20 # 深度

# step1 : 计算theta
A = math.pow(W, 1 / 3) / R
theta = round(10e-5*math.pow(W,1/3)*math.pow(A,-0.24),6)

# step2: 计算峰值压力
Pm = 54.2*(math.pow(A,1.13))

# step3: 计算第一个水泡出现时间
# tb1 = 0.295*(np.power(W,1/3)/np.power(1+0.1*H,5/6))
tb1 = 985*np.power(W,1/3)/np.power((H+10.33),5/6)/1000

# step4: 计算压力系数
pt = []
pt_t = 0
T = np.arange(0,2,0.0001)
print(T.shape)
# t.reshape()


# 计算第一次水泡的最大Rmax
Rmax = 1.6*np.power(W/(H+10.33),1/3)
Pmax = 7.24*np.power(W,1/3)/R

for i,t in enumerate(T):
    if t < theta:
        pt_t = Pm*np.exp(-t/theta)
    elif t >= theta and t < tb1-0.11*tb1:
        pt_t = 0.368*Pm*theta/t
        pt_t = pt_t*(1-np.power((t/tb1),1.5))
    elif t >= tb1 - 0.11*tb1 and t < tb1 + 0.11*tb1:
        pt.append(pt_t)


# print(np.array(pt).shape)
plt.plot(np.array(pt),'-ro',markersize=2,label='Pressure/MPa')
plt.xlabel("$T$/s")
plt.ylabel("$p$/MPa")
plt.legend()
plt.show()

print("炸药半径:R0 =",R_0,"m")
print("冲击波衰减时间:theta = ",theta,"s")
print("峰值压力值:Pm = ",Pm,"MPa")
print("第一个水泡出现时间点：Pb1 = ",tb1,"s")
