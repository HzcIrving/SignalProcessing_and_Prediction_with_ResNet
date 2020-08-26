#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He
# email: 1910646@tongji.edu.cn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

import numpy as np

# ------------test own datasets ----------------
bos_house = datasets.load_boston()
bos_house_data = bos_house['data']
bos_house_target = bos_house['target']

print(bos_house_data.shape)
print(bos_house_target.shape)

x_train,x_test,y_train,y_test = train_test_split(bos_house_data,bos_house_target,random_state=41)

print(x_train.shape)
print(y_train.shape)

forest_reg = RandomForestRegressor(random_state=41)
forest_reg.fit(x_train,y_train)
y_pred = forest_reg.predict(x_test)

a = np.ones((579,13))
b = np.random.rand(579,13)
print(a.shape)
print(mean_squared_error(a,b))

#mean_squared_error
print('MSE为：',mean_squared_error(y_test,y_pred))
print('MSE为(直接计算)：',np.mean((y_test-y_pred)**2))

print('RMSE为：',np.sqrt(mean_squared_error(y_test,y_pred)))

#median_absolute_error
print(np.median(np.abs(y_test-y_pred)))
print(median_absolute_error(y_test,y_pred))

#mean_absolute_error
print(np.mean(np.abs(y_test-y_pred)))
print(mean_absolute_error(y_test,y_pred))

#mean_squared_log_error
print(mean_squared_log_error(y_test,y_pred))
print(np.mean((np.log(y_test+1)-np.log(y_pred+1))**2))

#explained_variance_score
print(explained_variance_score(y_test,y_pred))
print(1-np.var(y_test-y_pred)/np.var(y_test))

#r2_score
print(r2_score(y_test,y_pred))
print(1-(np.sum((y_test-y_pred)**2))/np.sum((y_test -np.mean(y_test))**2))