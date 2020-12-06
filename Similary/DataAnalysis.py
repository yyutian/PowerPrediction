import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

data=pd.read_csv('./data/jinan_weather.csv',encoding='utf8',index_col='time')
power=pd.read_csv('../datasets/power/station149.csv',encoding='utf8',index_col='time')
fill_power=pd.read_csv('./data/FillData/fill_station149.csv',encoding='utf8',index_col='time')
#画出天气信息
def plot_Weather():
    values=data.values
    #对天气风向名称进行连续编码
    encoder = LabelEncoder()

    values[:,0] = encoder.fit_transform(values[:,0])
    values[:,-2] = encoder.fit_transform(values[:,-2])

    groups=[0,1,2,3,4]
    label=['weather','temperature','humidity','wind_direction','wind_speed']


    for i, group in enumerate(groups):
        i += 1
        plt.figure()
        plt.plot(values[:, group])
        plt.title(label[group], loc='right')
        plt.xlabel('天数')
        plt.ylabel(label[group])
        plt.savefig('./pic/'+label[group]+'.png')  # 保存图片
        plt.show()


fill_power.dayPower.plot()
plt.savefig('./pic/dayPower.png')  # 保存图片
plt.show()