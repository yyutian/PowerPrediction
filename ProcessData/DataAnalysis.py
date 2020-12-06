# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from dP_ResNet import load_stdata
from dP_ResNet.config import Config
import pandas as pd
np.random.seed(1337)  # 保证随机数的可复现性

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
DATAPATH = Config().DATAPATH
map_height, map_width = 16, 16
FilePath=os.path.join(DATAPATH, "JN_Fill_2017-2020_M{}x{}_Power.h5".format(map_height,map_width))
data, timestamps = load_stdata(FilePath)
FilePath=os.path.join(DATAPATH,"JN_Fill_2017-2020_M{}x{}_Power_Decomposition.h5".format(map_height,map_width))
data_dec, timestamps_dec = load_stdata(FilePath)

d=[]
t=[]
d_all={}
t_all={}
count=0
for i in range(16):
    for j in range(16):
        d=[]
        for l in range(1155):
            d.append(data_dec[l][0][i][j])
            t.append(timestamps[l])
        d_all[count]=d
        count+=1
T=7
j=0
d_=[]
t_=[]
while j*T<len(d_all[98]):
    d_.append(d_all[98][j*T])#周日
    t_.append(t[j*T])
    j=j+1
len(d_)
def closses(d_all):
    #邻近性
    plt.plot(d_all[22][21:29], '--.', label="单位容量发电量")
        #plt.yticks(range(1,10,1))
    plt.ylabel('单位容量发电量(kwh)')
    plt.xticks(range(8),t[21:29],rotation=45)
    plt.yticks(range(4))
    plt.legend()
    plt.savefig('./picture/closses.png')
    plt.show()

#画图-趋势性
def trend(d_):
    plt.scatter(list(range(14)),d_[:14],label='单位容量发电量')
    plt.ylabel('单位容量发电量(kwh)')
    plt.xticks(range(14),t_[:14],rotation=45)
    plt.legend()
    plt.savefig('./picture/trend.png')
    plt.show()
    #可以看出发电量呈上升趋势

#周期性
def period(d_all):
    for k in range(len(d_all)):
        if(k>50):
            continue
        plt.plot(d_all[k], '--.', label="单位容量发电量")
        plt.ylabel('单位容量发电量(kwh)')
        plt.xlabel('天数')
        #plt.yticks(range(1,10,1))
        plt.legend()
        #plt.savefig('./picture/perid_'+str(k)+'.png')
        plt.show()
closses(d_all)
trend(d_)
#period(d_all)