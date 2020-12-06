# -*- coding: utf-8 -*-
import os
from dP_ResNet import load_stdata
from dP_ResNet.config import Config
import tensorly as tl
import numpy as np
import h5py as h5
from tensorly.decomposition import tucker,non_negative_tucker,non_negative_parafac
np.random.seed(1337)  # 保证随机数的可复现性

DATAPATH = Config().DATAPATH
#mapHeight = 24
#mapWidth = 24
mapHeight = 16
mapWidth = 16
filePath = "../datasets/JN_Fill_2017-2020_M{}x{}_Power.h5".format(mapHeight,mapWidth)

data, timestamps = load_stdata(filePath)
print()
data_=data.reshape(data.shape[0],data.shape[2],data.shape[3])
data__=data_.copy()

#使用tucker分解填充无数据的地方，这里填充的值只保存了原有数据的一些临近性、趋势性、周期性，对预测结果并不能作为考究点
X = tl.tensor(data__.reshape(data__.shape[0],data__.shape[1],data__.shape[2]))
factors = non_negative_tucker(X, rank=3)
full_tensor=tl.tucker_to_tensor(factors)
#factors = non_negative_parafac(X, rank=3)
#full_tensor=tl.kruskal_to_tensor(factors)
data__=full_tensor.reshape(full_tensor.shape[0],1,full_tensor.shape[1],full_tensor.shape[2])

outPath="../datasets/JN_Fill_2017-2020_M{}x{}_Power_Decomposition.h5".format(mapHeight,mapWidth)
file = h5.File(outPath, "w")
file.create_dataset("data", data=data__)
file.create_dataset("date", data=timestamps)