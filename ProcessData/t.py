import os
from dP_ResNet import load_stdata
from dP_ResNet.config import Config
import tensorly as tl
import numpy as np
import h5py as h5
from tensorly.decomposition import tucker,non_negative_tucker,non_negative_parafac
np.random.seed(1337)  # 保证随机数的可复现性

DATAPATH = Config().DATAPATH
mapHeight = 16
mapWidth = 16
filePath = "../datasets/JN_Fill_2017-2020_M{}x{}_Power.h5".format(mapHeight,mapWidth)

data, timestamps = load_stdata(filePath)
print(data[0][0])
data_=data.reshape(data.shape[0],data.shape[2],data.shape[3])
data__=data_.copy()
print("--*" * 10)
print(data__[0])