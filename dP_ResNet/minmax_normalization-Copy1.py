"""
    最小最大规范化
"""
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # 保证随机数的可复现性


class MinMaxNormalization(object):
    '''MinMax Normalization(最小最大规范化,离差标准化) --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''
    #改成（0，1）规模化
    def __init__(self):
        pass
    #寻找最小最大值
        def print(self):
            print("最小值:", self._min, "最大值:", self._max)

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("最小值:", self._min, "最大值:", self._max)

    # 将数据映射到-1 到 1
    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        #X = X * 2. - 1.
        return X

    # 找最小最大值，并将数据映射到[-1 ,1]
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    # 将数据从[-1 ,1] 映射回原始值
    def inverse_transform(self, X):
        #X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X