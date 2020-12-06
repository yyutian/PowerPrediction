from __future__ import print_function
import os
import platform

#数据集的绝对路径
class Config(object):
    def __init__(self):
        super(Config, self).__init__()
        # 获取训练数据的绝对路径
        DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')
        self.DATAPATH = DATAPATH



