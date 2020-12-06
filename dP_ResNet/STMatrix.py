import pandas as pd
import numpy as np

from . import string2timestamp


class STMatrix(object):
    """docstring for STMatrix"""
    # timestamps[i] 2015101102
    def __init__(self, data, timestamps, T=1, CheckComplete=True):
        super(STMatrix, self).__init__()
        # 断言训练数据长度和对应日期长度一致
        assert len(data) == len(timestamps)

        self.data = data
        self.timestamps = timestamps
        self.T = T

        # 字符串的列表转化为时间戳（pd.Timestamp）的列表
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)

        # 检查是否缺失某天的数据
        if CheckComplete:
            self.check_complete()
        # 为时间戳加上索引 Timestamp('2014-04-01 00:00:00'): 0, Timestamp('2014-04-01 01:00:00'): 1, 。。。}
        self.make_index()

    def make_index(self):
        # get_index 初试化为一个空的字典
        self.get_index = dict()
        # 将一个可遍历的数据对象(self.pd_timestamps)组合为一个索引序列，同时列出数据和数据下标
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i
        # print(self.get_index)
        # self.get_index 为

    def check_complete(self):
        # 定义列表missing_timestamps保存确实的时间戳
        missing_timestamps = []
        # offset为一个周期的分钟数
        #offset = pd.DateOffset(minutes=24 * 60 // self.T)
        
        offset = pd.DateOffset(self.T)
        # 当前数据（pd.timestamps)时间戳的列表
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            # 计算前一个时间戳加上offset是否等于后一个时间戳，若不相等则说明缺失pd_timestamps[i-1] 到 pd_timestamps[i]的数据
            if pd_timestamps[i-1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i-1], pd_timestamps[i]))
            i += 1
        # 输出缺失的时间段
        for v in missing_timestamps:
            print(v)
        # 断言没有缺失信息
        assert len(missing_timestamps) == 0

    # timestamp是时间戳，get_index[timestamp]是从字典中获得对应的索引，
    # data[self.get_index[timestamp]] 是从数据中获的索引对应的数据。
    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    # 检查depends列表中的每一项的timestamps是否在get_index.key()中
    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_closeness=7, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=30):
        '''

        :param len_closeness: 邻近性依赖序列的长度 3
        :param len_trend: 趋势性依赖序列的长度 3
        :param TrendInterval: 趋势性片段的长度 7
        :param len_period: 周期性依赖序列的长度 3
        :param PeriodInterval:周期性片段的长度 1
        :return:
        '''
        # offset_week = pd.DateOffset(days=7)
        # offset_frame为每个周期分钟数，即各个周期的偏移量
        #offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        offset_frame = pd.DateOffset(days=self.T)
        # 邻近性组件的输入
        XC = []
        # 周期性组件的输入
        XP = []
        # 趋势性组件的输入
        XT = []

        Y = []
        timestamps_Y = []
        # XC的时间跨度是1
        # XP的时间跨度是7
        # XT的时间跨度是1
        depends = [range(1, len_closeness+1), # [1, 2, 3]
                   [PeriodInterval * self.T * j for j in range(1, len_period+1)], 
                   [TrendInterval * self.T * j for j in range(1, len_trend+1)]] 

        # depends = []
        # depends += range(1, len_closeness+1)
        # if len_period:
        #     depends += [PeriodInterval * self.T * j for j in range(1, len_period+1)]
        # if len_trend:
        #     depends += [TrendInterval * self.T * j for j in range(1, len_trend+1)]

        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)

        while i < len(self.pd_timestamps):
            # flag是为了标记 timestamps[i] 减去各时间跨度生成的依赖序列是否在pd.timestamps中
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                #print("?????:",self.pd_timestamps[i] - 30 * offset_frame)
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            # 不在时继续往后找
            if Flag is False:
                i += 1
                continue

            # 获的各依赖序列的数据
            # [ts[i] - 1*offset, ts[i] - 2*offset, ts[i]-3*offset]
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]

            # [ts[i] - 1*1*24*offset, ts[i] - 2*1*24*offset, ts[i] - 3*1*24*offset, ts[i] - 3*1*24*offset]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]

            # [ts[i] - 1*7*24*offset, ts[i] - 2*7*24*offset, ts[i] - 3*7*24*offset, ts[i] - 3*7*24*offset]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]

            # 获的各pd_timestamps[i]的数据，作为这些依赖序列的真实值
            y = self.get_matrix(self.pd_timestamps[i])

            # vstack将数组堆叠成一列，这里将各依赖序列添加到相应的输入列表
            if len_closeness > 0:
                XC.append(np.vstack(x_c))
            if len_period > 0:
                XP.append(np.vstack(x_p))
            if len_trend > 0:
                XT.append(np.vstack(x_t))
            # 将各依赖序列对应的真实值，添加到列表中
            Y.append(y)
            #各依赖序列对应的时间戳
            timestamps_Y.append(self.timestamps[i])
            i += 1
        # 将训练所用数据转化为列表
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print("邻近性 XC shape: ", XC.shape, "周期性 XP shape: ", XP.shape, "趋势性 XT shape: ", XT.shape, "真实值 Y shape:", Y.shape)
        # 返回
        return XC, XP, XT, Y, timestamps_Y


if __name__ == '__main__':
    pass
