# -*- coding: utf-8 -*-
import os
import pickle as pickle
import numpy as np
import h5py
from dP_ResNet import load_stdata, timestamp2vec
from dP_ResNet.minmax_normalization import MinMaxNormalization
from dP_ResNet.config import Config
from dP_ResNet.STMatrix import STMatrix
import random
np.random.seed(1337)  # 保证随机数的可复现性

# 训练数据路径
DATAPATH = Config().DATAPATH


# 加载气象数据
def load_meteorol(timeslots, fname=os.path.join(DATAPATH, 'JN_2017-2020_Meteorology_By_Day.h5')):
    '''
    timeslots: the predicted timeslots
    In real-world, we dont have the meteorol data in the predicted timeslot, instead, we use the meteoral at previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
    '''
    f = h5py.File(fname, 'r')
    Timeslot = f['date'][...]
    WindSpeed = f['windSpeed'][...]
    Weather = f['weather'][...]
    Temperature = f['temperature'][...]
    Humidity = f['humidity'][...]
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i
    WS = []  # WindSpeed
    WR = []  # Weather
    TE = []  # Temperature
    HM = []  # humidity
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])
        HM.append(Humidity[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)
    HM = np.asarray(HM)
    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())
    HM = 1. * (HM - HM.min()) / (HM.max() - HM.min())
    print("shape: ", WS.shape, WR.shape, TE.shape, HM.shape)

    # concatenate all these attributes
    merge_data = np.hstack([WR[:, None], WS[:, None], TE[:, None], HM[:, None]])

    # print('meger shape:', merge_data.shape)
    return merge_data


def load_data(T=1, nb_flow=1, len_closeness=3, len_period=3, len_trend=3,preprocess_name='preprocessing.pkl',
              meta_data=False, meteorol_data=False):
    '''
    :param T:周期数，一天划分为T个时间片段 1
    :param nb_flow:流的种类，这里只有日发电量
    :param len_closeness:邻近性依赖序列的长度 3
    :param len_period:周期性依赖序列的长度 0
    :param len_trend:趋势性依赖序列的长度 0
    :param len_test:测试数据的长度
    :return:
    '''
    assert (len_closeness + len_period + len_trend > 0)

    # 加载数据，data为城市中每个格子的发电量。timestamps为对应的时间
    DATAPATH = Config().DATAPATH

    FilePath = os.path.join(DATAPATH, 'JN_Fill_2017-2020_M16x16_Power_Decomposition.h5')
    #FilePath=os.path.join(DATAPATH, 'JN_Fill_2017-2020_M16x16_Power2.h5')
    print("加载文件：", FilePath)
    data, timestamps = load_stdata(FilePath)

    print(data.shape, timestamps.shape)

    # 验证数据的合法性
    data = data[:, :nb_flow]
    # 若流量 < 0 置为 0
    data[data < 0] = 0

    # 构建[numpy.array()]
    data_all = [data]
    timestamps_all = [timestamps]

    # 最小最大归一化方法将数据缩放到[-1,1]范围中
    mmn = MinMaxNormalization()
    # 找训练数据集中最小最大值
    mmn.fit(data)

    # 所有数据映射到[-1, 1]中，data_all_mmn 是所有数据映射到[-1 , 1]
    data_all_mmn = []
    for d in data_all:
        data_all_mmn.append(mmn.transform(d))
    # 序列化对象，将对象obj保存到文件file中去，复原数据时可以从中读取该对象将数据映射回去
    fpkl = open('preprocessing.pkl', 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    # zip（）打包为元组的列表 a=[1, 3, 5] b=[2, 4, 6]
    # list(zip(a,b)) = [(1,2), (3,4), (5,6)]
    # XC邻近性的输入， X趋势性周期性的输入， XT周期性的输入
    XC, XP, XT = [], [], []
    # 真实值，用来与预测值比较
    Y = []
    # 时间i
    timestamps_Y = []
    # data_all_mmn = [ 标准化的data]
    # timestamps_all = [ timestamps ]       -> zip() -> (标准化的data,timestamps)
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # 基于实例的数据集——>格式为（x，y）的序列，其中x是训练数据，y是数据对应日期。
        st = STMatrix(data, timestamps, T, CheckComplete=True)
        # _XC, _XP, _XT, _Y, _timestamps_Y, 分别对应邻近性，周期性，趋势性的依赖序列，当前时间的人群流量的真实值，对应的时间戳
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(len_closeness=len_closeness, len_period=len_period,
                                                             len_trend=len_trend)
        # 将各类训练数据加入到列表中
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y
    timestamps_Y = np.array(timestamps_Y)
    meta_feature = []
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_Y)
        meta_feature.append(time_feature)
    if meteorol_data:
        # load meteorol data 气象数据
        meteorol_feature = load_meteorol(timestamps_Y)
        meta_feature.append(meteorol_feature)

    meta_feature = np.hstack(meta_feature) if len(
        meta_feature) > 0 else np.asarray(meta_feature)
    metadata_dim = meta_feature.shape[1] if len(
        meta_feature.shape) > 1 else None
    # if metadata_dim < 1:
    # metadata_dim = None
    if meta_data and meteorol_data:
        print('time feature:', time_feature.shape, 'meteorol feature: '
              , meteorol_feature.shape, 'mete feature: ', meta_feature.shape)
    # 将各类数据分别堆砌为一列
    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("邻近性 XC shape: ", XC.shape, "周期性 XP shape: ", XP.shape, "趋势性 XT shape: ", XT.shape, "真实值 Y shape:", Y.shape)
    # 将各类数据划分为训练数据，测试数据

    len_train = int(Y.shape[0] * 0.8)

    index = [i for i in range(len(XC))]
    np.random.shuffle(index)
    #index = np.array(index)
    XC = XC[index]
    XP = XP[index]
    XT = XT[index]
    Y = Y[index]
    timestamps_Y = timestamps_Y[index]

    XC_train, XP_train, XT_train, Y_train = XC[:len_train], XP[:len_train], XT[:len_train], Y[:len_train]
    #XC_valid, XP_valid, XT_valid, Y_valid = XC[len_train:len_test], XP[len_train:len_test], XT[len_train:len_test], Y[len_train:len_test]
    XC_test, XP_test, XT_test, Y_test = XC[len_train:], XP[len_train:], XT[len_train:], Y[len_train:]
    timestamp_train,  timestamp_test = timestamps_Y[:len_train],  timestamps_Y[len_train:]

    X_train = []
    X_valid=[]
    X_test = []
    # 将依赖序列的长度，与依赖序列的列表组成元组，并加入到X_train X_test
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)

    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)

    if metadata_dim is not None:
        meta_feature_train,  meta_feature_test = meta_feature[:len_train],\
                                                meta_feature[len_train:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)

    print("\n训练数据形状为：")
    for _X in X_train:
        print(_X.shape)
    print("\n测试数据形状为：")
    for _X in X_test:
        print(_X.shape)
    return X_train, Y_train, X_test,\
           Y_test, mmn, metadata_dim, timestamp_train, timestamp_test
if __name__=='__main__':
    X_train, Y_train, X_test, \
    Y_test, mmn, metadata_dim, timestamp_train, timestamp_test=load_data\
        (T=1, nb_flow=1, len_closeness=3, len_period=3, len_trend=3,preprocess_name='preprocessing.pkl',
        meta_data=False, meteorol_data=False)
    print(timestamp_train)