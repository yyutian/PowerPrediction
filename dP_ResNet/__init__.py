import h5py
import numpy as np
from copy import copy
import time
# from temporal_contrast_normalization import TemporalConstrastNormalization
# from personal_temporal_contrast_normalization import PersonalTemporalConstrastNormalization
from .minmax_normalization import MinMaxNormalization
import pandas as pd
from datetime import datetime

def load_stdata(fname):
    '''
    :param fname:文件路径（字符串）
    :return:从文件中读取的数据与对应时间戳（numpy.ndarray())
    '''
    f = h5py.File(fname, 'r')
    data = f['data'][...]
    timestamps = f['date'][...]

    # print (type(data), type(timestamps))

    f.close()
    return data, timestamps


def stat(fname):
    def get_nb_timeslot(f):
        s = f['date'][0]
        e = f['date'][-1]
        year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
        ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
        te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        nb_timeslot = (time.mktime(te) - time.mktime(ts)) / (0.5 * 3600) + 48
        ts_str, te_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
        return nb_timeslot, ts_str, te_str

    with h5py.File(fname) as f:
        nb_timeslot, ts_str, te_str = get_nb_timeslot(f)
        nb_day = int(nb_timeslot / 48)
        mmax = f['data'].value.max()
        mmin = f['data'].value.min()
        stat = '=' * 5 + 'stat' + '=' * 5 + '\n' + \
               'data shape: %s\n' % str(f['data'].shape) + \
               '# of days: %i, from %s to %s\n' % (nb_day, ts_str, te_str) + \
               '# of timeslots: %i\n' % int(nb_timeslot) + \
               '# of timeslots (available): %i\n' % f['date'].shape[0] + \
               'missing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
               'max: %.3f, min: %.3f\n' % (mmax, mmin) + \
               '=' * 5 + 'stat' + '=' * 5
        print(stat)


# -----------------------------------------------------------------------------

# 字符串转化为时间戳
# strings[i] 格式为2015101101
def string2timestamp(strings, T=1):
    timestamps = []
    # slot表示该天的第几个时间片
    # 每周期多少小时， 小时则为slot * time_per_slot
    # time_per_slot = 24.0 / T

    # 60.0 * time_per_slot为每周期多少分钟
    # num_per_T 每小时多少周期， slot % num_per_T剩余的周期
    # slot % num_per_T * （60.0 * time_per_slot） 分钟数
    # num_per_T = T // 24
    # 因一天只划分了一个时隙，所以不需要上述操作
    # t=strings
    for t in strings:
        # print(t)
        year, month, day = int(t[:4]), int(t[4:6]), int(t[6:8])  # , int(t[8:])-1
        timestamps.append(pd.Timestamp(datetime(year, month, day)))
        # , hour=int(slot * time_per_slot), minute=(slot % num_per_T) * int(60.0 * time_per_slot))))

    return timestamps


def timestamp2string(timestamps, T=48):
    # timestamps = timestamp_str_new(timestamps)
    num_per_T = T // 24
    return ["%s%02i" % (ts.strftime('%Y%m%d'),
                        int(1 + ts.to_datetime().hour * num_per_T + ts.to_datetime().minute / (60 // num_per_T))) for ts
            in timestamps]
    # int(1+ts.to_datetime().hour*2+ts.to_datetime().minute/30)) for ts in timestamps]


# -----------------------------------------------------------------------------


def timestamp2vec(timestamps):
    # tm_wday 范围是 [0, 6], 星期一是0，这里vec存的是0-6带表该天是星期几
    # vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
    vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]  # python2
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]  # [0, 0, 0, 0, 0, 0, 0]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    # 转化为ndarray类型并返回，返回的结果是一个列表，其中每个元素，也是列表
    # 每个元素类似[0, 0, 1, 0, 0, 0, 0],其中0-6，中只有一个1,标识今天星期几，7是0或1,分别带表周末，工作日
    return np.asarray(ret)


def remove_incomplete_days(data, timestamps, T=1):
    '''
    删除删除48个时间片段的天
    :param data: 城市中各区域的流入流出数据
    :param timestamps: 数据对应的时间
    :param T: 一遍被分层多少个周期
    :return: 去除非法数据后的data，与timestamps，都为numpy.array()类型。
    '''
    days = []  # 数据完整的天
    days_incomplete = []  # 数据不完整的天
    # timestamps[8:0] 是该时间片是这一天的第几段时间
    # timestamps[8:0] 不为1时直接丢弃，为 1 时看索引加上T - 1是否为 T 若为T则该天的数据完整，否则不完整
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("数据不完整的天为: ", days_incomplete)
    # 用数据合法的天构建一个set
    days = set(days)
    idx = []
    # enumerate（） 将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
    # a         0，a
    # b   ---》 1，b
    # c         2，c
    # 这里是获取合法天对应的序号
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)
    # 这里data是numpy里的Ndarray
    # data[inx] 是获取idx中为序号的项，即合法天数对应的数据
    data = data[idx]
    # 获取合法天的时间戳
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps
