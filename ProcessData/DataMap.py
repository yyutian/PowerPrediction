import numpy as np
import h5py as h5
import os
import math
import datetime
import pandas as pd

dataPath = "../datasets/FinalData/"
mapHeight = 24
mapWidth = 24
h5Path = "../datasets/JN_Fill_2017-2020_M{}x{}_Power.h5".format(mapHeight,mapWidth)
flowType = 1
T = 1
nb_day = 1155
longitudeSpan = [116.991, 117.429]
latitudeSpan = [36.8359, 37.5079]
longitude_Length = (longitudeSpan[1] - longitudeSpan[0]) / mapWidth
latitudeSpan_Length = (latitudeSpan[1] - latitudeSpan[0]) / mapHeight

dataSet = np.zeros((T * nb_day, flowType, mapHeight, mapWidth), dtype=np.float)
dataNum = np.zeros((T * nb_day, flowType, mapHeight, mapWidth), dtype=np.float)

startDate = '2017-01-01 22'
endDate = '2020-02-29 22'
startDate = datetime.datetime.strptime(startDate, '%Y-%m-%d %H')
endDate = datetime.datetime.strptime(endDate, '%Y-%m-%d %H')
delta = datetime.timedelta(days=1)


#  id,time, longitude, latitude
def location(longitude, latitude, time):
    '''
    :param longitude: 经度
    :param latitude: 维度
    :param time: eg. 2018-10-29 22
    :return: 在划分地图后，所在的行和列,周期
    '''

    # time = (int(time[9:10]) - 2) * 48 + int(time[11:13]) - 1
    time = datetime.datetime.strptime(time, '%Y-%m-%d')
    #print(time)
    time = (time - startDate).days
    width = (longitude - longitudeSpan[0]) / longitude_Length
    width = math.ceil(width) - 1
    height = (latitude - latitudeSpan[0]) / latitudeSpan_Length
    height = mapHeight - math.ceil(height)
    #print(time)
    return height, width, time


# taxi id, date time, longitude, latitude
def readFile(FileName):
    print("load", FileName)
    datas = pd.read_csv(FileName)
    # print(data)
    # lines = file.readlines()
    height = 0
    width = 0
    time = 0
    for i in range(len(datas)):
        # data = lines[i].split(',')
        longitude = float(datas.iloc[i]['lng'])
        latitude = float(datas.iloc[i]['lat'])
        if longitude <= longitudeSpan[0] or longitude >= longitudeSpan[1] or latitude <= latitudeSpan[0] or latitude >= \
                latitudeSpan[1]:
            continue
        height, width, time = location(longitude, latitude, datas.iloc[i]['time'])
        #print(time, height, width)
        # 所有电站发电量/容量
        power = float(datas.iloc[i]['dayPower/capacity'])
        dataSet[time][0][height][width] += power
        # 区域内电站+1
        dataNum[time][0][height][width] += 1


def loadAllFile(Path):
    for file in os.listdir(Path):
        readFile(dataPath + "/" + file)


loadAllFile(dataPath)
dataSet=dataSet/dataNum
dataSet[np.isnan(dataSet)] = 0

dataSet2 = np.zeros((nb_day * T), dtype=h5.special_dtype(vlen=str))
i = datetime.timedelta(days=1)
#把对应时间序列写入文件中
while i <= (endDate - startDate + datetime.timedelta(days=1)):
    listdata = startDate + i-datetime.timedelta(days=1)#在date1的基础上加i天
    date=(listdata).strftime('%Y%m%d')
    dataSet2[i.days-1] = date
    i += datetime.timedelta(days=1)#i++
file = h5.File(h5Path, "w")
file.create_dataset("data", data=dataSet)
file.create_dataset("date", data=dataSet2)
file.close()
print("ok!")