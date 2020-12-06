import pandas as pd 
import numpy as np
import os

from dP_ResNet.config import Config
import datetime
import h5py as h5

DATAPATH=Config().DATAPATH
#删除数据全为-1的电站
def DropAll0File(FileName):
    print("load", FileName)
    data = pd.read_csv(FileName)
    le=len(data)
    i=0
    count=0
    count0=0
    while i<le:
        if(data.iloc[i]['dayPower']==-1):   #loc函数：通过行索引 "Index" 中的具体值来取行数据（如取"Index"为"A"的行）,iloc函数：通过行号来取行数据
            count +=1
        if (data.iloc[i]['dayPower'] == 0):
            count0 += 1
        i+=1
    no=(data['dayPower']==-1).sum()
    no+=(data['dayPower']==0).sum()
    if(count==le or count0==le or no>600):
        print("删除：",FileName)
        os.remove(FileName)
'''
#=======================start==========================
for file in os.listdir('../datasets/power'):
    DropAll0File('../datasets/power'+ "/" + file)
#=======================end==========================
'''
#确保capacity不是自己填充的
'''
由于当时取数据时使用的一个大于0的实数填充的无数据的电站容量
所以，如下方法中取众数来确保capacity

有数据可知，缺失的数据没有得到的多所以可用众数

也可再取数据时用小于0的实数填充
这样会更简单
'''
'''
#=======================start==========================
m=0
for file in os.listdir('../datasets/power'):
    print('load:','../datasets/power' + "/" + file)
    data=pd.read_csv('../datasets/power' + "/" + file,encoding='gbk')
    capacity=data['capacity']
    capacity=capacity.value_counts().index[0]
    data['capacity']=capacity
    data.to_csv('../datasets/power' + "/" + file,index=False)
#=======================end==========================

filePath = os.path.join(DATAPATH,'power')
#所取数据的起始时间
startDate='2017-01-01 22'
startDate = datetime.datetime.strptime(startDate, '%Y-%m-%d %H')
#所取数据的结束时间
endDate='2020-02-29 22'
endDate=datetime.datetime.strptime(endDate,'%Y-%m-%d %H')

def FillData(FileName):
    print("load", FileName)
    data = pd.read_csv(FileName)
    for le in range(len(data)):
        if (data.iloc[le]['dayPower'] == -1 or data.iloc[le]['dayPower']==0):
            date=data.iloc[le]['time']
            dayPower=0
            print(le,':',date,':',dayPower)
            delta=datetime.datetime.strptime(date,'%Y-%m-%d %H')
            if delta==endDate:
                break
            while dayPower<=0:
                date=date[:2]+str(int(date[2:4])+1)+date[4:]
                delta=datetime.datetime.strptime(date,'%Y-%m-%d %H')
                if delta>=endDate:
                    break
                dayPower=float(data[data['time']==date]['dayPower'])
                print(le,':',date,':',dayPower,'      ',delta)
                if dayPower<=0:
                    dayPower=0
            data.loc[le,'dayPower']=dayPower
    for le2 in range(len(data)):
        if (data.iloc[le2]['dayPower'] == -1 or data.iloc[le2]['dayPower']==0):
            date=data.iloc[le2]['time']
            dayPower=0
            print(le2,':',date,':',dayPower)
            delta=datetime.datetime.strptime(date,'%Y-%m-%d %H')
            if delta==endDate:
                break
            while dayPower<=0:
                date=date[:2]+str(int(date[2:4])-1)+date[4:]
                delta=datetime.datetime.strptime(date,'%Y-%m-%d %H')
                if delta<=startDate:
                    break
                dayPower=float(data[data['time']==date]['dayPower'])
                print(le2,':',date,':',dayPower,'      ',delta)
                if dayPower<=0:
                    dayPower=0
            data.loc[le2,'dayPower']=dayPower
    outPath='../datasets/FillData/'+'station'+str(data.loc[0,'id'])+'.csv'
    print(outPath)
    data.to_csv(outPath,index=False)
#=======================start==========================
for file in os.listdir(filePath):
    FillData(filePath + "/" + file)
#=======================end==========================

# 直到前后项大于0时，停止寻找，然后平均值填充数据
def FillData(FileName):
    print("load", FileName)
    data = pd.read_csv(FileName)
    le = len(data) - 1
    # 前后项平均值填充
    while le >= 0:
        if (data.iloc[le]['dayPower'] == -1 or data.iloc[le]['dayPower'] == 0):
            tmp1 = 0.
            tmp2 = 0.
            i = 1
            while tmp1 <= 0:
                if ((le - i) >= 0):
                    tmp1 = float(data.iloc[le - i]['dayPower'])
                else:
                    break
                i = i + 1

            i = 1
            while tmp2 <= 0:
                if ((le + i) < len(data)):
                    tmp2 = float(data.iloc[le + i]['dayPower'])
                else:
                    break
                i = i + 1
            if tmp1 <= 0:
                tmp1 = 0
            if tmp2 <= 0:
                tmp2 = 0
            print(le, ':', tmp1, tmp2)
            data.loc[le, 'dayPower'] = (tmp1 + tmp2) / 2.
            # print(data.iloc[le])
        else:
            # 确保capacity是属于这一电站
            if data.iloc[le]['capacity'] > 0:
                capacity = data.iloc[le]['capacity']
                data['capacity'] = capacity
        le = le - 1
    id = int(data.iloc[0]['id'])
    dayPower = data['dayPower']
    dayPower[dayPower < 0] = 0
    data['dayPower'] = dayPower

    outPath = os.path.join(DATAPATH, 'FillData', 'station' + str(id) + '.csv')
    print(outPath)
    # print(data)
    data.to_csv(outPath, index=False)
#=======================start==========================
for file in os.listdir('../datasets/FillData'):
        FillData('../datasets/FillData' + "/" + file)
#=======================end==========================
'''
#按照前后项平均值填充后，再进行数据的聚合
def merge_station_positon_power():
    # 聚合电站与经纬度信息数据
    station = pd.read_csv('../datasets/station.csv', encoding='GB2312')
    regionId = station['changZhanId']
    for id in regionId:
        print(id)
        no = station[station['changZhanId'] == id].index.values[0]
        # 后项填充数据后聚合
        # station_power=pd.read_csv('../datasets/BFillData/station'+str(id)+'.csv')
        # 去0值后聚合
        try:
            station_power = pd.read_csv('../Similary/data/FillData/fill_station' + str(id) + '.csv')
        except:
            print(id, '不存在')
            continue

        station_power['lat'] = station.iloc[no]['latitude']
        station_power['lng'] = station.iloc[no]['longitude']
        outPath = os.path.join(DATAPATH,'MergeData','merge_station'+str(id) + '.csv')
        station_power.to_csv(outPath, index=False)

def dayPower_capacity(FileName):
    print("load", FileName)
    data = pd.read_csv(FileName)
    data['dayPower/capacity']=data['dayPower']/data['capacity']
    data=data[['id','time','dayPower/capacity','lat','lng']]
    data.to_csv(os.path.join(DATAPATH,'FinalData','final_station'+str(data.iloc[0]['id'])+'.csv'),index=False)

#=======================start==========================
merge_station_positon_power()

for file in os.listdir(os.path.join(DATAPATH,'MergeData')):
    dayPower_capacity(os.path.join(DATAPATH,'MergeData')+'/'+file)
print('ok')
#=======================end============================

