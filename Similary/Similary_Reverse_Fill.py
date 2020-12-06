'''
这个相较于SimilaryFill是把数据逆序以后再填充
因为SimilaryFill是基于已知历史数据进行填充，如果历史数据中就存在缺失值，无法处理这些历史数据
这里把数据逆序了以后，就相当于对上述的历史数据进行处理
'''
import math
import os
import datetime
import pandas as pd
import numpy as np
import operator
# from .Method import Method
import Method
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

data = pd.read_csv('./data/jinan_weather6-19.csv', encoding='utf8')
def load_data(url):
    print(url)
    power = pd.read_csv(url, encoding='utf8')
    id=power.id.iloc[0]

    #以便划分数据
    data.time=data.time.str.split(' ').str[0]
    power.time=power.time.str.split(' ').str[0]
    power.to_csv(url, encoding='utf8',index=False)
    time=data.time.drop_duplicates().to_list()
    #找出天气缺失天
    delete=[]
    for t in power.time:
        if t not in time:

            delete.append(t)
    print(delete)
    #按一天为索引把数据存储在一个字典里，其中一天包含24小时的数据
    weatherList={}
    tmpList=[data.iloc[0].values]
    j=0
    for i in range(1,len(data)):
        if(i==len(data)-1):
            tmpList.append(data.iloc[i].values)
            weatherList[j]=tmpList
            tmpList=[]
        if(data.iloc[i].time!=data.iloc[i-1].time):
            weatherList[j]=tmpList
            tmpList=[]
            j+=1
            tmpList.append(data.iloc[i].values)
        else:
            tmpList.append(data.iloc[i].values)

    Temperature=[]
    humidity=[]
    weatherCode=[]

    for key,values in weatherList.items():
        tem=[]
        hum=[]
        wea=[]
        for j in values:
            tem.append(j[2])
            hum.append(j[3])
            wea.append(j[6])
        Temperature.append(tem)
        humidity.append(hum)
        weatherCode.append(wea)

    weatherCode=np.asarray(weatherCode)
    Temperature=np.asarray(Temperature)
    humidity=np.asarray(humidity)

    T_MAX=np.hstack(Temperature).max()
    T_MIN=np.hstack(Temperature).min()
    H_MAX=np.hstack(humidity).max()
    H_MIN=np.hstack(humidity).min()
    wcount=data['weatherCode'].value_counts().count()

    method=Method.Method(T_MAX,T_MIN,H_MAX,H_MIN,wcount)

    #归一化处理
    uniformTemperatureList=method.temperatureUniform(Temperature)
    uniformHumidityList=method.humidityUniform(humidity)
    uniformWeatherList=method.weatherUniform(weatherCode)

    uniformTemperatureList.reverse()
    uniformHumidityList.reverse()
    uniformWeatherList.reverse()
    #查看数据的从第48个数据开始时错误的，这里可以随意设置
    forecastTemperatureList=uniformTemperatureList[10:]
    forecastHumidityList=uniformHumidityList[10:]
    forecastWeatherList=uniformWeatherList[10:]
    time.reverse()
    dateListResult = time

    dateListResult = dateListResult[10:]

    #相似日填充
    for i in range(len(dateListResult)):
        startDate = datetime.datetime.strptime(time[0], '%Y-%m-%d')
        endDate = datetime.datetime.strptime(dateListResult[i], '%Y-%m-%d')
        delta = datetime.timedelta(days=1)
        j = 0
        DistanceList = {}
        dic = {}
        while startDate > endDate:
            if datetime.datetime.strftime(startDate, '%Y-%m-%d') in delete:
                startDate = startDate - delta
                continue
            # print(startDate)
            weatherDistance = method.getDistance(uniformWeatherList[j], forecastWeatherList[i])
            humidityDistance = method.getDistance(uniformHumidityList[j], forecastHumidityList[i])
            temperatureDistance = method.getDistance(uniformTemperatureList[j], forecastTemperatureList[i])
            Distance = 0.6 * weatherDistance + 0.2 * humidityDistance + 0.2 * temperatureDistance  # 对三种因素采取不同的权重
            DistanceList[startDate.strftime('%Y-%m-%d')] = Distance  # 把每个相似日计算出的欧氏距离放入字典
            startDate = startDate - delta
            j = j + 1

        sortedDistance = sorted(DistanceList.items(), key=operator.itemgetter(1))
        #print(dateListResult[i])
        '''
        print('欧氏距离为：', DistanceList)
        print('相似日期为：',sortedDistance[0][0],sortedDistance[1][0],sortedDistance[2][0],
                             sortedDistance[3][0],sortedDistance[4][0])#,sortedDistance[5][0],
                             sortedDistance[6][0])
        '''
        s_p0 = power[power['time'] == sortedDistance[0][0]]['dayPower'].values[0]
        s_p1 = power[power['time'] == sortedDistance[1][0]]['dayPower'].values[0]
        s_p2 = power[power['time'] == sortedDistance[2][0]]['dayPower'].values[0]
        s_p3 = power[power['time'] == sortedDistance[3][0]]['dayPower'].values[0]
        s_p4 = power[power['time'] == sortedDistance[4][0]]['dayPower'].values[0]
        dic['time'] = dateListResult[i]
        s_p = power[power['time'] == dateListResult[i]]['dayPower'].values[0]
        if (s_p == 0 or s_p == -1):
            if (s_p4 != 0 and s_p4 != -1):
                s_p = s_p4
            if (s_p3 != 0 and s_p3 != -1):
                s_p = s_p3
            if (s_p2 != 0 and s_p2 != -1):
                s_p = s_p2
            if (s_p1 != 0 and s_p1 != -1):
                s_p = s_p1
            if (s_p0 != 0 and s_p0 != -1):
                s_p = s_p0
            index = power[power['time'] == dateListResult[i]]['dayPower'].index[0]
            power.loc[index, 'dayPower'] = s_p
    power.to_csv('./data/FillData/fill_station'+str(id)+'.csv',index=False,encoding='utf8')

if __name__=='__main__':
    import re
    for file in os.listdir('./data/FillData'):
        id=int(re.findall(r"\d+\d*",file)[0])   #找出数字
        print(id)
        if id <= 151:
            #if file == 'fill_station151.csv':
            load_data(os.path.join('./data/FillData',file))