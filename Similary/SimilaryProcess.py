'''
得到测试数据
'''
import math
import datetime
import pandas as pd
import numpy as np
import operator
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from .Method import Method


data=pd.read_csv('./data/jinan_weather6-19.csv',encoding='utf8')
weather_by_day=pd.read_csv('./data/jinan_weather_by_day.csv',encoding='utf8')
power=pd.read_csv('./data/FillData/fill_station281.csv',encoding='utf8')
power.dayPower = power.dayPower/power.capacity
power.drop(columns='capacity')
print(weather_by_day[:5])

data.time=data.time.str.split(' ').str[0]
time=data.time.drop_duplicates()

delete=[]
for t in power.time:
    if t not in time.values:
        delete.append(t)
print("缺失天:",delete)
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

method=Method(T_MAX,T_MIN,H_MAX,H_MIN,wcount)
uniformTemperatureList=method.temperatureUniform(Temperature)
uniformHumidityList=method.humidityUniform(humidity)
uniformWeatherList=method.weatherUniform(weatherCode)

#从第60天开始训练，测试
forecastTemperatureList=uniformTemperatureList[60:]
forecastHumidityList=uniformHumidityList[60:]
forecastWeatherList=uniformWeatherList[60:]
dateListResult = time.to_list()
dateListResult = dateListResult[60:]

final_data=[]

for i in range(len(dateListResult)):

    weather=weather_by_day[weather_by_day.time==dateListResult[i]]
    startDate = datetime.datetime.strptime(time[0], '%Y-%m-%d')
    endDate = datetime.datetime.strptime(dateListResult[i], '%Y-%m-%d')
    delta = datetime.timedelta(days=1)
    j = 0
    DistanceList = {}
    dic = {}
    while startDate < endDate:
        if datetime.datetime.strftime(startDate, '%Y-%m-%d') in delete:
            startDate = startDate + delta
            continue
        # print(startDate)
        weatherDistance = method.getDistance(uniformWeatherList[j], forecastWeatherList[i])
        humidityDistance = method.getDistance(uniformHumidityList[j], forecastHumidityList[i])
        temperatureDistance = method.getDistance(uniformTemperatureList[j], forecastTemperatureList[i])
        Distance = 0.6 * weatherDistance + 0.2 * humidityDistance + 0.2 * temperatureDistance  # 对三种因素采取不同的权重
        DistanceList[startDate.strftime('%Y-%m-%d')] = Distance  # 把每个相似日计算出的欧氏距离放入字典
        startDate = startDate + delta
        j = j + 1

    sortedDistance = sorted(DistanceList.items(), key=operator.itemgetter(1))
    print(dateListResult[i])
    '''
    print('欧氏距离为：', DistanceList)
    print('相似日期为：',sortedDistance[0][0],sortedDistance[1][0],sortedDistance[2][0],
                         sortedDistance[3][0],sortedDistance[4][0])#,sortedDistance[5][0],
                         #sortedDistance[6][0])
    '''
    dic['day_s0'] = sortedDistance[0][0]
    s_p0 = power[power['time'] == sortedDistance[0][0]]['dayPower'].values[0]
    dic['power_s0'] = s_p0

    dic['day_s1'] = sortedDistance[1][0]
    s_p1 = power[power['time'] == sortedDistance[1][0]]['dayPower'].values[0]
    dic['power_s1'] = s_p1

    dic['day_s2'] = sortedDistance[2][0]
    s_p2 = power[power['time'] == sortedDistance[2][0]]['dayPower'].values[0]
    dic['power_s2'] = s_p2

    dic['time'] = dateListResult[i]

    dic['weatherCode']=weather.weatherCode.values[0]
    dic['humidity']=weather.humidity.values[0]
    dic['temperature']=weather.temperature.values[0]
    dic['windSpeed']=weather.windSpeed.values[0]

    s_p = power[power['time'] == dateListResult[i]]['dayPower'].values[0]

    dic['dayPower'] = s_p
    final_data.append(dic)

final_data=pd.DataFrame(final_data)
final_data.to_csv('./data/TestData/similary_281.csv',index=False,encoding='utf_8_sig')