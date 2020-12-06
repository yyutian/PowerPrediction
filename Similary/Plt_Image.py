import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

data=pd.read_csv('./data/jinan_weather_by_day.csv',encoding='utf8')
fill_power=pd.read_csv('./data/FillData/fill_station149.csv',encoding='utf8')
power=pd.merge(data,fill_power,on='time',how='right')
def plot_windSpeed_dayPower():
    wind_3=[]
    wind_8=[]
    for i in range(len(power)):
        if power.windSpeed.iloc[i] == 3:
            wind_3.append(power.dayPower.iloc[i])
        elif power.windSpeed.iloc[i] == 8:
            wind_8.append(power.dayPower.iloc[i])

    l = len(wind_3) if len(wind_8) > len(wind_3) else len(wind_8)

    plt.plot(wind_3[:l], label='wind-1', color='b')
    plt.plot(wind_8[:l], label='wind-8', color='r')
    plt.ylabel('日发电量(kwh')
    plt.xlabel('index')
    plt.legend()
    plt.savefig('./pic/windSpeed_dayPower.png')
    plt.show()
plot_windSpeed_dayPower()
def plot_humidity_dayPower():
    '''
        温度对发电量的影响不容易分辨
    '''
    hum_20 = []
    hum_90 = []
    for i in range(len(power)):
        if power.humidity.iloc[i] >= 10 and power.humidity.iloc[i] <= 20:
            hum_20.append(power.dayPower.iloc[i])
        elif power.humidity.iloc[i] >= 80 and power.humidity.iloc[i] <= 90:
            hum_90.append(power.dayPower.iloc[i])

    l = len(hum_20) if len(hum_90) > len(hum_20) else len(hum_90)

    plt.plot(hum_20[:l], label='humidity10-20', color='b')
    plt.plot(hum_90[:l], label='humidity80-90', color='r')
    plt.ylabel('日发电量(kwh')
    plt.xlabel('index')
    plt.legend()
    plt.savefig('./pic/humidity_dayPower.png')
    plt.show()
plot_humidity_dayPower()
def plot_temperature_dayPower():
    '''
    温度对发电量的影响不容易分辨
    '''
    plt.figure(figsize=(10, 5))
    tem_10 = []
    tem_40 = []
    for i in range(len(power)):
        if power.temperature.iloc[i]<8:
            tem_10.append(power.dayPower.iloc[i])
        elif power.temperature.iloc[i]>=30 and power.temperature.iloc[i]<=40:
            tem_40.append(power.dayPower.iloc[i])

    l = len(tem_10) if len(tem_40)>len(tem_10)  else len(tem_40)
    tem_10=tem_10[:l]
    tem_40=tem_40[:l]
    plt.plot(tem_10,label='tmperature0-10',color='b')
    plt.plot(tem_40,label='tmperature30-40',color='r')
    plt.ylabel('日发电量(kwh')
    plt.xlabel('样本数')
    plt.legend()
    plt.savefig('./pic/temperature_dayPower.png')
    plt.show()
plot_temperature_dayPower()
def plot_weather_dayPower():
    sun=[]
    for i in range(len(data)):
        if  data['weather'].iloc[i] == '晴天':
            #print(data['weather'].iloc[i])
            sun.append(data['time'].iloc[i])

    yin=[]
    for i in range(len(data)):
        if '雨' in data['weather'].iloc[i]:
            #print(data['weather'].iloc[i])
            yin.append(data['time'].iloc[i])

    l = len(yin) if len(sun)>len(yin)  else len(sun)

    sun=sun[-l:]
    yin=yin[-l:]

    yin_power=[]
    sun_power=[]
    for x in fill_power.time:
        if x in yin:
            yin_power.append(fill_power[fill_power.time == x].dayPower.iloc[0])
    for x in fill_power.time:
        if x in sun:
            sun_power.append(fill_power[fill_power.time == x].dayPower.iloc[0])
    plt.plot(yin_power,color='b',label='雨天')
    plt.plot(sun_power,color='r',label='晴天')
    plt.ylabel('日发电量(kwh')
    plt.xlabel('样本数')
    plt.legend()
    plt.savefig('./pic/weather_dayPower.png')
    plt.show()
plot_weather_dayPower()