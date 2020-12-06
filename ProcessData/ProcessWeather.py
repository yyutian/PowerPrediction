import pandas as pd
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

weather=pd.read_csv('../datasets/jinan_weather_by_day.csv',encoding='gbk')

weather['weatherCode']=encoder.fit_transform(weather['weather'])
print(weather[:5])
weather.to_csv('../datasets/jinan_weather_by_day.csv',encoding='gbk',index=False)

#==============================确定最终外部输入信息==========================
import numpy as np
import h5py as h5
import os
import math
import datetime
import pandas as pd

data=pd.read_csv('../datasets/jinan_weather_by_day.csv',encoding='gbk')
times=data['time']

date=[]
for t in times:

    time=t.split('/')
    print(time)
    if int(time[1])<10:
        d = time[0] +'0'+time[1]
    else:
        d = time[0] + time[1]
    if int(time[2]) < 10:
        d=d+'0'+time[2]
    else:
        d = d + time[2]
    date.append(d)
print(date)
date=np.asarray(date,dtype=h5.special_dtype(vlen=str))
windSpeed=np.asarray(data['windSpeed'],dtype=np.float)
weather=np.asarray(data['weatherCode'],dtype=np.float)
temperature=np.asarray(data['temperature'],dtype=np.float)
humidity=np.asarray(data['humidity'],dtype=np.float)

file = h5.File('../datasets/JN_2017-2020_Meteorology_By_Day.h5', "w")
file.create_dataset("date", data=date)
file.create_dataset("windSpeed", data=windSpeed)
file.create_dataset("weather", data=weather)
file.create_dataset("humidity", data=humidity)
file.create_dataset("temperature", data=temperature)
file.close()
print("ok!")