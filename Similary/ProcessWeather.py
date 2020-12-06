import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data=pd.read_csv('./data/jinan_weather.csv',encoding='utf8')
data2=pd.read_csv('./data/jinan_weather_by_day.csv',encoding='utf8')

weather=[]

#由于光伏发电是在白天工作，这里我们取时间段为6-19
for i in range(len(data)):
    date=datetime.datetime.strptime(data['time'].iloc[i],'%Y-%m-%d %H:%M')
    if date.hour>6 and date.hour<=19:
        if data.iloc[i]['weather']=='晴朗':
            data.loc[i,'weather']='晴天'
        weather.append(data.iloc[i])
weather=pd.DataFrame(weather,columns=data.columns)
weather['weatherCode'] = encoder.fit_transform(weather.weather)
weather.to_csv('./data/jinan_weather6-19.csv',index=False,encoding='utf_8_sig')

if 'weatherCode' in data2.columns.values:
    data2.drop(columns=['weatherCode'],inplace=True)
print('=='*10,"每日天气",'=='*10)
print(data2[:5])
for i in range(len(data2)):
    if data2.iloc[i]['weather'] == '晴朗':
        data2.loc[i, 'weather'] = '晴天'
    if data2.iloc[i]['weather'] in weather.weather.values:
        #print(weather[weather.weather==data2.iloc[i]['weather']].iloc[0]['weatherCode'])
        data2.loc[i, 'weatherCode']=weather[weather.weather==data2.iloc[i]['weather']].iloc[0]['weatherCode']
data2.to_csv('./data/jinan_weather_by_day.csv',index=False,encoding='utf_8_sig')