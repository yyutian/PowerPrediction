import os
import pandas as pd
import datetime

'''
用来查询是否出现非法数据
'''
if __name__=='__main__':
    '''
    for f in os.listdir('./data/FillData'):
        data=pd.read_csv('./data/FillData/'+f)
        index=data[data.dayPower <= 0].index.values
        if len(index) !=0:
            print(data.iloc[index[0]].id)
       # id.append()
    '''

    for f in os.listdir('./data/FinalData'):
        data=pd.read_csv('./data/FinalData/'+f)
        #print(f)
        index = data[data.dayPower <= 0].index.values
        if len(index) != 0:
            print(data.iloc[index[0]].id)

    '''
    sta=pd.read_csv('./data/SPower/station133.csv')
    time=[]
    for t in sta.time:
        tmp=datetime.datetime.strptime(t,'%Y/%m/%d')
        tmp=datetime.datetime.strftime(tmp,'%Y-%m-%d')
        time.append(tmp)
    sta.time=time
    sta.to_csv('./data/SPower/station133.csv',index=False,encoding='utf_8_sig')
    print(sta[:5])
    '''