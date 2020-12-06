# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

score={}
#sid=[]
for file in os.listdir('../datasets/power'):
    data=pd.read_csv('../datasets/power'+'/'+file)
    sid=data.iloc[0]['id']
    score[sid]=(data['dayPower']<=0).sum()/len(data)

score=pd.Series(score)
score=score.sort_index()
score.plot(kind='line',label='ratio',use_index = True,style='--.',legend=True)
plt.title('各电站数据缺失值比例')
plt.ylabel('Nan_ratio(%)')
plt.xlabel('StationId')
plt.savefig('./picture/Nan_ration.png')
plt.show()