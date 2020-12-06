'''
确保电站的capacity数据都是正确的
'''
import os
import pandas as pd

for file in os.listdir('../datasets/power'):
    print('load:','../datasets/power' + "/" + file)
    data=pd.read_csv('../datasets/power' + "/" + file,encoding='utf8')
    capacity=data['capacity']
    capacity=capacity.value_counts().index[0]
    data['capacity']=capacity
    data.to_csv('../datasets/power' + "/" + file,index=False,encoding='utf_8_sig')