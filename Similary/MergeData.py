import pandas as pd
import os
'''
当前只聚合了140多个电站数据
'''
data=pd.read_csv(os.path.join('./data/FinalData','similary_1.csv'),low_memory=False)
columns = data.columns.values
all_data=pd.DataFrame(columns=columns)
for file in os.listdir('./data/FinalData'):
    print(file)
    if file !='similary_all.csv':
        data=pd.read_csv(os.path.join('./data/FinalData',file))
        all_data=pd.concat([all_data,data])

all_data.to_csv('./data/FinalData/similary_all.csv',encoding='utf8',index=False)