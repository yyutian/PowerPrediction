'''
这里使用数据集整合时留下的电站数据作为测试，这样可以更好的看出模型的优劣
'''
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from Similary import metrics
from Similary.TrainingLSTM import lstm,load_Data
import pandas as pd
import  numpy as np
import math
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

model=lstm()
model.load_weights('./model/lstm2.h5')
url = './data/FinalData/similary_all.csv'
url2 = './data/FinalData/similary_99.csv'
data=pd.read_csv(url)
data2=pd.read_csv(url2)
#保存真实值
true=data2['dayPower'].iloc[:].values

train_X, train_y, valid_X, valid_y, test_X, test_y, scaler= load_Data(url)

data2 = data2[['weatherCode', 'humidity', 'temperature', 'windSpeed', 'power_s0', 'power_s1', 'power_s2', 'dayPower']]
values = data2.values
for i in range(4):
    values[:, i] = scaler[i].transform(values[:, i].reshape(data2.shape[0], 1)).reshape(data2.shape[0])
values[:, 4:] = scaler[-1].transform(values[:, 4:])
all_X = values[:,:-1]
all_y = values[:,-1]
all_X=all_X.reshape(all_X.shape[0],1,all_X.shape[1])
score = model.evaluate(all_X, all_y, batch_size=all_y.shape[0] // 48, verbose=0)
print('数据集上评估 loss: %.6f        rmse (norm): %.6f ' % (score[0], score[1]))




testPredict = model.predict(all_X)
all_X = all_X.reshape((all_X.shape[0], all_X.shape[2]))
#反归一化
inv_yhat = np.concatenate((all_X[:, :],testPredict), axis=1)
inv_yhat = scaler[-1].inverse_transform(testPredict)
plt.title('99号电站数据集的预测结果')
plt.plot(true,label='true')
plt.plot(inv_yhat,label='except')
plt.ylabel('日发电量(KWH)')
plt.xlabel('样本个数')
plt.legend()
plt.savefig('./pic/lstm_pre_test_99.png')
plt.show()