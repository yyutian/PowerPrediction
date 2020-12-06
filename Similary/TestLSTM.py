from Similary.TrainingLSTM import lstm,load_Data
import pandas as pd
import  numpy as np
import math
import matplotlib.pyplot as plt
import os
from Similary.metrics import mse,rmse
from keras.utils.vis_utils import plot_model    # 使用graphviz进行模型的可视化
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号

model=lstm()
model.load_weights('./model/lstm2.h5')


#os.environ["PATH"] += os.pathsep + 'D:/SoftWarePro/Graphviz2.38/bin/' # 添加graphviz的环境变量(os.pathsep + 'graphviz安装目录下的bin目录')

# 绘制模型图
#plot_model(model, to_file='pic/model.png', show_shapes=True)

url = './data/FinalData/similary_all.csv'
data=pd.read_csv(url)
#各数据集长度
train_len = math.ceil(len(data) * 0.6)
valid_len = math.ceil(len(data) * 0.2)
#保存真实值
true=data['dayPower'].iloc[:].values
true_train=data['dayPower'].iloc[:train_len+valid_len].values
true_test=data['dayPower'].iloc[train_len+valid_len:].values

train_X, train_y, valid_X, valid_y, test_X, test_y, scaler= load_Data(url)
print('训练集输入格式：', train_X.shape, '训练集输出格式：', train_y.shape)
print('验证集输入格式：', valid_X.shape, '验证集输出格式：', valid_y.shape)
print('测试集输入格式：', test_X.shape, '测试集输出格式：', test_y.shape)

score = model.evaluate(train_X, train_y, batch_size=train_y.shape[0] // 48, verbose=0)
print('训练数据集上评估 loss: %.6f        rmse (norm): %.6f ' % (score[0], score[1]))
score = model.evaluate(test_X, test_y, batch_size=test_y.shape[0] // 48, verbose=0)
print('测试数据集上评估 loss: %.6f        rmse (norm): %.6f ' % (score[0], score[1]))

test_X=test_X[:math.ceil(len(test_X)*0.05)]
test_y=test_y[:math.ceil(len(test_y)*0.05)]
true_test=true_test[:math.ceil(len(true_test)*0.05)]

testPredict = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#反归一化
inv_yhat = np.concatenate((test_X,testPredict), axis=1)
inv_yhat = scaler[-1].inverse_transform(inv_yhat[:,4:])

plt.figure(figsize=(16,8))
plt.title('测试集的预测结果')
plt.plot(true_test,label='true')
plt.plot(inv_yhat[:,-1],label='except')
plt.ylabel('日发电量(KWH)')
plt.xlabel('样本个数')
plt.legend()
plt.savefig('./pic/lstm_pre_test.png')
plt.show()
print('result:',mse(true_test,inv_yhat[:,-1]),rmse(true_test,inv_yhat[:,-1]))