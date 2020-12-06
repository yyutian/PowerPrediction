from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import math
from DP_ResNet.minmax_normalization import MinMaxNormalization
from Similary.metrics import mse,rmse

# 100棵决策树，停止的条件：样本个数为2，叶子节点个数为1
alg = RandomForestRegressor(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
#随机森林输入是整数
data=pd.read_csv('./data/FinalData/similary_all.csv')
data=data[['weatherCode','humidity','temperature','windSpeed','power_s0','power_s1','power_s2','dayPower']]
#data=data.astype('int')

df = data.copy()
len_train = math.ceil(len(df)*0.7)
values = df.values
true = values[len_train:,-1].copy()
m, n = values.shape
# 对每一个特征归一化
mmn = []
for j in range(5):
    mmn.append(MinMaxNormalization())
for i in range(4):
    values[:, i] = mmn[i].fit_transform(values[:, i].reshape(m, 1)).reshape(m)
values[:, 4:] = mmn[-1].fit_transform(values[:, 4:])
train_X = values[:len_train,:-1]
train_y = values[:len_train,-1]
test_X = values[len_train:,:-1]
test_y = values[len_train:,-1]

print('训练集输入格式：', train_X.shape, '训练集输出格式：', train_y.shape)  # （n_train*1*2），（n_train*1）
print('测试集输入格式：', test_X.shape, '测试集输出格式：', test_y.shape)
alg.fit(train_X,train_y)

train_predictions = alg.predict(train_X)
test_predictions = alg.predict(test_X)
len_test = int(len(test_X)*0.01)
pre = mmn[-1].inverse_transform(test_predictions)

#画出预测值和真实值
plt.title('rand-test')
plt.plot(true[:len_test],label='true')
plt.plot(pre[:len_test],label='pre')
plt.ylabel('power(KWH)')
plt.xlabel('n_sample')
plt.legend()
plt.savefig('./pic/rand_pre_test.png')
plt.show()

print('result:',mse(train_y,train_predictions),rmse(train_y,train_predictions))
print('result:',mse(test_y,test_predictions),rmse(test_y,test_predictions))
print('result:',mse(true,pre),rmse(true,pre))
