import pandas as pd
import matplotlib.pyplot as plt
import math
from PowerPrediction.Similary.metrics import  mse,rmse
from sklearn.linear_model import LinearRegression
from PowerPrediction.dP_ResNet.minmax_normalization import MinMaxNormalization
# 训练集交叉验证，得到平均值
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold

# 初始化现行回归算法
alg = LinearRegression()

data = pd.read_csv('./data/FinalData/similary_all.csv')
#test_data=pd.read_csv('./data/TestData/similary_150.csv')
data = data[['weatherCode','humidity','temperature','windSpeed','power_s0','power_s1','power_s2','dayPower']]
#test_data=test_data[['weatherCode','humidity','temperature','windSpeed','power_s0','power_s1','power_s2','dayPower']]
df = data.copy()

len_train = math.ceil(len(df)*0.7)
values = df.values
true = values[len_train:,-1].copy()

# 对每一个特征归一化
m, n = values.shape
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
len_test = math.ceil(len(test_X)*0.01)
train_predictions = alg.predict(train_X)
test_predictions = alg.predict(test_X)

pre = mmn[-1].inverse_transform(test_predictions)
#画出预测值和真实值
plt.title('test')
plt.plot(true[:len_test],label='true')
plt.plot(pre[:len_test],label='pre')
plt.ylabel('power(KWH)')
plt.xlabel('n_sample')
plt.legend()
plt.savefig('./pic/linear_pre_test.png')
plt.show()

print('result:',mse(train_y,train_predictions),rmse(train_y,train_predictions))
print('result:',mse(test_y,test_predictions),rmse(test_y,test_predictions))
print('result:',mse(true,pre),rmse(true,pre))