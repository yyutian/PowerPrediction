from dP_ResNet.config import Config
from dP_ResNet import metrics, PowerJN
from keras import backend as K
from Predict.TrainDP_ResNet import build_model
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from dP_ResNet.metrics import mse,rmse
K.set_image_data_format('channels_first')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

DATAPATH = Config().DATAPATH    # 获取训练数据的地址
nb_epoch = 500      # 选取部分训练数据作为测试数据，训练的册数
batch_size = 32     # 批大小
T = 1              # 时间间隔数

len_closeness = 3   # 邻近性依赖序列的长度
# 只考虑邻近性
len_period = 3     # 周期性依赖序列的长度
len_trend = 3       # 趋势性依赖序列的长度
nb_residual_unit = 4    # 残差单元的数量

nb_flow = 1     # 电量维度

#城市被划分为16 * 16的区域
map_height, map_width = 16, 16


X_train, Y_train, X_test, \
Y_test, mmn, metadata_dim, timestamp_train, \
timestamp_test = PowerJN.load_data(T=T,nb_flow=nb_flow,
                                    len_closeness=len_closeness,
                                    len_period=len_period,
                                    len_trend=len_trend,
                                    preprocess_name='preprocessing.pkl',
                                    meteorol_data=True,
                                    meta_data=True)
model=build_model(len_closeness,len_period,len_trend, nb_residual_unit,map_height, map_width,metadata_dim)

model.load_weights('./MODEL/c7.t5.resunit6.lr0.0002.map16x16.model.h5')
#model.load_weights('./MODEL/c3.t3.resunit3.lr0.0002.map16x16.model.h5')
# 绘制模型图
plot_model(model, to_file='pic/c3.t3.resunit3.lr0.0002.map16x16.model.png', show_shapes=True)

print("使用最终的模型进行评价：")
score = model.evaluate(X_train, Y_train, verbose=0)
print('训练数据集上评估 loss: %.6f        rmse (norm): %.6f       rmse (real): %.6f' %(score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
score = model.evaluate(X_test, Y_test, verbose=0)
print('测试数据集上评估 loss: %.6f        rmse (norm): %.6f       rmse (real): %.6f' %(score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
'''
#=======================训练集===================
pre=model.predict(X_train)
#反归一化
pre=mmn.inverse_transform(pre)
Y=mmn.inverse_transform(Y_train)
plt.imshow(pre[0][0])
plt.show()
plt.imshow(Y[0][0])
plt.show()
true=[]
pre_=[]
for i in range(len(Y)):
    true.append(Y[i][0][0][9])
for i in range(len(pre)):
    pre_.append(pre[i][0][0][9])
#画出某区域的预测值
plt.plot(true,label='true')
plt.plot(pre_,label='pre')
plt.xlabel('n_sample')
plt.ylabel('单位容量发电量(kwh)')
plt.legend()
plt.savefig('./pic/ResNet_pre_train.png')
plt.show()
'''
#=======================测试集===================
pre=model.predict(X_test)
#反归一化
pre=mmn.inverse_transform(pre)
Y=mmn.inverse_transform(Y_test)
true=[]
pre_=[]
for i in range(len(Y)):
    true.append(Y[i][0][0][9])
for i in range(len(pre)):
    pre_.append(pre[i][0][0][9])
#画出某区域的预测值
plt.plot(true,label='true')
plt.plot(pre_,label='pre')
plt.xlabel('n_sample')
plt.ylabel('单位容量发电量(kwh)')
plt.legend()
plt.savefig('./pic/ResNet_pre_test.png')
plt.show()