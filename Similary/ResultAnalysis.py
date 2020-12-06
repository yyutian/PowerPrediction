import pickle
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#fr = open('./RET/lstm.history.pkl','rb')
fr = open('./RET/lstm2.history.pkl','rb')
history=pickle.load(fr)
#画出目标函数的变化

plt.title('mean_square_error 均方误差变化曲线')
plt.plot(history.get('loss'),label='loss')
plt.plot(history.get('val_loss'),label='val_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('./pic/lstm_loss.png')
plt.show()

plt.title('root_mean_square_error 均方根误差变化曲线')
plt.plot(history.get('rmse'),label='rmse')
plt.plot(history.get('val_rmse'),label='val_rmse')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend()
plt.savefig('./pic/lstm_rmse.png')
plt.show()