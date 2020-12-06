import pickle
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


fr = open('./RET/c7.t5.resunit6.lr0.0002.map16x16.result.history.pkl','rb')
history=pickle.load(fr)
#画出目标函数的变化
plt.plot(history.get('loss'),label='loss')
plt.plot(history.get('val_loss'),label='val_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('./pic/loss.png')
plt.show()

plt.plot(history.get('rmse'),label='rmse')
plt.plot(history.get('val_rmse'),label='val_rmse')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend()
plt.savefig('./pic/RMSE.png')
plt.show()
