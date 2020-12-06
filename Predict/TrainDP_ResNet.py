#训练过程
import os
import pickle as pickle
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from dP_ResNet.ST_DP_ResNet import stresnet
from dP_ResNet.config import Config
from dP_ResNet import metrics, PowerJN
from keras.callbacks import TensorBoard
from keras import backend as K
from dP_ResNet.metrics import rmse
from keras.utils import plot_model

K.set_image_data_format('channels_first')
'''
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config=tf.ConfigProto()
config.gpu_options.allocator_type='BFC'
config.gpu_options.per_process_gpu_memory_fraction=0.3
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
'''
DATAPATH = Config().DATAPATH  # 获取训练数据的地址    congig绝对路径
nb_epoch = 500  # 选取部分训练数据作为测试数据，训练的册数
# nb_epoch = 700
nb_epoch_cont = 100  # 测试数据，训练数据进行验证
batch_size = 32  # 批大小
T = 1  # 一天中的时间间隔数

lr = 0.0002  # adam的常设0.001
len_closeness = 3  # 邻近性依赖序列的长度
# 只考虑邻近性
len_period = 0  # 周期性依赖序列的长度
len_trend = 3  # 趋势性依赖序列的长度
nb_residual_unit = 3  # 残差单元的数量  3层

nb_flow = 1  # 电量维度

map_height, map_width = 16, 16

# 结果路径、模型路径 并创建
path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:   #判断某一路径是否为目录
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)


# 建立基于纽约自行车数据集的人群流动预测模型，external_dim为元数据的长度，在这里为8。
def build_model(len_closeness=len_closeness,len_period=len_period,len_trend=len_trend,
                nb_residual_unit=nb_residual_unit,map_height=map_height,map_width=map_width,  #初始化
                metadata_dim=None):
    # 为时空数据的特性分别创建元组，来保存写属性，len_*为依赖序列的长度，nb_flow, map_height, map_width,为流的种类，地图高，宽
    c_conf = (len_closeness, nb_flow, map_height, map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height, map_width) if len_period >= 0 else None
    t_conf = (len_trend, nb_flow, map_height, map_width) if len_trend > 0 else None

    model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=metadata_dim, nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[rmse])  #性能函数（均方误差）
    model.summary()

    return model


def main():
    # 加载数据
    print("加载数据...")
    # X_train, Y_train, 训练的输入数据， 训练的真实数据
    # X_test, Y_test, 测试的输入数据， 测试的真实数据
    # mmn, external_dim, 进行最小最大归一化的对象， 每一项元数据的维度
    X_train, Y_train, X_test, Y_test, mmn, metadata_dim, \
    timestamp_train,  timestamp_test = PowerJN.load_data(T=T,
                                                         nb_flow=nb_flow,
                                                         len_closeness=len_closeness,
                                                         len_period=len_period,
                                                         len_trend=len_trend,
                                                         preprocess_name='preprocessing.pkl',
                                                         meteorol_data=True,
                                                         meta_data=True)
    # timestamp_test[0::T], 从0开始，每周期取一个时间戳，然后v[:8]获取日期
    print("使用这几天作为测试数据: ", [v[:8] for v in timestamp_test[0::T]])  #0~ 以T为步长

    print('=' * 10)
    print("编译模型...")
    # 建立模型，external_dim是元数据列表中每个元素的形状
    model = build_model(len_closeness,len_period,len_trend,nb_residual_unit,map_height, map_width,metadata_dim)

    # 模型保存的目录及文件名
    # hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}'.format(len_closeness, len_period, len_trend, nb_residual_unit, lr)
    hyperparams_name = 'c{}.t{}.resunit{}.lr{}.map{}x{}.model'.\
        format(len_closeness,len_trend,nb_residual_unit,lr,map_width, map_height)
    # fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
    fname_param = 'c{}.t{}.resunit{}.lr{}.map{}x{}.result'.\
        format(len_closeness,len_trend,nb_residual_unit,lr,map_width, map_height)

    # 当被监测的数据不再提升，则停止训练。
    # monitor: 被监测的数据。
    # min_delta: 在被监测的数据中被认为是提升的最小变化， 例如，小于min_delta的绝对变化会被认为没有提升。
    # patience: 没有进步的训练轮数，在这之后训练就会被停止。
    # verbose: 详细信息模式。
    # mode: {auto, min, max} 其中之一。
    # min模式中， 当被监测的数据停止下降，训练就会停止；在
    # max模式中，当被监测的数据停止上升，训练就会停止；在
    # auto模式中，方向会自动从被监测的数据的名字中判断出来。
    # baseline: 要监控的数量的基准值。 如果模型没有显示基准的改善，训练将停止。
    # restore_best_weights: 是否从具有监测数量的最佳值的时期恢复模型权重。 如果为False，则使用在训练的最后一步获得的模型权重
    early_stopping = EarlyStopping(monitor='val_rmse', patience=20, mode='min')   #在验证集上面开始下降的时候中断训练

    # 在每个训练期之后保存模型
    # filepath: 字符串，保存模型的路径。
    # monitor: 被监测的数据。
    # verbose: 详细信息模式，0或者1。
    # save_best_only: 如果 save_best_only = True， 被监测数据的最佳模型就不会被覆盖。
    # mode: {auto, min, max}的其中之一。 如果 save_best_only = True，那么是否覆盖保存文件的决定就取决于被监测数据的最大或者最小值。
    # 对于val_acc，模式就会是max，而对于val_loss，模式就需要是min，等等。 在auto模式中，方向会自动从被监测的数据的名字中判断出来。
    # save_weights_only: 如果True，那么只有模型的权重会被保存(model.save_weights(filepath))， 否则的话，整个模型会被保存(model.save(filepath))。
    # period: 每个检查点之间的间隔（训练轮数）。
    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print('=' * 10)
    print("交叉验证...")
    print("训练模型（挑选部分训练数据，测试评估）...")

    # 以给定数量的轮次（数据集上的迭代）训练模型。
    # x: 训练数据
    # y: 目标数据（标签）
    # batch_size: 整数或None。每次梯度更新的样本数。如果未指定，默认为32。
    # epochs: 整数。训练模型迭代轮次。一个轮次是在整个x和y上的一轮迭代。
    # 请注意，与initial_epoch（开始训练的轮次）一起，epochs被理解为 「最终轮次」。模型并不是训练了epochs轮，而是到第epochs轮停止训练。
    # validation_split: 0 和 1 之间的浮点数。用作验证集的训练数据的比例。 模型将分出一部分不会被训练的验证数据，并将在每一轮结束时评估这些验证数据的误差和任何其他模型指标。
    # 验证数据是混洗之前 x 和y 数据的最后一部分样本中。
    # verbose: 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
    # callbacks：一系列可以在训练时使用的回调函数。
    # 返回一个 History 对象。其 History.history 属性是连续 epoch 训练损失和评估值，以及验证集损失和评估值的记录（如果适用）。
    history = model.fit(X_train, Y_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        validation_data=[X_test, Y_test],
                        callbacks=[early_stopping,],#TensorBoard()
                        verbose=1)
    model.save_weights("model.h5", overwrite=True)

    #model.save_weights(os.path.join(path_model,'{}.h5'.format(hyperparams_name)), overwrite=True)
    #pickle.dump((history.history), open(os.path.join( path_result,'{}.history.pkl'.format(fname_param)), 'wb'))

    '''
    print('=' * 10)
    print("使用最优loss值的模型对验证集进行评价：")

    model.load_weights(fname_param)
    # 在测试模式下返回模型的误差值和评估标准值。
    # x: 训练数据
    # y: 目标数据（标签）
    # batch_size: 整数或 None。每次评估的样本数。如果未指定，默认为 32。
    # verbose: 0 = 安静模式，1 = 进度条。
    # steps: 整数 或 None。 声明评估结束之前的总步数（批次样本）。默认值None。
    # 标量测试误差（如果模型只有一个输出且没有评估标准） 或标量列表（如果模型具有多个输出和或评估指标）
    score = model.evaluate(X_train, Y_train, batch_size=32, verbose=0)
    print('训练数据集上评估： loss: %.6f        rmse (norm): %.6f       rmse (real): %.6f' %(score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))

    score = model.evaluate(X_test, Y_test, batch_size=32, verbose=0)
    print('测试数据集上评估： loss: %.6f        rmse (norm): %.6f       rmse (real): %.6f' %(score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))

    print('=' * 10)
    print("训练模型 (训练数据训练，测试数据测试)...")
    fname_param = os.path.join('MODEL', '{}.cont.best.h5'.format(hyperparams_name))

    model_checkpoint = ModelCheckpoint(fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')

    history = model.fit(X_train, Y_train, epochs=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[model_checkpoint], validation_data=(X_test, Y_test))
    pickle.dump((history.history), open(os.path.join(path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    model.save_weights(os.path.join('MODEL', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
    '''
    print('=' * 10)
    print("使用最终的模型进行评价：")
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
    print('训练数据集上评估 loss: %.6f        rmse (norm): %.6f       rmse (real): %.6f' % (
    score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))

    score = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0] // 48, verbose=0)
    print('测试数据集上评估 loss: %.6f        rmse (norm): %.6f       rmse (real): %.6f' % (
    score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    return model, X_train, Y_train, X_test, Y_test
    plot_model(model,to_file="model.png",show_shapes=True)

if __name__=="__main__":
    model=main()