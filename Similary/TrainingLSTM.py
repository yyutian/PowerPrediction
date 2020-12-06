from keras.layers import (
    Input,
    Activation,
    Dense,
    Reshape,
    LSTM,
    Dropout,
    Bidirectional
)
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from keras.models import Sequential
from keras.models import Model
import numpy as np
import math
import pandas as pd
# <<<<<<< Updated upstream
from dP_ResNet.minmax_normalization import MinMaxNormalization
from dP_ResNet import metrics
# =======
from dP_ResNet.minmax_normalization import*
import Similary.metrics
# >>>>>>> Stashed changes
import pickle

def load_Data(url):
    data = pd.read_csv(url)

    data = data[['weatherCode', 'humidity', 'temperature', 'windSpeed', 'power_s0', 'power_s1', 'power_s2', 'dayPower']]

    values = data.values
    print(values[:5])
    '''
    label = np.array([x for x in range(len(data))])
    np.random.seed(7379)
    np.random.shuffle(values)
    np.random.seed(7379)
    np.random.shuffle(label)
    print('标签:',label)
    '''

    m, n = values.shape
    print((m,n))
    #对每一个特征归一化
    mmn=[]
    for j in range(5):
        mmn.append(MinMaxNormalization())
    for i in range(4):
        values[:, i] = mmn[i].fit_transform(values[:, i].reshape(m, 1)).reshape(m)
    values[:,4:]=mmn[-1].fit_transform(values[:,4:])
    #values=scaler.fit_transform(values)

    train_len = math.ceil(len(values) * 0.6)
    valid_len = math.ceil(len(values) * 0.2)
    #训练集60%，验证集20%，测试集20%
    train = values[:train_len, :]
    valid = values[train_len:train_len+valid_len, :]
    test = values[train_len+valid_len:, :]

    train_X, train_y = train[:, :-1], train[:, -1]
    valid_X, valid_y = valid[:,:-1], valid[:,-1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape为LSTM输入格式，即[samples,timesteps,features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    return train_X,train_y,valid_X,valid_y,test_X,test_y,mmn

def lstm(input_shpae=(1,7)):
    model=Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shpae, return_sequences=True))
    model.add(LSTM(128,activation='relu', return_sequences=True))
    model.add(LSTM(256,activation='relu',  return_sequences=False))
    #model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mse', optimizer='adam',metrics=[metrics.rmse])

    return model
def Bi_LSTM():
    model=Sequential()
    #model.add(Embedding(len(train_X),3,input_length=len(train_X)))
    #model.add(Dropout(0.2))
    #model.add(Similary(32))
    input = Input(shape=(train_X.shape[1], train_X.shape[2]))
    lstm=Bidirectional(LSTM(50,activation='relu',input_shape=(train_X.shape[1], train_X.shape[2])),merge_mode='concat')(input)
    lstm=Dense(256,activation='relu')(lstm)
    dropout=Dropout(0.2)(lstm)
    output=Dense(1,activation='sigmoid')(dropout)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=[metrics.rmse])
    return model

if __name__=='__main__':
    import warnings
    import os
    from keras import backend as K

    K.set_image_data_format('channels_first')
    warnings.filterwarnings('ignore')  # 过滤掉版本即将过期等无关警告
    #一个基于一个电站的数据输入训练，一个基于148个电站数据进行训练
    #url = './data/FinalData/similary149.csv'
    url = './data/FinalData/similary_all.csv'
    train_X, train_y, valid_X, valid_y, test_X, test_y, mmn= load_Data(url)

    print('训练集输入格式：', train_X.shape, '训练集输出格式：', train_y.shape)
    print('验证集输入格式：', valid_X.shape, '验证集输出格式：', valid_y.shape)
    print('测试集输入格式：', test_X.shape, '测试集输出格式：', test_y.shape)

    path_result = 'RET'
    path_model = 'MODEL'
    if os.path.isdir(path_result) is False:
        os.mkdir(path_result)
    if os.path.isdir(path_model) is False:
        os.mkdir(path_model)

    batch_size = 128
    epochs = 500
    input_shape=(train_X.shape[1], train_X.shape[2])
    print(input_shape, "66666")
    model = lstm(input_shape)

    early_stopping = EarlyStopping(monitor='val_rmse', patience=10, mode='min')
    model_checkpoint = ModelCheckpoint('./MODEL/lstm2_best.h5', monitor='val_rmse', verbose=0, save_best_only=True,
                                       mode='min')
    history = model.fit(train_X, train_y,
                        validation_data=[valid_X,valid_y],
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[early_stopping,model_checkpoint,TensorBoard()],
                        shuffle=False)
    # model.save_weights('./MODEL/Bi_lstm.h5', overwrite=True)
    pickle.dump((history.history), open(os.path.join(path_result, 'lstm2.history.pkl'), 'wb'))
    model.save_weights('./model/lstm2.h5', overwrite=True)

