'''
    ST-dP_ResNet: Deep Spatio-temporal Residual Networks
'''

# 学习使用代码
from keras.layers import (
    Input,
    Activation,
    Dense,
    Reshape
)
from keras.layers.convolutional import Convolution2D
from keras.layers import add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from dP_ResNet.iLayer import iLayer
from keras.utils import plot_model
#from keras.utils.visualize_util import plot


def _shortcut(input, residual):
    # 对张量执行求和运算, 用来作为残差单元的捷径
    # 返回一个计算捷径的函数
    return add([input, residual])


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    '''
    :param nb_filter:滤波器的个数
    :param nb_row:卷积核的行数
    :param nb_col:卷积核的列数
    :param subsample:步长
    :param bn:是否在relu之前添加批标准化
    :return:返回一个函数，先后对input进行批标准化，relu，和卷积。
    '''
    def f(input):
        if bn:
            input = BatchNormalization(axis=1, momentum=0)(input)#使用BN对输入标准化
        activation = Activation('relu')(input)#使用input为参数调用relu函数进行处理
        # print(input)
        return Convolution2D(filters=nb_filter, kernel_size=(nb_row, nb_col), strides=subsample, border_mode="same")(activation)
         #进行卷积处理
    return f


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    '''
    残差单元
    :param nb_filter: 滤波器的个数
    :param init_subsample: 初始步长
    :return: 返回一个残差单元
    '''
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)#对input做一次3X3卷积
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)#对input再做一次3X3卷积
        return _shortcut(input, residual)#返回input和residual的和
    return f


def ResUnits(residual_unit, nb_filter, repetations=1):
    '''
    构建一系列残差单元
    :param residual_unit:用来构建残差单元的函数
    :param nb_filter:滤波器的个数
    :param repetations:残差单元的个数
    :return:返回生出repetations个残差单元堆叠成的残差模块的函数
    '''
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)#步长
            input = residual_unit(nb_filter=nb_filter,init_subsample=init_subsample)(input)#以input为参数构建残差单元
        return input
    return f


def stresnet(c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32), external_dim=12, nb_residual_unit=3):
    '''
    C - 时间邻近性
    P - 周期性
    T - 趋势性
    conf = (序列长度, 流的种类, 地图高度, 地图宽度)
    external_dim 元数据列表元素的长度
    nb_residual_unit 残差单元的数量
    '''

    # main input
    main_inputs = []
    outputs = []    #用来存储将时间邻近性、周期性和趋势性矩阵处理后的参数矩阵
    for conf in [c_conf, p_conf, t_conf]:    #依次取出c_conf, p_conf, t_conf
        if conf is not None and conf[0] != 0:
            # 一下分别为：序列长度、流的种类、地图高度、地图宽度
            len_seq, nb_flow, map_height, map_width = conf   #分别赋值   流的种类：流入、流出两张图  len_seq：取三个时间点，每个时间点取两张图片（流入、流出）
            # 实例化一个keras张量，shape=[?, nb_flow * len_seq, map_height, map_width],
            # 预期的输入将是一批[nb_flow * len_seq, map_height, map_width]
            input = Input(shape=(nb_flow * len_seq, map_height, map_width))#创建一个指定shape的输入（？，6,32,32）
            main_inputs.append(input)#将input存入main_inputs中
            # 卷积1
            print("卷积1中输入的shape", input.shape)
            conv1 = Convolution2D(filters=64, kernel_size=(3,3), padding="same")(input)
            #对input进行滤波器为64，3X3的卷积，通过设置超参数 padding=’same’使残差单元不会改变输入输出的形状
            # nb_residual_unit个残差单元
            residual_output = ResUnits(_residual_unit, nb_filter=64,repetations=nb_residual_unit)(conv1)
            #创建nb_residual_unit（3）个残差单元
            # 卷积2
            activation = Activation('relu')(residual_output)#以残差单元为参数调用relu函数
            conv2 = Convolution2D(filters=nb_flow, kernel_size=(3,3), padding="same")(activation)
            # 对activation进行滤波器为nb_flow（2），3X3的卷积，通过设置超参数 padding=’same’使残差单元不会改变输入输出的形状
            outputs.append(conv2)#将处理完的数据放到outputs内

    # 基于参数矩阵的融合

    if len(outputs) == 1:
        # 只有一个组件时，无需融合
        main_output = outputs[0]
    else:
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))# 构建一层网络，计算该组件的输出乘以权重， X * W
        main_output = add(new_outputs)# 基于参数的矩阵融合，用以融合各组件的输出， 将各组件的输出乘以权重后的结构相加。
    # 与外部组件融合
    if external_dim != None and external_dim > 0:
        # 外部输入
        external_input = Input(shape=(external_dim,))#创建一个指定格式的输入（？，12） 1维
        main_inputs.append(external_input)   #将external_input放到main_inputs中
        embedding = Dense(units=100)(external_input)#Dense全连接层，units=100：输出的维度大小为100，改变external_input的最后一维(?,100)
        #全连接层通常在CNN的尾部进行重新拟合，减少特征信息的损失
        embedding = Activation('relu')(embedding)#以embedding为参数调用relu函数
        embedding = Dense(units=10)(embedding)#Dense全连接层，units=10：输出的维度大小为10，改变external_input的最后一维(?,10)
        embedding = Activation('relu')(embedding)#以embedding为参数调用relu函数
        h1 = Dense(units=int(main_output.shape[1]) * int(main_output.shape[2]) * int(main_output.shape[3]))(embedding)
        #Dense全连接层，units=nb_flow * len_seq * map_height * map_width：输出的维度大小为6144，改变external_input的最后一维
        activation = Activation('relu')(h1)#以h1为参数调用relu函数
        external_output = Reshape((int(main_output.shape[1]), int(main_output.shape[2]), int(main_output.shape[3])))(activation)
        #reshape为和main_output一样的shape
        main_output = add([main_output, external_output])# 三个组件的输出与外部组件的输出相加，进一步融合
    else:
        print('external_dim:', external_dim)
    #
    main_output = Activation('relu')(main_output)#以main_output为参数调用relu函数
    #main_output = Convolution2D(filters=nb_flow, kernel_size=(3, 3), padding="same")(main_output)
    model = Model(inputs=main_inputs, outputs=main_output)#以main_inputs为输入，以main_output为输出创建模型
    return model
    plot_model(model,to_file="model.png",show_shapes=True)

if __name__ == '__main__':
    model = stresnet(external_dim=12, nb_residual_unit=12)
    #plot(model, to_file='ST-DP_ResNet.png', show_shapes=True)
    model.summary()#输出模型各层的参数状况