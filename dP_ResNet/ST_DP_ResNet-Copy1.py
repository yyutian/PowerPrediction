'''
    ST-dP_ResNet: Deep Spatio-temporal Residual Networks
'''


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
            input = BatchNormalization(axis=1, momentum=0)(input)
        activation = Activation('relu')(input)
        return Convolution2D(filters=nb_filter, kernel_size=(nb_row, nb_col), strides=subsample, padding="same")(activation)
    return f


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    '''
    残差单元
    :param nb_filter: 滤波器的个数
    :param init_subsample: 初始步长
    :return: 返回一个残差单元
    '''
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)
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
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter,init_subsample=init_subsample)(input)
        return input
    return f


def stresnet(c_conf=(3, 1, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32), external_dim=12, nb_residual_unit=3):
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
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None and conf[0] != 0:
            # 一下分别为：序列长度、流的种类、地图高度、地图宽度
            len_seq, nb_flow, map_height, map_width = conf
            # 实例化一个keras张量，shape=[?, nb_flow * len_seq, map_height, map_width],
            # 预期的输入将是一批[nb_flow * len_seq, map_height, map_width]
            input = Input(shape=(nb_flow * len_seq, map_height, map_width))
            main_inputs.append(input)
            # 卷积1
            print("卷积1中输入的shape", input.shape)
            conv1 = Convolution2D(filters=64, kernel_size=(3, 3), padding="same")(input)
            # nb_residual_unit个残差单元
            residual_output = ResUnits(_residual_unit, nb_filter=64,
                              repetations=nb_residual_unit)(conv1)
            # 卷积2
            activation = Activation('relu')(residual_output)
            conv2 = Convolution2D(
                filters=nb_flow, kernel_size=(3, 3), padding="same")(activation)
            outputs.append(conv2)

    # 基于参数矩阵的融合
    if len(outputs) == 1:
        # 只有一个组件时，无需融合
        main_output = outputs[0]
    else:
        from .iLayer import iLayer
        new_outputs = []
        for output in outputs:
            # 构建一层网络，计算该组件的输出乘以权重， X * W
            new_outputs.append(iLayer()(output))
        # 基于参数的矩阵融合，用以融合各组件的输出， 将各组件的输出乘以权重后的结构相加。
        main_output = add(new_outputs)

    # 与外部组件融合
    if external_dim != None and external_dim > 0:
        # 外部输入
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=100)(external_input)
        embedding = Activation('relu')(embedding)
        embedding = Dense(units=10)(embedding)
        embedding = Activation('relu')(embedding)
        h1 = Dense(units=nb_flow * map_height * map_width)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((nb_flow, map_height, map_width))(activation)
        # 三个组件的输出与外部组件的输出相加，进一步融合
        main_output = add([main_output, external_output])
    else:
        print('external_dim:', external_dim)
    #
    main_output = Activation('relu')(main_output)
    #main_output = Convolution2D(filters=nb_flow, kernel_size=(3, 3), padding="same")(main_output)
    model = Model(inputs=main_inputs, outputs=main_output)

    return model


if __name__ == '__main__':
    model = stresnet(external_dim=12, nb_residual_unit=12)
    #plot(model, to_file='ST-dP_ResNet.png', show_shapes=True)
    model.summary()