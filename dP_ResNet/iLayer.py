import os
import tensorflow
from keras import backend as K
from keras.engine.topology import Layer
# from keras.layers import Dense
import numpy as np


class iLayer(Layer):
    '''
    编写Layer继承类,我们可以通过继承来实现自己的层。
    要定制自己的层，需要实现下面三个方法
    build(input_shape)：这是定义权重的方法，可训练的权应该在这里被加入列表self.trainable_weights中。其他的属性还包括self.non_trainabe_weights（列表）和self.updates（需要更新的形如（tensor,new_tensor）的tuple的列表）。这个方法必须设置self.built = True，可通过调用super([layer],self).build()实现。
    call(x)：这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心call的第一个参数：输入张量。
    compute_output_shape(input_shape)：如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法，这个函数使得Keras可以做自动shape推断。
    '''
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        这是定义权重的方法
        :param input_shape: 输入形状
        '''
        # 实例化了一个共享变量，作为矩阵融合的参数。
        initial_weight_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        # 将可训练的权加入列表self.trainable_weights中
        self.trainable_weights = [self.W]
    # 定义层功能的方法
    def call(self, x, mask=None):
        return x * self.W

    # 指定shape变化的方法
    def get_output_shape_for(self, input_shape):
        return input_shape
