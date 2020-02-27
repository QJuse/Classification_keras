# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:02:52 2019

@author: zhengqj
------------------------------------------------------------------
Architecture: Convs strides=1, padding=same 
                params=f*k*k*C + C , calc= params*N*N
    b1_conv1: 64, 3x3,       [params:1,792, calc:89,915,392]
    b1_conv2: 64, 3x3,       [params:36,928, calc:1,852,899,328]
    b1_pool: 2x2, strides=2                     
    b2_conv1: 128, 3x3,      [params:73,856, calc:959,832,576]
    b2_conv2: 128, 3x3,      [params:147,584, calc:1,918,001,664]
    b2_pool: 2x2, strides=2
    b3_conv1: 256, 3x3,      [params:295,168, calc:925,646,848]
    b3_conv2: 256, 3x3,      [params:590,080, calc:1,850,490,880]
    b3_conv3: 256, 3x3,      [params:590,080, calc:1,850,490,880]
    b3_pool: 3x3, strides=2
    b4_conv1: 512, 3x3,      [params:1,180,160, calc:925,245,440]
    b4_conv2: 512, 3x3,      [params:2,359,808, calc:1,850,089,472]
    b4_conv3: 512, 3x3,      [params:2,359,808, calc:1,850,089,472]
    b4_pool: 3x3, strides=2
    b5_conv1: 512, 3x3,      [params:2,359,808, calc:462,522,368]
    b5_conv2: 512, 3x3,      [params:2,359,808, calc:462,522,368]
    b5_conv3: 512, 3x3,      [params:2,359,808, calc:462,522,368]
    b5_pool: 3x3, strides=2 
    FC1: 4096                [params:102,764,544, calc:102,764,544]
    FC2: 4096                [params:16,781,312, calc:16,781,312]
    FC3: 1000                [params:4,097,000, calc:4,097,000]
                [params: 130M, calc:159e8 Flos]
------------------------------------------------------------------
"""


import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from tensorflow.keras.utils import plot_model
from collections import OrderedDict


class VGG:
    def __init__(self, net_config, nclass=1000):
        self.nclass = nclass
        self.vgg16 = OrderedDict()
        for op in net_config:
            self.vgg16.update(op)

    def build(self, input_shape):
        x_input = KL.Input(input_shape, name='x_input')
        x = x_input
        for key, param in self.vgg16.items():
            if key.startswith('conv'):
                channel, kernel = param
                x = KL.Conv2D(channel, (kernel, kernel),
                           activation='relu',
                           padding='same',
                           name=key)(x)  # outmap [224, 224, 64]
            elif key.startswith('maxpool'):
                x = KL.MaxPooling2D((2, 2), strides=(2, 2),
                                 name=key)(x)  # outmap [112, 112, 64]
            elif key.startswith('fc'):
                if key.startswith('fc1'):
                    x = KL.Flatten(name='flatten')(x)
                channel = param[0]
                x = KL.Dense(channel, activation='relu', name=key)(x)
            elif key.startswith('dropout'):
                rate = param[0]
                x = KL.Dropout(rate, name=key)(x)
        y_output = KL.Dense(self.nclass, activation='softmax', name='y_output')(x)
        self.model = KM.Model(x_input, y_output)


network = [
    # block1
    {'conv1_b1':[64, 3]}, {'conv2_b1':[64, 3]}, {'maxpool_b1':[2, 2]},
    # block2
    {'conv3_b2':[128, 3]}, {'conv4_b2':[128, 3]}, {'maxpool_b2':[2, 2]},
    # block3
    {'conv5_b3':[256, 3]}, {'conv6_b3':[256, 3]}, {'conv7_b3':[256, 3]}, {'maxpool_b3':[2, 2]},
    # block4
    {'conv8_b4':[512, 3]}, {'conv9_b4':[512, 3]}, {'conv10_b4':[512, 3]}, {'maxpool_b4':[2, 2]},
    # block5
    {'conv11_b5':[512, 3]}, {'conv12_b5':[512, 3]}, {'conv13_b5':[512, 3]}, {'maxpool_b5':[2, 2]},
    # head
    {'fc1':[4096]}, {'dropout1':[0.5]}, {'fc2':[4096]}, {'dropout2':[0.5]}
]
def test_vgg():
    vgg16 = VGG(network)
    vgg16.build((224, 224, 3))

    SAVE_NAME = None
    if SAVE_NAME:
        vgg16.model.summary()
        plot_model(vgg16.model, to_file='model_plot/' + SAVE_NAME, show_shapes=True)

# %%
# 可以将keras预训练的权值加载到自定义的vgg16上
import os
root = os.path.abspath(os.path.join(os.getcwd()))
pre_weight = os.path.join(root, 'pre_weight','vgg16.h5')

vgg16 = VGG(network)
vgg16.build((224, 224, 3))
vgg16.model.load_weights(pre_weight)
# 查看层和权值
layer1 = vgg16.model.layers[1]
print(layer1.get_weights()[0].shape)
print(layer1.input, layer1.output)
print(layer1.get_config(), end=' ')  ## !!




