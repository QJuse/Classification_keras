# -*- coding: utf-8 -*-
"""
author: zhengqj@fotile.com
"""
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 # 299
from tensorflow.keras.applications.resnet import ResNet50, ResNet101
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.nasnet import NASNetMobile

import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

assert tf.__version__.startswith('2'), 'please use tensorflow 2.0'

MODELS = {
    'vgg16': VGG16,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'densenet121': DenseNet121,
    'mobilenet': MobileNet,
    'mobilenet_v2': MobileNetV2,
    'nasnet': NASNetMobile,
    'inception_v3': InceptionV3,
    'inception_resnet': InceptionResNetV2
}


class CustomModel(object):
    def __init__(self, network, pre_weight, nclass):
        assert network in MODELS, "please choose the right keras model, in, " \
            "vgg16, resnet50, resnet101, densenet121, inception_v3," \
            "nasnet, mobilenet, mobilenet_v2, inception_resnet"
        self.NetWork = MODELS[network]
        self.img_size = 299 if network.startswith('inception') else 224
        self.pre_weight = pre_weight
        self.nclass = nclass

    def build_model(self,
                    include_top=True,
                    weights=None,
                    pooling=None):
        if include_top:
            base_model = self.NetWork(include_top=include_top,
                                            weights=weights)
            base_model.load_weights(self.pre_weight)
            input_x = base_model.layers[-2].output
            y = KL.Dense(self.nclass, activation='softmax', name='mypred')(input_x)
        else:
            input_shape = (self.img_size, self.img_size, 3)
            base_model = self.NetWork(include_top=include_top,
                                           weights=weights,
                                           input_shape=input_shape,
                                           pooling=pooling)
            base_model.load_weights(self.pre_weight, by_name=True)  # 取得特征图
            y = base_model.output
        model = KM.Model(inputs=base_model.input, outputs=y)
        return model

__all__ = ['MODELS', 'CustomModel']


