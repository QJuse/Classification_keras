# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:54:15 2019

@author: zhengqj

Inception_Resnet_V2: LSVRC-ImageNet
-----------------------------------------------------------------
Architecture:  
    see the paper <Inception-v4, Inception-ResNet and 
                    the Impact of Residual Connections on Learning>
------------------------------------------------------------------
"""
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import (Activation, AveragePooling2D,
                                     BatchNormalization, Concatenate, Conv2D,
                                     Dense, Dropout, GlobalAveragePooling2D,
                                     Input, MaxPooling2D)
from tensorflow.keras.models import Model

SAVE_MODEL_PLOT = True
if SAVE_MODEL_PLOT:
    from tensorflow.keras.utils import plot_model
    SAVE_NAME = 'Image_inception_V4.jpg'


def conv2d_bn(x, filters, kernel, strides=(1, 1), padding='same', name=None):
    """Utility function to apply conv + BN. conv ignore bias"""
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(filters,
               kernel,
               strides=strides,
               padding=padding,
               use_bias=False,
               name=conv_name)(x)
    x = BatchNormalization(axis=3, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def Inception_Resnet_V2(input_shape=None, nclass=1000, debug=True):
    if input_shape is None:
        input_shape = (299, 299, 3)
    # truck
    x_input = Input(input_shape, name='x_input')
    x = conv2d_bn(x_input, 32, (3, 3), strides=(2, 2),
                  padding='valid')  # [149, 149, 32]
    x = conv2d_bn(x, 32, (3, 3), padding='valid')  # [147, 147, 32]
    x = conv2d_bn(x, 64, (3, 3))
    x = stem_maxpool(x, 96)  # [73, 73, 160]
    x = stem_conv(x)  # [71, 71, 192]
    x = stem_maxpool(x, 192)  # [35, 35, 384]
    # inception resnet module A
    for i in range(5):
        x = incpt_module_a(x)
    x = reduce_35(x)  # [17, 17, 1152]
    for i in range(10):
        x = incpt_module_a(x)
    x = reduce_17(x)
    for i in range(5):
        x = incpt_module_c(x)
    print(x)

    # truck
    x = GlobalAveragePooling2D(name='g_avg_pool')(x)  # [None, 1536]
    x = Dropout(0.2)(x)
    y_output = Dense(nclass, activation='softmax', name='y_output')(x)

    print(y_output)
    model = Model(x_input, y_output)
    # plot the model pictures
    #    if SAVE_MODEL_PLOT:
    #        plot_model(model, to_file='model_plot/' + SAVE_NAME, show_shapes=True)

    if debug:
        for i, layer in enumerate(model.layers):
            print("pt1:layer_{} name is {}".format(i, layer.name))

    return model


def stem_maxpool(base_tensor, filters=None, name=None):
    tower_conv = conv2d_bn(base_tensor,
                           filters, (3, 3),
                           strides=(2, 2),
                           padding='valid')
    tower_pool = MaxPooling2D((3, 3), strides=(2, 2))(base_tensor)
    mixout = Concatenate(axis=-1, name=name)([tower_conv, tower_pool])
    return mixout


def stem_conv(base_tensor, name=None):
    # branch 1
    tower_0 = conv2d_bn(base_tensor, 64, (1, 1))
    tower_0 = conv2d_bn(tower_0, 96, (3, 3), padding='valid')
    # branch 2
    tower_1 = conv2d_bn(base_tensor, 64, (1, 1))
    tower_1 = conv2d_bn(tower_1, 64, (7, 1))
    tower_1 = conv2d_bn(tower_1, 64, (1, 7))
    tower_1 = conv2d_bn(tower_1, 96, (3, 3), padding='valid')
    mixout = Concatenate(axis=-1, name=name)([tower_0, tower_1])
    return mixout


def incpt_module_a(base_tensor, name=None):
    """  the feature map is 35x35  """
    outdims = base_tensor.get_shape().as_list()[-1]
    # conv1
    tower_0 = conv2d_bn(base_tensor, 32, (1, 1))
    # conv3
    tower_1 = conv2d_bn(base_tensor, 32, (1, 1))
    tower_1 = conv2d_bn(tower_1, 32, (3, 3))
    # conv5
    tower_2 = conv2d_bn(base_tensor, 32, (1, 1))
    tower_2 = conv2d_bn(tower_2, 48, (3, 3))
    tower_2 = conv2d_bn(tower_2, 64, (3, 3))
    # merge
    mixout = Concatenate(axis=-1, name=name)([tower_0, tower_1, tower_2])
    mixout = Conv2D(outdims, (1, 1))(mixout)
    # merge
    output = layers.add([base_tensor, mixout])
    output = Activation('relu')(output)
    return output


def incpt_module_b(base_tensor, name=None):
    """  the feature map is 35x35  """
    outdims = base_tensor.get_shape().as_list()[-1]
    # conv1
    tower_0 = conv2d_bn(base_tensor, 192, (1, 1))
    # conv3
    tower_1 = conv2d_bn(base_tensor, 128, (1, 1))
    tower_1 = conv2d_bn(tower_1, 160, (1, 7))
    tower_1 = conv2d_bn(tower_1, 192, (7, 1))
    # merge
    mixout = Concatenate(axis=-1, name=name)([tower_0, tower_1])
    mixout = Conv2D(outdims, (1, 1))(mixout)  # no BN
    # merge
    output = layers.add([base_tensor, mixout])
    output = Activation('relu')(output)
    return output


def incpt_module_c(base_tensor, name=None):
    """  the feature map is 35x35  """
    outdims = base_tensor.get_shape().as_list()[-1]
    # conv1
    tower_0 = conv2d_bn(base_tensor, 192, (1, 1))
    # conv3
    tower_1 = conv2d_bn(base_tensor, 192, (1, 1))
    tower_1 = conv2d_bn(tower_1, 224, (1, 3))
    tower_1 = conv2d_bn(tower_1, 256, (3, 1))
    # merge
    mixout = Concatenate(axis=-1, name=name)([tower_0, tower_1])
    mixout = Conv2D(outdims, (1, 1))(mixout)  # no BN
    # merge
    output = layers.add([base_tensor, mixout])
    output = Activation('relu')(output)
    return output


def reduce_35(base_tensor, name=None):
    """ reduce feature map dims 35 x 35 to 17 x 17  """
    # branch
    tower_0 = conv2d_bn(base_tensor,
                        384, (3, 3),
                        strides=(2, 2),
                        padding='valid')
    tower_1 = MaxPooling2D((3, 3), strides=(2, 2))(base_tensor)
    tower_2 = conv2d_bn(base_tensor, 256, (1, 1))
    tower_2 = conv2d_bn(tower_2, 256, (3, 3))
    tower_2 = conv2d_bn(tower_2, 384, (3, 3), strides=(2, 2), padding='valid')
    # merge
    output = Concatenate(axis=-1, name=name)([tower_0, tower_1, tower_2])
    return output


def reduce_17(base_tensor, name=None):
    """ reduce feature map dims 17 x 17 to 8 x 8  """
    # branch
    tower_0 = conv2d_bn(base_tensor, 256, (1, 1))
    tower_0 = conv2d_bn(base_tensor,
                        384, (3, 3),
                        strides=(2, 2),
                        padding='valid')
    tower_1 = MaxPooling2D((3, 3), strides=(2, 2))(base_tensor)
    tower_2 = conv2d_bn(base_tensor, 256, (1, 1))
    tower_2 = conv2d_bn(tower_2, 288, (3, 3))
    tower_2 = conv2d_bn(tower_2, 320, (3, 3), strides=(2, 2), padding='valid')
    # branch
    tower_3 = conv2d_bn(base_tensor, 256, (1, 1))
    tower_3 = conv2d_bn(base_tensor,
                        288, (3, 3),
                        strides=(2, 2),
                        padding='valid')
    # merge
    output = Concatenate(axis=-1,
                         name=name)([tower_0, tower_1, tower_2, tower_3])
    return output


if __name__ == "__main__":
    Inception_Resnet_V2()
