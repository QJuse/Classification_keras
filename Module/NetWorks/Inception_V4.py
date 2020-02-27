# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:54:15 2019

@author: zhengqj

Inception_V4: LSVRC-ImageNet
-----------------------------------------------------------------
Architecture:  
    Inception:  similarity with inception_V3
    see the paper <Inception-v4, Inception-ResNet and 
                    the Impact of Residual Connections on Learning>
------------------------------------------------------------------
"""

from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Activation, Concatenate
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
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

    x = Conv2D(filters, kernel, strides=strides, padding=padding,
        use_bias=False, name=conv_name)(x)
    x = BatchNormalization(axis=3, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def Inception_V4(input_shape=None, nclass=1000, debug=True):
    if input_shape is None:
        input_shape = (299, 299, 3)
    # truck
    x_input = Input(input_shape, name='x_input')
    x = conv2d_bn(x_input, 32, (3, 3), strides=(2, 2), padding='valid')# [149, 149, 32]
    x = conv2d_bn(x, 32, (3, 3), padding='valid') # [147, 147, 32]
    x = conv2d_bn(x, 64, (3, 3))  
    x = stem_maxpool(x, 96)     # [73, 73, 160]
    x = stem_conv(x)            # [71, 71, 192]
    x = stem_maxpool(x, 192)    # [35, 35, 384]
    print(x)
    
    # inception at 35x35
    for i in range(4):
        x = inception_35(x, [96, 64, 96, 64, 96, 96, 96]) 
    x = reduce_35(x)            # [17, 17, 1024]
    # inception at 17x17
    for i in range(7):
        x = inception_17(x, [384, 192, 224, 256, 192, 192, 224, 224, 256, 128]) 
    x = reduce_17(x)            # [8, 8, 1536]
    # inception at 8x8
    for i in range(3):
        x = inception_8(x) 
    print(x)
    
    # truck
    x = GlobalAveragePooling2D(name='g_avg_pool')(x) # [None, 1536]
    x = Dropout(0.2)(x)
    y_output = Dense(nclass, activation='softmax', name='y_output')(x)
    
    print(y_output)
    model = Model(x_input, y_output)
    # plot the model pictures
    if SAVE_MODEL_PLOT:
        plot_model(model, to_file='model_plot/' + SAVE_NAME, show_shapes=True)
        
    if debug:
        for i, layer in enumerate(model.layers):
            print("pt1:layer_{} name is {}".format(i, layer.name))
            
    return model
    

    
def stem_maxpool(base_tensor, filters=None ,name=None):
    tower_conv = conv2d_bn(base_tensor, filters, (3, 3), strides=(2, 2), padding='valid')
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
 
    
def inception_35(base_tensor, filters, name=None): 
    """  two 3x3 conv replaces a 5x5 conv, when the feature map is 35x35  """
    f1, f31, f33, f51, f53, f533, fp1 = filters
    # conv1
    tower_0 = conv2d_bn(base_tensor, f1, (1, 1))
    # conv3
    tower_1 = conv2d_bn(base_tensor, f31, (1, 1))
    tower_1 = conv2d_bn(tower_1, f33, (3, 3))
    # conv5
    tower_2 = conv2d_bn(base_tensor, f51, (1, 1))
    tower_2 = conv2d_bn(tower_2, f53, (3, 3))
    tower_2 = conv2d_bn(tower_2, f533, (3, 3))
    # maxpool
    tower_3 = AveragePooling2D((3, 3),(1, 1), padding='same')(base_tensor)
    tower_3 = conv2d_bn(tower_3, fp1, (1, 1))
    # merge
    output = Concatenate(axis=-1, name=name)([tower_0, tower_1, tower_2, tower_3])
    return output 
  
    
def inception_17(base_tensor, filters, name=None): 
    """  1x7 & 7x1 conv replaces a 3x3 or 5x5 conv, when the feature map is 17x17  """
    f1, f31, f317, f371, f51, f517, f571, f5_17, f5_71, fp1 = filters
    # conv1
    tower_0 = conv2d_bn(base_tensor, f1, (1, 1))
    # conv3
    tower_1 = conv2d_bn(base_tensor, f31, (1, 1))
    tower_1 = conv2d_bn(tower_1, f317, (1, 7))
    tower_1 = conv2d_bn(tower_1, f371, (7, 1))
    # conv5
    tower_2 = conv2d_bn(base_tensor, f51, (1, 1))
    tower_2 = conv2d_bn(tower_2, f517, (1, 7))
    tower_2 = conv2d_bn(tower_2, f571, (7, 1))
    tower_2 = conv2d_bn(tower_2, f5_17, (1, 7))
    tower_2 = conv2d_bn(tower_2, f5_71, (7, 1))
    # maxpool
    tower_3 = AveragePooling2D((3, 3),(1, 1), padding='same')(base_tensor)
    tower_3 = conv2d_bn(tower_3, fp1, (1, 1))
    # merge
    output = Concatenate(axis=-1, name=name)([tower_0, tower_1, tower_2, tower_3])
    return output    


def inception_8(base_tensor, name=None): 
    """  1x3 & 3x1 conv replaces a 3x3 conv parallelly, when the feature map is 8x8  """
    # conv1x1
    tower_0 = conv2d_bn(base_tensor, 256, (1, 1))
    # conv3x3
    tower_1 = conv2d_bn(base_tensor, 384, (1, 1))
    tower_1_1 = conv2d_bn(tower_1, 256, (1, 3))
    tower_1_2 = conv2d_bn(tower_1, 256, (3, 1))
    tower_1 = Concatenate(axis=-1)([tower_1_1, tower_1_2])
    # conv5
    tower_2 = conv2d_bn(base_tensor, 384, (1, 1))
    tower_2 = conv2d_bn(tower_2, 448, (1, 3))
    tower_2 = conv2d_bn(tower_2, 512, (3, 1))
    tower_2_1 = conv2d_bn(tower_2, 256, (1, 3))
    tower_2_2 = conv2d_bn(tower_2, 256, (3, 1))
    tower_2 = Concatenate(axis=-1)([tower_2_1, tower_2_2])
    # maxpool
    tower_3 = AveragePooling2D((3, 3),(1, 1), padding='same')(base_tensor)
    tower_3 = conv2d_bn(tower_3, 256, (1, 1))
    # merge
    output = Concatenate(axis=-1, name=name)([tower_0, tower_1, tower_2, tower_3])
    return output    


def reduce_35(base_tensor, name=None):
    """ reduce feature map dims 35 x 35 to 17 x 17  """
    # branch
    tower_0 = conv2d_bn(base_tensor, 384, (3, 3), strides=(2, 2), padding='valid')
    tower_1 = MaxPooling2D((3, 3), strides=(2, 2))(base_tensor)
    tower_2 = conv2d_bn(base_tensor, 192, (1, 1))
    tower_2 = conv2d_bn(tower_2, 224, (3, 3))
    tower_2 = conv2d_bn(tower_2, 256, (3, 3), strides=(2, 2), padding='valid')
    # merge
    output = Concatenate(axis=-1, name=name)([tower_0, tower_1, tower_2])
    return output


def reduce_17(base_tensor, name=None):
    """ reduce feature map dims 17 x 17 to 8 x 8  """
    # branch
    tower_0 = conv2d_bn(base_tensor, 192, (1, 1))
    tower_0 = conv2d_bn(base_tensor, 192, (3, 3), strides=(2, 2), padding='valid')
    tower_1 = MaxPooling2D((3, 3), strides=(2, 2))(base_tensor)
    tower_2 = conv2d_bn(base_tensor, 256, (1, 1))
    tower_2 = conv2d_bn(tower_2, 256, (1, 7))
    tower_2 = conv2d_bn(tower_2, 320, (7, 1))
    tower_2 = conv2d_bn(tower_2, 320, (3, 3), strides=(2, 2), padding='valid')
    # merge
    output = Concatenate(axis=-1, name=name)([tower_0, tower_1, tower_2])
    return output


if __name__ == "__main__":
    Inception_V4()

