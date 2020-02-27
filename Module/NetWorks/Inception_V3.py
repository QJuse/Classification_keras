# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:13:12 2019

@author: zhengqj

Inception_V3: LSVRC-ImageNet
-----------------------------------------------------------------
feature:
    Inception: large Conv factorization to samller Conv
    struction: at 35x35 size use two 3x3 conv, split a 5x5 conv 
               at 17x17 size use 1x7 & 7x1 conv, split a large conv
               
               not split too early, best when feature map is 12~20
               when decrease feature_map size, increase channels
    
------------------------------------------------------------------
Architecture:  ignore bias 
    
------------------------------------------------------------------
"""

from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Activation, Concatenate
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model

SAVE_MODEL_PLOT = True
if SAVE_MODEL_PLOT:
    from tensorflow.keras.utils import plot_model
    SAVE_NAME = 'Image_inception_V3.jpg'

def conv2d_bn(x, filters, kernel, padding='same', strides=(1, 1),name=None):
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


def Inception_V3(input_shape=None, nclass=1000, debug=True):
    if input_shape is None:
        input_shape = (299, 299, 3)

    x_input = Input(input_shape, name='x_input')
    x = conv2d_bn(x_input, 32, (3, 3), strides=(2, 2), padding='valid', name='C1')# outmap [149, 149, 32]
    x = conv2d_bn(x, 32, (3, 3), padding='valid', name='C2') # outmap [147, 147, 32]
    x = conv2d_bn(x, 64, (3, 3), name='C3')  
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)  # outmap [73, 73, 64]
    
    x = conv2d_bn(x, 80, (3, 3), name='C4')   # outmap [73, 73, 80]
    x = conv2d_bn(x, 192, (3, 3), padding='valid', name='C5') # outmap [71, 71, 192]
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool2')(x) # outmap [35, 35, 192]

    # incption block_3   
    x = inception_struct_a(x, [64, 48, 64, 64, 96, 96, 32], name='block_3a')
    x = inception_struct_a(x, [64, 48, 64, 64, 96, 96, 64], name='block_3b')
    x = inception_struct_a(x, [64, 48, 64, 64, 96, 96, 64], name='block_3c')  # outmap [35, 35, 288]

    # incption block_4
    x = reduce_35_17(x, name='reduce_a')   # outmap [17, 17, 768]
    x = inception_struct_b(x, [192, 128, 128, 192, 128, 128, 128, 128, 192, 192], name='block_4a')

    x = inception_struct_b(x, [192, 160, 160, 192, 160, 160, 160, 160, 192, 192], name='block_4b')
    x = inception_struct_b(x, [192, 160, 160, 192, 160, 160, 160, 160, 192, 192], name='block_4c')
    x = inception_struct_b(x, [192, 192, 192, 192, 192, 192, 192, 192, 192, 192], name='block_4d')
    # incption block_5
    x = reduce_17_8(x, name='reduce_b')   # outmap [8, 8, 1280]
    x = inception_struct_c(x, name='block_5a')  # outmap [8, 8, 2048]
    x = inception_struct_c(x, name='block_5b')   # outmap [8, 8, 2048]
    
    x = GlobalAveragePooling2D(name='g_avg_pool')(x)  # outmap [None,2048]
    x = Dropout(0.4, name='droput')(x)
    y_output = Dense(nclass, activation='softmax', name='y_output')(x)
    
    print(y_output)
    model = Model(x_input, y_output)
    
    if SAVE_MODEL_PLOT:
        plot_model(model, to_file='model_plot/' + SAVE_NAME, show_shapes=True)
    if debug:
        for i, layer in enumerate(model.layers):
            print("pt1:layer_{} name is {}".format(i, layer.name))
    return model


def inception_struct_a(base_tensor, filters, name=None): 
    """  two 3x3 conv replaces a 5x5 conv, when the feature map is 35x35  """
    f1, f31, f33, f51, f53, f533, fp1 = filters
    # conv1
    tower_0 = conv2d_bn(base_tensor, f1, (1, 1), name=name + '_c1')
    # conv3
    tower_1 = conv2d_bn(base_tensor, f31, (1, 1), name=name + '_c31')
    tower_1 = conv2d_bn(tower_1, f33, (3, 3), name=name + '_c33')
    # conv5
    tower_2 = conv2d_bn(base_tensor, f51, (1, 1), name=name + '_c51')
    tower_2 = conv2d_bn(tower_2, f53, (3, 3),  name=name + '_c53')
    tower_2 = conv2d_bn(tower_2, f533, (3, 3),  name=name + '_c533')
    # maxpool
    tower_3 = AveragePooling2D((3, 3),(1, 1), padding='same', name=name + '_pool')(base_tensor)
    tower_3 = conv2d_bn(tower_3, fp1, (1, 1), name=name + '_pool_c1')
    # merge
    output = Concatenate(axis=-1, name=name + '_mixed')([tower_0, tower_1, tower_2, tower_3])
    return output


def inception_struct_b(base_tensor, filters, name=None): 
    """  1x7 & 7x1 conv replaces a 3x3 or 5x5 conv, when the feature map is 17x17  """
    f1, f31, f317, f371, f51, f517, f571, f5_17, f5_71, fp1 = filters
    # conv1
    tower_0 = conv2d_bn(base_tensor, f1, (1, 1), name=name + '_c1')
    # conv3
    tower_1 = conv2d_bn(base_tensor, f31, (1, 1), name=name + '_c31')
    tower_1 = conv2d_bn(tower_1, f317, (1, 7), name=name + '_c317')
    tower_1 = conv2d_bn(tower_1, f371, (7, 1), name=name + '_c371')
    # conv5
    tower_2 = conv2d_bn(base_tensor, f51, (1, 1), name=name + '_c51')
    tower_2 = conv2d_bn(tower_2, f517, (1, 7),  name=name + '_c517')
    tower_2 = conv2d_bn(tower_2, f571, (7, 1),  name=name + '_c571')
    tower_2 = conv2d_bn(tower_2, f5_17, (1, 7),  name=name + '_c5_17')
    tower_2 = conv2d_bn(tower_2, f5_71, (7, 1),  name=name + '_c5_71')
    # maxpool
    tower_3 = AveragePooling2D((3, 3),(1, 1), padding='same', name=name + '_pool')(base_tensor)
    tower_3 = conv2d_bn(tower_3, fp1, (1, 1), name=name + '_pool_c1')
    # merge
    output = Concatenate(axis=-1, name=name + '_mixed')([tower_0, tower_1, tower_2, tower_3])
    return output


def inception_struct_c(base_tensor, name=None): 
    """  1x3 & 3x1 conv replaces a 3x3 conv parallelly, when the feature map is 8x8  """
    # conv1x1
    tower_0 = conv2d_bn(base_tensor, 320, (1, 1), name=name + '_c1')
    # conv3x3
    tower_1 = conv2d_bn(base_tensor, 384, (1, 1), name=name + '_c31')
    tower_1_1 = conv2d_bn(tower_1, 384, (1, 3), name=name + '_c313')
    tower_1_2 = conv2d_bn(tower_1, 384, (3, 1), name=name + '_c331')
    tower_1 = Concatenate(axis=-1, name=name + '_t1_mixed')([tower_1_1, tower_1_2])
    # conv5
    tower_2 = conv2d_bn(base_tensor, 448, (1, 1), name=name + '_c51')
    tower_2 = conv2d_bn(tower_2, 448, (3, 3),  name=name + '_c53')
    tower_2_1 = conv2d_bn(tower_2, 384, (1, 3),  name=name + '_c571')
    tower_2_2 = conv2d_bn(tower_2, 384, (3, 1),  name=name + '_c5_17')
    tower_2 = Concatenate(axis=-1, name=name + '_t2_mixed')([tower_2_1, tower_2_2])
    # maxpool
    tower_3 = AveragePooling2D((3, 3),(1, 1), padding='same', name=name + '_pool')(base_tensor)
    tower_3 = conv2d_bn(tower_3, 192, (1, 1), name=name + '_pool_c1')
    # merge
    output = Concatenate(axis=-1, name=name + '_mixed')([tower_0, tower_1, tower_2, tower_3])
    return output    

    
def reduce_35_17(base_tensor, name=None):
    """ reduce feature map dims 35 x 35 to 17 x 17  """
    # branch
    tower_0 = conv2d_bn(base_tensor, 384, (3, 3), strides=(2, 2), padding='valid', name=name + '_c1')
    tower_1 = MaxPooling2D((3, 3), strides=(2, 2), name=name + '_pool')(base_tensor)
    tower_2 = conv2d_bn(base_tensor, 64, (1, 1), name=name + '_c31')
    tower_2 = conv2d_bn(tower_2, 96, (3, 3),  name=name + '_c33')
    tower_2 = conv2d_bn(tower_2, 96, (3, 3), strides=(2, 2), padding='valid', name=name + '_c333')
    # merge
    output = Concatenate(axis=-1, name=name + '_mixed')([tower_0, tower_1, tower_2])
    return output


def reduce_17_8(base_tensor, name=None):
    """ reduce feature map dims 17 x 17 to 8 x 8  """
    # branch
    tower_0 = conv2d_bn(base_tensor, 192, (1, 1), name=name + '_c1')
    tower_0 = conv2d_bn(base_tensor, 320, (3, 3), strides=(2, 2), padding='valid', name=name + '_c2')
    tower_1 = MaxPooling2D((3, 3), strides=(2, 2), name=name + '_pool')(base_tensor)
    tower_2 = conv2d_bn(base_tensor, 192, (1, 1), name=name + '_c31')
    tower_2 = conv2d_bn(tower_2, 192, (1, 7),  name=name + '_c317')
    tower_2 = conv2d_bn(tower_2, 192, (7, 1),  name=name + '_c371')
    tower_2 = conv2d_bn(tower_2, 192, (3, 3), strides=(2, 2), padding='valid', name=name + '_c333')
    # merge
    output = Concatenate(axis=-1, name=name + '_mixed')([tower_0, tower_1, tower_2])
    return output
  
    
if __name__ == "__main__":
    Inception_V3()
