# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:17:01 2019

@author: zhengqj

GoogLeNet: LSVRC-ImageNet
-----------------------------------------------------------------
feature:
    Inception: large Conv factorization to samller Conv
    struction: 1x1, #1x1, 3x3, #1x1, 5x5, avgpool, 1x1 
    Conv use bias or not ?
    
------------------------------------------------------------------
Architecture:  ignore bias 
    Conv1: 64, 7x7, strides=2,     [params:9,472, calc:118,013,952]
    mpool1: 3x3, strides=2                   
    Conv2: 192, 3x3, strides=1,    [params:114,944, calc:360,464,384]
    mpool2: 3x3, strides=2
        params=C*f1+(C+k3*k3*f3_3)*f3_1+(C+k5*k5*f5_5)*f5_1+C*fp_1, calc= params*N*N  
    Conv3a: 256,                [params:163,328] 
    Conv3b: 480,                [params:388,096]   [calc:432,316,416]
    mpool3: 3x3, strides=2
    Conv4a: 512,                [params:375,552]
    Conv4b: 512,                [params:448,512]
    Conv4c: 512,                [params:509,440]
    Conv4d: 528,                [params:604,672]
    Conv4e: 832,                [params:867,328]   [calc:549,878,784]
    mpool4: 3x3, strides=2
    Conv5a: 832,                [params:1,042,432]
    Conv5b: 1024,               [params:1,442,816] [calc:121,777,152]
    avgpool: 7x7, stride=1
    FC1: 1000                   [params:1,024,000, calc:1,024,000]
        params=6,990,592=6.7M  calc=1,583,474,688=15.8e8 flos
------------------------------------------------------------------

"""
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, AveragePooling2D
from tensorflow.keras.models import Model

SAVE_MODEL_PLOT = True
if SAVE_MODEL_PLOT:
    from tensorflow.keras.utils import plot_model
    SAVE_NAME = 'Image_GoogLeNet.jpg'


def calc_params(C, filters):
    f1, f3_1, f3_3, f5_1, f5_5, fp_1 = filters
    params = C*f1 + (C + 9*f3_3)*f3_1 + (C + 25*f5_5)*f5_1 + C*fp_1
    return params

def calc_incpt_param():
    calc_I3a = calc_params(192, filters=[64, 96, 128, 16, 32, 32])
    calc_I3b = calc_params(256, filters=[128, 128, 192, 32, 96, 64]) # map=28
    sum_I3 = calc_I3a + calc_I3b
    calc_I4a = calc_params(480, filters=[192, 96, 208, 16, 48, 64])
    calc_I4b = calc_params(512, filters=[160, 112, 224, 24, 64, 64])
    calc_I4c = calc_params(512, filters=[128, 128, 256, 24, 64, 64])
    calc_I4d = calc_params(512, filters=[112, 144, 288, 32, 64, 64])
    calc_I4e = calc_params(528, filters=[256, 160, 320, 32, 128, 128])  # map=14
    sum_I4 = calc_I4a + calc_I4b + calc_I4c + calc_I4d + calc_I4e
    calc_I5a = calc_params(832, filters=[256, 160, 320, 32, 128, 128])  
    calc_I5b = calc_params(832, filters=[384, 192, 384, 48, 128, 128])  # map=7
    sum_I5 = calc_I5a + calc_I5b
    return sum_I3, sum_I4, sum_I5


def inception(input_tensor, filters, name='incpt'):
    f1, f3_1, f3_3, f5_1, f5_5, fp_1 = filters
    # conv1
    tower_0 = Conv2D(f1, (1, 1), activation='relu', padding='same', name=name + '_t0_conv1')(input_tensor)
    # conv3
    tower_1 = Conv2D(f3_1, (1, 1), activation='relu', padding='same', name=name + '_t1_conv1')(input_tensor)
    tower_1 = Conv2D(f3_3, (3, 3), activation='relu', padding='same', name=name + '_t1_conv3')(tower_1)
    # conv5
    tower_2 = Conv2D(f5_1, (1, 1), activation='relu', padding='same', name=name + '_t2_conv1')(input_tensor)
    tower_2 = Conv2D(f5_5, (5, 5), activation='relu', padding='same', name=name + '_t2_conv5')(tower_2)
    # maxpool
    tower_3 = MaxPooling2D((3, 3),(1, 1), padding='same', name=name + '_t3_pool')(input_tensor)
    tower_3 = Conv2D(fp_1, (1, 1), activation='relu', padding='same', name=name + '_t3_conv1')(tower_3)
    # merge
    output = Concatenate(axis=-1, name=name + '_concat')([tower_0, tower_1, tower_2, tower_3])
    return output


def googlenet(input_shape=None, nclass=1000, debug=True):
    if input_shape is None:
        input_shape = (224, 224, 3)
    
    x_input = Input(input_shape, name='x_input')
    x = Conv2D(64, (7, 7), strides=(2, 2), activation='relu', padding='same', name='conv1')(x_input) # outmap [112, 112, 64]
    x = MaxPooling2D((3, 3), (2, 2), padding='same', name='pool1')(x)  #outmap [56, 56, 64]
#    x = LRN()
    x = Conv2D(64, (1, 1), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
#    x = LRN()
    x = MaxPooling2D((3, 3), (2, 2), padding='same', name='pool2')(x)  # outmap [28, 28, 192]
    # inception a
    x = inception(x, filters=[64, 96, 128, 16, 32, 32], name='incpt_3a')
    # inception b
    x = inception(x, filters=[128, 128, 192, 32, 96, 64], name='incpt_3b')
    x = MaxPooling2D((3, 3), (2, 2), padding='same', name='pool3')(x)  # outmap [14, 14, 480]

    # inception 4a
    x = inception(x, filters=[192, 96, 208, 16, 48, 64], name='incpt_4a')
    # inception 4b
    x = inception(x, filters=[160, 112, 224, 24, 64, 64], name='incpt_4b')
    # inception 4c
    x = inception(x, filters=[128, 128, 256, 24, 64, 64], name='incpt_4c')
    # inception 4d
    x = inception(x, filters=[112, 144, 288, 32, 64, 64], name='incpt_4d')
    # inception 4e
    x = inception(x, filters=[256, 160, 320, 32, 128, 128], name='incpt_4e')
    x = MaxPooling2D((3, 3), (2, 2), padding='same', name='pool4')(x)  # outmap [7, 7, 832]
    # inception 5a
    x = inception(x, filters=[256, 160, 320, 32, 128, 128], name='incpt_5a')
    # inception 5b
    x = inception(x, filters=[384, 192, 384, 48, 128, 128], name='incpt_5b')
    x = AveragePooling2D((7, 7), (1, 1), name='avgpool')(x)
    print(x)
    x = Dropout(0.4)(x)
    y_output = Dense(nclass, activation='softmax', name='y_output')(x)
    
    print(y_output)
    model = Model(x_input, y_output)
    # plot model picture
    if SAVE_MODEL_PLOT:
        plot_model(model, to_file='model_plot/' + SAVE_NAME, show_shapes=True)
        
    if debug:
        for layer in model.layers:
            print("pt1:",layer.name)
    return model
    
    
if __name__ == "__main__":
    googlenet()