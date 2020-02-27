# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:23:35 2019

@author: zhengqj
------------------------------------------------------------------
Architecture:
                    params=f*k*k*C + C , calc= params*N*N
    Conv1: 96, 11x11, strides=4, padding=X      [params:34,944, calc:105,705,600]
    mpool1: 3x3, strides=2                   
    Conv2: 256, 5x5, strides=1, padding='same'  [params:307,456, calc:224,135,424]
    mpool2: 3x3, strides=2
    Conv3: 384, 3x3, strides=1, padding='same'  [params:885,120, calc:149,585,280] # communicate
    Conv4: 384, 3x3, strides=1, padding='same'  [params:663,936, calc:112,205,184]
    Conv5: 256, 3x3, strides=1, padding='same'  [params:442,624, calc:74,803,456]
    FC1: 4096                                   [params:37,752,832, calc:37,752,832]
    FC2: 4096                                   [params:16,781,312, calc:16,781,312]
    FC3: 1000                                   [params:4,097,000, calc:4,097,000]
                    [params: 58M, calc:7.5e8 Flos]
------------------------------------------------------------------
"""
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from tensorflow.keras.utils import plot_model


def AlexNet(input_shape=None, nclass=1000, SAVE_NAME = 'Image_Alexnet.jpg'):
    if input_shape is None:
        input_shape = (224, 224, 3)
    
    x_input = KL.Input((input_shape), name='x_input')  # add zeropadding
    x = KL.ZeroPadding2D((2, 2))(x_input)
    x = KL.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv1')(x)   # outmap [55, 55, 96]
#    x = LRN()    
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), name='mpool1')(x)  # outmap [27, 27, 96]
    x = KL.Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv2')(x) # outmap [27, 27, 256]
#    x = LRN()  
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), name='mpool2')(x)  # outmap [13, 13, 256]
    x = KL.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3')(x) # outmap [13, 13, 384]
    x = KL.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv4')(x) # outmap [13, 13, 384]
    x = KL.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv5')(x) # outmap [13, 13, 256]
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), name='mpool3')(x)  # outmap [6, 6, 256]
    x = KL.Flatten(name='flatten')(x)
    x = KL.Dense(4096, activation='relu', name='FC1')(x)
    x = KL.Dropout(0.5, name='dropout1')(x)
    x = KL.Dense(4096, activation='relu', name='FC2')(x)
    x = KL.Dropout(0.5, name='dropout2')(x)
    y_output = KL.Dense(nclass, activation='softmax', name='y_output')(x)
    print(y_output)
    model = KM.Model(x_input, y_output)
    # plot model picture
    if SAVE_NAME:
        plot_model(model, to_file='network_visual/' + SAVE_NAME, show_shapes=True)
        model.summary()
    return model


# 怎么处理输入？
# 调用summary（）一直报模型未build的错误
class AlexNet(tf.keras.Model):
    def __init__(self, nclass=1000):
        super().__init__()
        self.nclass = nclass

    def __call__(self, inputs):
        x = KL.ZeroPadding2D((2, 2))(inputs)
        x = KL.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv1')(x)  # outmap [55, 55, 96]
        #    x = LRN()
        x = KL.MaxPooling2D((3, 3), strides=(2, 2), name='mpool1')(x)  # outmap [27, 27, 96]
        x = KL.Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv2')(
            x)  # outmap [27, 27, 256]
        #    x = LRN()
        x = KL.MaxPooling2D((3, 3), strides=(2, 2), name='mpool2')(x)  # outmap [13, 13, 256]
        x = KL.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3')(
            x)  # outmap [13, 13, 384]
        x = KL.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv4')(
            x)  # outmap [13, 13, 384]
        x = KL.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv5')(
            x)  # outmap [13, 13, 256]
        x = KL.MaxPooling2D((3, 3), strides=(2, 2), name='mpool3')(x)  # outmap [6, 6, 256]
        x = KL.Flatten(name='flatten')(x)
        x = KL.Dense(4096, activation='relu', name='FC1')(x)
        x = KL.Dropout(0.5, name='dropout1')(x)
        x = KL.Dense(4096, activation='relu', name='FC2')(x)
        x = KL.Dropout(0.5, name='dropout2')(x)
        y_output = KL.Dense(self.nclass, activation='softmax', name='y_output')(x)
        return y_output

def test_AlexNet():
    x = tf.random.normal((10, 24, 224, 3), name='x_input')
    alexnet = AlexNet(10)
    y = alexnet(x)
    print(y)
    alexnet.summary()


# if __name__ == "__main__":
#     # AlexNet()

