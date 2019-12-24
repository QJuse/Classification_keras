# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:40:49 2019

@author: zhengqj@fotile.com
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageData(object):
    """  create data generator of batch from image_dirs
    returns:
        datagen: a generation, produce bacth image
    """
    def __init__(self, img_dir, batch, img_size=224, rescale=1./255):
        self.img_dir = img_dir
        self.batch = batch
        self.img_size = img_size
        self.data_generate(rescale)

    def data_generate(self, rescale=1./255):
        data_generator = ImageDataGenerator(rescale=rescale)
        self.datagen = data_generator.flow_from_directory(
                            self.img_dir,
                            target_size=(self.img_size, self.img_size),
                            batch_size=self.batch,
                            class_mode='categorical')
        self.classmap = self.datagen.class_indices




