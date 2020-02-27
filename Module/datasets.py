# -*- coding: utf-8 -*-
"""
author: zhengqj@fotile.com
数据生成的三种方案：
1.keras自带的生成器；
2.构建字典 + tf.dataset + imgaug增强  (推荐！灵活，测试下代码运行效率！)
3.tf.dataset + imagaug
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageGen(object):
    """ 使用keras自带的类，从图片文件夹中构建数据生成器。怎么和tf.dataset结合
        后续数据增强的pipeline用imgaug在生成器上做。
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


import cv2
import tensorflow as tf
from Module.utils import DataSet
import imgaug.augmenters as iaa

class ImgPipe(DataSet):
    """ 用tf的dataset产生batch数据, 构建图片处理的pipeline
    """
    def __init__(self, img_dir, batch, img_size=224):
        super(ImgPipe, self).__init__(img_dir)
        self.img_size = img_size
        self.batch = batch
        self.classmap = self.img_dict['str2num']

    def pipeline(self, ismap=None):
        """ if ismap is None, should build pipeline after the batch
        """
        self.shuffle()
        images = self.img_dict['files']
        labels = self.img_dict['labels']

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(self._tf_parse_image)
        if ismap:
            dataset = dataset.map(self._py_imgaug)
        img_gen = dataset.batch(self.batch).repeat()
        return img_gen

    def _tf_parse_image(self, filename, label):
        """ 使用tf完成图片read，resize, 标签one-hod等基本处理 """
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.img_size, self.img_size])
        label = tf.one_hot(label, self.img_dict['nums'][1], on_value=0.9, off_value=0.02)
        return image, label

    def _py_imgaug(self, filename, label):
        """ 使用python函数,封装图片处理的pipeline """
        img = cv2.imread(filename)
        # 使用imaug构建图像增强pipeline
        def imgaug_seq(image):
            seq = iaa.Sequential([
                iaa.Affine(rotate=(-25, 25)),   # 仿射
                iaa.AdditiveGaussianNoise(scale=(0, 0.2)),  # 高斯噪声
                iaa.Fliplr(0.5)  # 随机翻转
            ])
            images_aug = seq.augment_image(image)
            return images_aug

        [image, ] = tf.py_function(imgaug_seq, [img], [tf.float32])
        image.set_shape(img.shape)
        return image, label


def imgaug_process(img_gen):
    """ 取出Dataset生成的batch数据，使用imgaug模块进行数据增强。
    """
    for image, label in img_gen:
        seq = iaa.Sequential([
            # iaa.Affine(rotate=(-25, 25)),  # 仿射
            # iaa.AdditiveGaussianNoise(scale=(0, 0.2)),  # 高斯噪声
            # iaa.Fliplr(0.5),  # 随机翻转
            # iaa.SigmoidContrast(gain=(4, 12), cutoff=(0.4, 0.7)) # 对比度
        ])
        image = image.numpy()
        images_aug = seq.augment_images(image)
        yield images_aug, label


import pathlib
class ImgPipe1(object):
    """ 使用tf.data直接从文件夹中读取图片, 和字典方案比较
        未收集数据集的大小信息，classmap信息
    """
    def __init__(self, root, batch):
        self.root = pathlib.Path(root)
        self.batch = batch

    def pipeline(self):
        list_ds = tf.data.Dataset.list_files(str(self.root / '*/*'))
        dataset = list_ds.map(self._tf_parse_image)
        img_gen = dataset.shuffle(50).batch(self.batch).repeat(1)
        return img_gen

    def _tf_parse_image(self, filename):
        parts = tf.strings.split(filename, '\\')
        label = parts[-2]

        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [128, 128])
        return image, label

__all__ = ['ImageGen', 'ImgPipe', 'ImgPipe1']


