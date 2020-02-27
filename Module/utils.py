# -*- coding: utf-8 -*-
"""
author: zhengqj@fotile.com
"""

import os
import random
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

class DataSet(object):
    """ 收集图片数据集信息，存储在定义的字典数据结构中 """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_dict = self._load_anns()

    def _load_anns(self):
        """ use a img_dict store anns"""
        file_dict = {'files': [], 'labels': [], 'str2num': {}}
        class_dir = os.listdir(self.data_dir)
        class_dir.sort()   # 按字母顺序排列类别

        for i, sub_dir in enumerate(class_dir):
            # 获得类别
            file_dict['str2num'][sub_dir] = i  # subdir name as classname:num
            sub_dir_path = os.path.join(self.data_dir, sub_dir)
            # 获得图片路径和标签
            if os.path.isdir(sub_dir_path):  # ignore file
                for file in os.listdir(sub_dir_path):
                    file_path = os.path.join(sub_dir_path, file)
                    file_dict['files'].append(file_path)
                    file_dict['labels'].append(i)
        # 记录数据集大小
        file_dict['nums'] = (len(file_dict['files']), len(file_dict['str2num']))
        return file_dict

    def shuffle(self):
        random.seed(42)
        random.shuffle(self.img_dict['files'])
        random.seed(42)
        random.shuffle(self.img_dict['labels'])

    def calc_mean_std(self):
        """ 计算图片集的均值和方差 """
        arr = []
        for file in self.img_dict['files']:
            img = cv2.imread(file)
            arr.append(np.expand_dims(img, axis=0))
        arr = np.vstack(arr)
        means, stds = [], []
        for i in range(3):
            pixel = arr[..., i]
            means.append(np.mean(pixel))
            stds.append(np.std(pixel))
        means.reverse()
        stds.reverse()
        self.img_dict['means'] = means
        self.img_dict['stds'] = stds


    def split_data(self, target, small=50):
        """ 分层选择固定比例的标签数据，同时拷贝成两份 """
        os.makedirs(target, exist_ok=True)
        os.makedirs(target + '_L', exist_ok=True)

        for sdir in os.listdir(self.data_dir):
            # sub dir path of target
            new_pdir = os.path.join(target, sdir)
            os.makedirs(new_pdir, exist_ok=True)
            new_pdir_L = os.path.join(target + '_L', sdir)
            os.makedirs(new_pdir_L, exist_ok=True)
            # random choice sample
            pdir = os.path.join(self.data_dir, sdir)
            slice = random.sample(os.listdir(pdir), small)
            other = [i for i in os.listdir(pdir) if i not in slice]
            # copy file
            for file in slice:
                pfile = os.path.join(pdir, file)  # old file path
                new_pfile = os.path.join(new_pdir, file)  # new file path
                open(new_pfile, 'wb').write(open(pfile, 'rb').read())
            for file in other:
                pfile = os.path.join(pdir, file)  # old file path
                new_pfile = os.path.join(new_pdir_L, file)  # new file path
                open(new_pfile, 'wb').write(open(pfile, 'rb').read())

# 绘制模型训练结果
def plot_loss(history, plotacc=False):
    """  plot the accuracy on train/valid
    """
    # 绘制训练/验证集的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    if plotacc:
        # 绘制训练/验证集准确率
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

# 绘制模型图片
def save_plot_model(model):
    from tensorflow.keras.utils import plot_model  # save model as a picture
    # depend on pydot and graphviz  developer env
    SAVE_NAME = 'inception_resnet_v2.jpg'
    plot_model(model, to_file='lib/' + SAVE_NAME, show_shapes=True)  # keras 模型对象

# 检查模型是否正确
def check_model(model):
    model.summary()

# 将字典按指定格式保存成json文件
def save_json(var, path):
    with open(path, 'w+') as f:
        json.dump(var, f, indent=4, separators=(",", ":"), ensure_ascii=True)

