# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:03:49 2019

@author: zhengqj@fotile.com
"""


import os
import os.path as osp
import json
import sys

class Config(object):
    def __init__(self, network, pre_weight, version, config_file=None):
        self.config = {'data':{}, 'model':{}, 'params':{}}
        if config_file:
            self.load_config(config_file)
            self.parsar_config()
        else:
            print("Config.build_config() to create the file")
            self.build_config(network, pre_weight, version)
            self.parsar_config()

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)

    def build_config(self, network, pre_weight, version):
        """
        network: one of keras models, eg. vgg16
        version: the version of tuned model, eg. vgg16_1.0
        pre_weight: vgg16.h5
        """
        # TODO 去除root,采用相对路径，保证可移植性; 所有\\转为/
        # data infos
        for key in ['train', 'valid', 'test']:
            self.config['data'][key] = osp.join('dataset', key)
        # model infos
        path_version = osp.join('out_model', version)
        os.makedirs(path_version, exist_ok=True)  # 不用检查文件夹是否已存在
        self.config['model'] = {
            'network': network,
            'pre_weight': pre_weight,
            'version': version,
            'nclass': 5,
            'image_size': 224,
            'output': {
                'classmap': osp.join(path_version, 'classmap.json'),
                'tuned_model': osp.join(path_version, 'tuned.h5'),
                'tuned_model_pb': osp.join(path_version, 'export_pb'),
                'tb_logs': osp.join(path_version, 'logs'),
            }
        }
        ckpt_path = os.path.join(path_version, 'checkpoint')
        os.makedirs(ckpt_path, exist_ok=True)
        ckpt_name = 'model_ex-{epoch:03d}_acc-{accuracy:03f}.h5'
        self.config['model']['output']['checkpoint'] = osp.join(ckpt_path, ckpt_name)
        # params infos
        self.config['params'] = {
            'init_lr': 1e-3,
            'epochs': 20,
            'batch_size': 64,
            'include_top': True,
            'ckpt_interval': 5,
        }
        # save config to json file
        config_file = osp.join('configs', '{}.json'.format(version))
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=4, separators=(",", ":"), ensure_ascii=True)

    def parsar_config(self):
        # TODO 模型传递参数的方式很丑陋, 想改动config名称全得改

        self.train = self.config['data']['train'].replace('\\', '/')
        self.valid = self.config['data']['valid'].replace('\\', '/')
        self.test = self.config['data']['test'].replace('\\', '/')
        # model
        self.network = self.config['model']['network']
        self.pre_weight = self.config['model']['pre_weight'].replace('\\', '/')
        self.nclass = self.config['model']['nclass']
        self.image_size = self.config['model']['image_size']
        # output
        self.checkpoint = self.config['model']['output']['checkpoint'].replace('\\', '/')
        self.tb_logs = self.config['model']['output']['tb_logs'].replace('\\', '/')
        self.classmap = self.config['model']['output']['classmap'].replace('\\', '/')
        self.tuned_model = self.config['model']['output']['tuned_model'].replace('\\', '/')
        self.tuned_model_pb = self.config['model']['output']['tuned_model_pb'].replace('\\', '/')
        # train
        self.init_lr = self.config['params']['init_lr']
        self.epochs = self.config['params']['epochs']
        self.batch_size = self.config['params']['batch_size']
        self.include_top = self.config['params']['include_top']
        self.ckpt_interval = self.config['params']['ckpt_interval']





