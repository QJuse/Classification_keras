# -*- coding: utf-8 -*-
"""
author: zhengqj@fotile.com
"""


import os
import os.path as osp
import json
from Module.utils import save_json


class BuildConfig(object):
    def __init__(self, network, version, pre_weight):
        """ config结构设计好后，就不做改变 """
        self.config = {'params':{}, 'model':{}}
        self._build_config(network, version, pre_weight)

    def _build_config(self, network, version, pre_weight):
        """
        network: one of keras models, eg. vgg16
        version: the version of tuned model, eg. vgg16_1.0
        pre_weight: vgg16.h5
        """
        path_version = osp.join('out_model', version)
        os.makedirs(path_version, exist_ok=True)   # 创建模型版本目录
        self.config['model'] = {
            'network': network,
            'version': version,
            'pre_weight': pre_weight,
            'data':{
                'train': osp.join('dataset', 'train'),
                'valid': osp.join('dataset', 'valid'),
                'test': osp.join('dataset', 'test'),
                    },
            'output': {
                'classmap': osp.join(path_version, 'classmap.json'),
                'tuned_model': osp.join(path_version, 'tuned.h5'),
                'tuned_model_pb': osp.join(path_version, 'export_pb'),
                'tb_logs': osp.join(path_version, 'logs'),
                    }
        }
        # add checkpoint infos
        ckpt_path = os.path.join(path_version, 'checkpoint')
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(osp.join(path_version, 'export_pb'), exist_ok=True)
        os.makedirs(osp.join(path_version, 'logs'), exist_ok=True)
        ckpt_name = 'model_ex-{epoch:03d}_acc-{accuracy:03f}.h5'
        self.config['model']['output']['checkpoint'] = osp.join(ckpt_path, ckpt_name)
        # params infos
        self.config['params'] = {
            'nclass': 5,
            'image_size': 224,
            'init_lr': 0.001,
            'epochs': 50,
            'batch_size': 64,
            'include_top': True,
            'ckpt_interval': 10,
        }
        # save config to json file
        config_file = osp.join('configs', '{}.json'.format(version))
        save_json(self.config, config_file)


class LoadConfig():
    def __init__(self, cfg_file):
        with open(cfg_file, 'r') as f:
            self.cfg = json.load(f)
        self._parsar_config()

    def _parsar_config(self):
        # TODO linux和windos相对路径统一, 所有路径针对根目录而言。
        self.network = self.cfg['model']['network']
        self.pre_weight = self.cfg['model']['pre_weight'].replace('\\', '/')
        # data
        self.train = self.cfg['model']['data']['train'].replace('\\', '/')
        self.valid = self.cfg['model']['data']['valid'].replace('\\', '/')
        self.test = self.cfg['model']['data']['test'].replace('\\', '/')
        # output
        self.checkpoint = self.cfg['model']['output']['checkpoint'].replace('\\', '/')
        self.tb_logs = self.cfg['model']['output']['tb_logs'].replace('\\', '/')
        self.classmap = self.cfg['model']['output']['classmap'].replace('\\', '/')
        self.tuned_model = self.cfg['model']['output']['tuned_model'].replace('\\', '/')
        self.tuned_model_pb = self.cfg['model']['output']['tuned_model_pb'].replace('\\', '/')
        # train params
        self.nclass = self.cfg['params']['nclass']
        self.image_size = self.cfg['params']['image_size']
        self.init_lr = self.cfg['params']['init_lr']
        self.epochs = self.cfg['params']['epochs']
        self.batch_size = self.cfg['params']['batch_size']
        self.include_top = self.cfg['params']['include_top']
        self.ckpt_interval = self.cfg['params']['ckpt_interval']


__all__ = ['BuildConfig', 'LoadConfig']
