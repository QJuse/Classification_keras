import os.path as osp
from Module.config import Config
from Module.models import CustomModel
from Module.datasets import ImageData
from Module.utils import plot_loss
from train import fit, evaluate

import pytest
@pytest.mark.skip(reason="跳过")
def test_Config():
    cfg_f = osp.join('configs', 'mobilenet_v1_1.0.json')
    cfg = Config(cfg_f)
    print(cfg.nclass, cfg.network)

@pytest.mark.skip(reason="跳过")
def test_Config_build():
    network = 'nasnet'
    version = 'nasnet_v0'
    pre_weight = 'pre_weight/nasnet.h5'
    cfg = Config()
    cfg.build_config(network, pre_weight, version)

@pytest.mark.skip(reason="跳过")
def test_CustomModel():
    network = 'nasnet_mobile'
    pre_weights = 'pre_weight/nasnet.h5'
    nclass = 5
    Model = CustomModel(network, pre_weights, nclass)
    model = Model.build_model()
    print(model.output)

@pytest.mark.skip(reason="跳过")
def test_ImageData():
    img_d = r'C:\Users\qjbook\Pictures\image_s_2\dataset_cls\train'
    batch = 5
    img_s = 224
    img_train = ImageData(img_d, batch, img_s)
    print(img_train.classmap)
    print(img_train.datagen.next()[0].shape)
    assert img_train.datagen.next()[0].shape[0] == batch

#@pytest.mark.skip(reason="跳过")
def test_train():
    workflow = 'train'
    config_file = 'configs/vgg16_v0.json'
    cfg = Config(config_file)
    if workflow == 'train':
        history = fit(cfg)
        # plot err
        plot_loss(history)

    if workflow == 'evaluate':
        metric = evaluate(cfg)
        print(metric)