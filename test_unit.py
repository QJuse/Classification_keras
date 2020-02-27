
import pytest
from Module.config import BuildConfig, LoadConfig
from Module.datasets import ImgPipe, ImgPipe1, ImageGen
from Module.models import CustomModel
import Module.utils as utils
from train import ModelTrain, run_pyfile

# 测试config接口
@pytest.mark.skip(reason="pass")
def test_buildconfig():
    network = 'mobilenet'
    version = 'mobilenet_t1'
    pre_weight = r"C:\QJ_WCode\Cproject\Classification\cls_module_keras\pre_weight\mobilenet.h5"
    BuildConfig(network, version, pre_weight)

@pytest.mark.skip(reason="pass")
def test_loadcfg():
    cfg_file = 'configs/vgg16_t1.json'
    config = LoadConfig(cfg_file)


# 测试dataset接口
@pytest.mark.skip(reason="pass")
def test_ImgPipe():
    import matplotlib.pyplot as plt
    def show(image, label):
        plt.figure()
        plt.imshow(image)
        plt.title(label.numpy())
        plt.axis('off')
        plt.show()
    root = r"C:\QJ_WCode\Cproject\Classification\cls_module_keras\dataset\dataset_20200230\trian"
    imgp = ImgPipe(root, 32)
    img_gen = imgp.pipeline()
    # test
    count = 0
    for image, label in img_gen:
        show(image[0], label[0])
        print(image[0])
        count += 1
        if count > 5:
            break

# 测试models接口
@pytest.mark.skip(reason="pass")
def test_CustomModel():
    network = 'vgg16'
    pre_weight = r"C:\QJ_WCode\Cproject\Classification\cls_module_keras\pre_weight\vgg16.h5"
    nclass = '5'
    Model = CustomModel(network, pre_weight, nclass)
    model = Model.build_model()
    model.summary()

# 测试train接口
# @pytest.mark.skip(reason="xx")
def test_train():
    run_pyfile()