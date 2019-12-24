### 主要功能

使用keras提供的预训练模型进行图片的分类，尝试的模型如下：

| 模型名称    | 大小 | top1误差 | 模型名称     | 大小 | top1误差 |
| ----------- | ---- | -------- | ------------ | ---- | -------- |
| VGG16       |      |          | MobileNet    |      |          |
| ResNet50    |      |          | MobileNetV2  |      |          |
| ResNet101   |      |          | NASNetMobile |      |          |
| DenseNet121 |      |          |              |      |          |
| InceptionV3 |      |          |              |      |          |

主要的目的：

* 对同一数据集，可快速的使用各模型进行训练，比较性能，可用于选择合适的模型。
* 提供特定数据集训练后的各个模型，进行tflite和SavedModel转换，打通模型部署的流程。

#### 使用环境

>python = 3.6.4
>tensorflow = 2.0.0

#### 目录结构

根目录下的configs用于存放”训练过的或即将训练的“模型的配置文件。out_model目录存储不同版本的模型的输出，包括训练好的h5文件，checkpoints，label文件和tensorboard日志等，train.py文件用于训练。

### 使用示例

* 进入根目录，添加数据集和预训练模型
  * 添加dataset目录，目录下包含train/valid/test三个文件夹，存放图片。
  * 添加pre_weight目录，目录下包含下载的keras预训练的权值文件。文件名称需要改成同模型名称的小写一致。

* 在根目录运行train.py文件

  ```python
  python train.py -n vgg16 -v vgg16_b32_lr0.001 -w 'pre_weight/vgg16' -wf 'train'
  ```

  其中 -n：参数代表模型名称，和keras提供的模型名称的小写一致。

  ​         -v：自定义的版本号，用于区分同一模型不同版本的训练。

  ​        -w：keras预训练的权值文件， -wf：选择工作流是train或evaluate

  【改进点】如果需要重新配置参数，需要打开config文件对参数进行修改！

### 功能特性

#### Feature

* 代码模块化，Module包中构建config，datasets, models三个类，工具代码在utils中。
* 在keras训练中添加：checkpoint回调，学习率规划，tensorboard回调
* 提供命令行调用，自动处理不同版本，绘制误差曲线

#### debug信息

* config配置文件

  * 相关路径的保存从\\\改成/，保证在linux系统上可运行
  * 相关路径采用相对路径，保证不同平台均可跑

* 增加pytest单元测试，在pycharm中直接调试

  * 用于复杂程序调通代码

  

#### TODO

* 完成单图片推理代码打印结果，测试集AP和混淆矩阵评估  （2）

* **改写dataloader不使用keras提供的API，集成imgaug做图片预处理**

* 将模型转换成tflite格式，并在树莓派等嵌入端使用C++部署

* **将模型转换成SavedModel格式，使用tf-serving在服务器上部署**

  * 在带GPU的服务器上使用tensorrt优化模型，完成部署

  

### 版本信息

