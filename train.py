
import tensorflow as tf
assert tf.__version__.startswith('2'), 'please use tensorflow 2.0'

import tensorflow.keras.optimizers as KO
import tensorflow.keras.callbacks as KC
import tensorflow.keras.models as KM

from Module.config import LoadConfig
from Module.datasets import ImgPipe, ImgPipe1, imgaug_process
from Module.models import CustomModel
import Module.utils as utils

class ModelTrain():
    def __init__(self, config):
        self.config = config

    def fit(self):
        # model
        Model = CustomModel(self.config.network,
                            self.config.pre_weight,
                            self.config.nclass)
        model = Model.build_model()
        optimizer = KO.Adam(lr=self.config.init_lr, decay=1e-4)  # optimizer args
        lr_metric = self._metrics_get_lr(optimizer)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy', lr_metric])
        # data pipeline
        img_train = ImgPipe(self.config.train,
                            self.config.batch_size,
                            self.config.image_size)
        img_train_gen = img_train.pipeline()
        ## img_train_gen = imgaug_process(img_train_gen)
        utils.save_json(img_train.classmap, self.config.classmap)  # 保存类别映射
        img_valid = ImgPipe(self.config.valid,
                            self.config.batch_size,
                            self.config.image_size)
        img_valid_gen = img_valid.pipeline()
        # fit model
        num_train = img_train.img_dict['nums'][0]
        print("Number of experiments (Epochs) : ", self.config.epochs)
        history = model.fit_generator(
            generator=img_train_gen,
            #steps_per_epoch=int(num_train/self.config.batch_size),
            steps_per_epoch=2,   # 测试时使用
            epochs=self.config.epochs,
            validation_data=img_valid_gen,
            validation_steps=1,
            callbacks=self.callbacks())
        # save model
        model.save(self.config.tuned_model)
        tf.saved_model.save(model, self.config.tuned_model_pb)

        return history

    def evaluate(self):
        img_test = ImgPipe(self.config.test,
                           self.config.batch_size,
                           self.config.image_size)
        img_test_gen = img_test.pipeline()
        # img_test_gen = imgaug_seq(img_test_gen)
        num_test = img_test.img_dict['nums'][0]
        model = KM.load_model(self.config.tuned_model)
        history = model.evaluate_generator(
            generator=img_test_gen,
            steps=int(num_test/self.config.batch_size),
            # steps=4, # 测试时使用
            max_queue_size=10,
            workers=1,
            use_multiprocessing=True,   # debug
            verbose=1)
        return history

    def callbacks(self):
        """ set callbacks when model fit """
        checkpoint = KC.ModelCheckpoint(
            filepath=self.config.checkpoint,
            monitor='val_loss',
            verbose=1,
            save_weights_only=False,
            period=self.config.ckpt_interval)

        def lr_schedule(epoch):
            lr = self.config.init_lr
            total_epochs = self.config.epochs
            check_1 = int(total_epochs * 0.3)
            check_2 = int(total_epochs * 0.6)
            check_3 = int(total_epochs * 0.8)
            if epoch > check_1:
                lr *= 3e-1
            if epoch > check_2:
                lr *= 3e-1
            if epoch > check_3:
                lr *= 3e-1
            return lr
        lr_scheduler = KC.LearningRateScheduler(lr_schedule)

        tensorboard = KC.TensorBoard(
            log_dir=self.config.tb_logs,
            batch_size=self.config.batch_size,
            histogram_freq=0,  # validation data can not be generator
            write_graph=False,
            write_grads=False)
        callbacks = [checkpoint, lr_scheduler, tensorboard]   # TODO bug：Failed to create a directory: out_model/mobilenet_t1/logs\train;
        return callbacks

    def _metrics_get_lr(self, optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr


def run_terminal():
    # 设置GPU占用量
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # 命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='train/evaluate CNN model')
    parser.add_argument(dest='filenames', metavar='filename', nargs='*')
    parser.add_argument('-cfg', '--configfile', dest='configfile', action='store',
                        help='config file for model evlauate')
    parser.add_argument('-wf', '--workflow', dest='workflow', action='store',
                    choices={'train', 'evaluate'}, default='train',
                    help='model work in train or evaluate mode')
    args = parser.parse_args()
    # 模型训练-评估
    config = LoadConfig(args.configfile)
    modelpipe = ModelTrain(config)
    if args.workflow == 'train':
        history = modelpipe.fit()
        utils.plot_loss(history)  # plot err
    if args.workflow == 'evaluate':
        metric = modelpipe.evaluate()
        print(metric)

def run_pyfile():
    configfile = 'configs/mobilenet_t1.json'
    workflow = 'train'
    config = LoadConfig(configfile)
    modelpipe = ModelTrain(config)
    if workflow == 'train':
        history = modelpipe.fit()
        utils.plot_loss(history)  # plot err
    if workflow == 'evaluate':
        metric = modelpipe.evaluate()
        print(metric)


if __name__ == "__main__":
    run_pyfile()


