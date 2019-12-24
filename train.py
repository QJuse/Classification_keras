
import os
import tensorflow as tf
import tensorflow.keras.optimizers as KO
import tensorflow.keras.callbacks as KC
import tensorflow.keras.models as KM

from Module.config import Config
from Module.datasets import ImageData
from Module.models import CustomModel
from Module.utils import save_json, plot_loss

print("tf version {}, keras verion {}"\
      .format(tf.__version__, tf.keras.__version__))


def callbacks(config):
    """ set callbacks when model fit """
    # checkpoint
    checkpoint = KC.ModelCheckpoint(
        filepath=config.checkpoint,
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        period=config.ckpt_interval)
    # lr_scheduler
    def lr_schedule(epoch):
        lr = config.init_lr
        total_epochs = config.epochs
        check_1 = int(total_epochs * 0.8)
        check_2 = int(total_epochs * 0.4)
        if epoch > check_1:
            lr *= 1e-2
        elif epoch > check_2:
            lr *= 1e-1
        return lr
    lr_scheduler = KC.LearningRateScheduler(lr_schedule)
    # tensorboard
    tensorboard = KC.TensorBoard(
        log_dir=config.tb_logs,
        batch_size=config.batch_size,
        histogram_freq=0,         # validation data can not be generator
        write_graph=False,
        write_grads=False)
    callbacks=[checkpoint, lr_scheduler, tensorboard]
    return callbacks


def fit(config):
    Model = CustomModel(config.network, config.pre_weight, config.nclass)
    model = Model.build_model()
    # optim
    optimizer = KO.Adam(lr=config.init_lr, decay=1e-4)  # optimizer args
    model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])
    # input data
    img_train = ImageData(config.train, config.batch_size, config.image_size)
    img_valid = ImageData(config.valid, config.batch_size, config.image_size)
    save_json(img_train.classmap, config.classmap)  # 保存类别映射
    # fit model
    num_train = len(img_train.datagen.filenames)
    print(num_train)
    print(img_train.datagen)
    print(img_train.datagen.next()[0].shape)
    # num_valid = len(img_valid.datagen.filenames)
    print("Number of experiments (Epochs) : ", config.epochs)
    history = model.fit_generator(
        img_train.datagen,
        steps_per_epoch=int(num_train / config.batch_size),
        #steps_per_epoch=1,
        epochs=config.epochs,
        validation_data=img_valid.datagen,
        validation_steps=5,
        callbacks=callbacks(config))
    # save model
    model.save(config.tuned_model)
    return history


def evaluate(config):
    img_test = ImageData(config.test, config.batch_size, config.image_size)
    num_test = len(img_test.datagen.filenames)
    model = KM.load_model(config.tuned_model)
    history = model.evaluate_generator(
        img_test.datagen,
        steps=int(num_test / config.batch_size),
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        verbose=1)
    return history

# 命令行解析

def main():
    import argparse
    parser = argparse.ArgumentParser(description='train/evaluate CNN model')
    parser.add_argument(dest='filenames', metavar='filename', nargs='*')

    parser.add_argument('-n', '--network', dest='network', action='store',
                        help='string, one of keras model')
    parser.add_argument('-v', '--version', dest='version', action='store',
                        help='the id of output model')
    parser.add_argument('-w', '--pre_weight', dest='pre_weight', action='store',
                        help='the pre_weights')
    parser.add_argument('-wf', '--workflow', dest='workflow', action='store',
                    choices={'train', 'evaluate'}, default='train',
                    help='model work in train or evaluate mode')
    args = parser.parse_args()

    cfg = Config(args.network, args.pre_weight, args.version)
    if args.workflow == 'train':
        history = fit(cfg)
        # plot err
        plot_loss(history)

    if args.workflow == 'evaluate':
        metric = evaluate(cfg)
        print(metric)


if __name__ == "__main__":
    main()
