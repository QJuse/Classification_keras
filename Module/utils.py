# -*- coding: utf-8 -*-
"""
Created on Thu May  9 19:54:45 2019

@author: zhengqj@fotile.com
"""
import json
import tensorflow as tf
import matplotlib.pyplot as plt

# 绘制模型图片
def save_plot_model(model):
    from tensorflow.keras.utils import plot_model     # save model as a picture
    # depend on pydot and graphviz  developer env
    SAVE_NAME = 'inception_resnet_v2.jpg'
    plot_model(model, to_file='lib/' + SAVE_NAME, show_shapes=True)  # keras 模型对象

# 检查模型是否正确
def check_model(model):
    model.summary()

# 将字典按指定格式保存成json文件
def save_json(var,path):
     with open(path, 'w+') as f:
         json.dump(var, f, indent=4, separators=(",", ":"), ensure_ascii=True)


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
   
     
def save_pb_model(sess, model, class_json, export_path):
    """ save tensorflow model to be a pb model 
    
    args:
        sess:
        model:
        class_json:
        export_path:  
    """
    x = model.input
    y = model.output
    P_prob, indices = tf.nn.top_k(y, 5)
    with open(class_json, 'r') as f:
        class_map = json.load(f)
    const = tf.constant([i for i in class_map.values()])
    table = tf.contrib.lookup.index_to_string_table_from_tensor(const)
    P_class = table.lookup(tf.cast(indices, tf.int64))
    
    # prediction_signature map
    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y)
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_x},
                outputs={'scores': tensor_info_y},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    # classification_signature map   
    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    classification_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
    classification_outputs_classes = tf.saved_model.utils.build_tensor_info(P_class)
    classification_outputs_scores = tf.saved_model.utils.build_tensor_info(P_prob)
    classification_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                        classification_inputs },
            outputs={tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                         classification_outputs_classes,
                     tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                         classification_outputs_scores },
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))   
    # builder 
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(sess, 
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                classification_signature,},
        main_op=tf.tables_initializer(),         # excute when Graphdef load
        strip_default_attrs=True)
    builder.save()

