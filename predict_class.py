# -*- coding: utf-8 -*-
from __future__ import division
import tensorflow as tf
import os
import utils
import xml.dom.minidom as xmldom
import SimpleITK as sitk
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='6'


def compute_specificity(pre_list, y_class_list):
    pre_num = len(pre_list)
    y_num = len(y_class_list)
    if pre_num != y_num:
        print('pre_num != y_num')
    pre_positive_num = 0
    pre_negetive_num = 0
    for i in range(pre_num):
        if pre_list[i] == 0 and y_class_list[i] == 0:
            pre_negetive_num += 1
        if pre_list[i] == 1 and y_class_list[i] == 0:
            pre_positive_num += 1
    if pre_negetive_num == 0 and pre_positive_num == 0:
        print("error")
        exit()
    else:
        specificity = pre_negetive_num / (pre_negetive_num + pre_positive_num)
        return specificity
def compute_sensitivity(pre_list, y_class_list):
    pre_num = len(pre_list)
    y_num = len(y_class_list)
    if pre_num != y_num:
        print('pre_num != y_num')
    pre_positive_num = 0
    pre_negetive_num = 0
    for i in range(pre_num):
        if pre_list[i] == 1 and y_class_list[i] == 1:
            pre_positive_num += 1
        if pre_list[i] == 0 and y_class_list[i] == 1:
            pre_negetive_num += 1
    
    sensitivity = pre_positive_num / (pre_positive_num + pre_negetive_num)
    return sensitivity
def compute_accuracy(pre_list, y_class_list):
    pre_num = len(pre_list)
    y_num = len(y_class_list)
    if pre_num != y_num:
        print('pre_num != y_num')
    right_num = 0
    for i in range(y_num):
        if pre_list[i] == y_class_list[i]:
            right_num += 1
    accur = right_num/pre_num
    return accur

if __name__ == '__main__':
    num = 5
    main_path = '/home/luochao/project/Data/classification'
    train_X, train_y_class, train_y_seg, valid_X, valid_y_class, valid_y_seg, test_X, test_y, test_class_y = utils.load_data_new_dcm_128(num)
    # x = tf.placeholder(tf.float32, [None, 128, 128, 1])
    # y = tf.placeholder(tf.float32, [None, 128, 128, 1])
    # num = 5 #临时配置，测完删除
    HOME_DIR = os.environ['HOME'] + '/project/residual-attention-network_new_dcmdata/trained_models_'+str(num)+'/'
    # predict
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(HOME_DIR + 'model.ckpt-99.meta')
        saver.restore(sess, HOME_DIR + 'model.ckpt-99')  # .data文件
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name('x').outputs[0]
        y = tf.get_collection("pre_class")[0]
        # r = 142
        r = 97
        pre_list = []
        for k in range(r):
            result = sess.run(y, feed_dict={x: test_X[k:k+1]})
            data = result[0]
            pre_list.append(np.argmax(data))
        y_class_list = np.argmax(test_class_y, 1)
        acurracy = compute_accuracy(pre_list, y_class_list)
        sensitivity = compute_sensitivity(pre_list, y_class_list)
        specificity = compute_specificity(pre_list, y_class_list)
        print("accuracy", acurracy)
        print("sensitivity", sensitivity)
        print("specificity", specificity)

            