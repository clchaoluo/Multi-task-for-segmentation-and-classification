#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:17:31 2017

@author: hwj
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append('/home/hwj/tf_practice/models/research/slim')
import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np
import nets.vgg as vgg
from datasets import imagenet
from preprocessing import vgg_preprocessing
import scipy.io as scio

'''
extract specific layer feature of vgg_16, heckpoints_dir and layer name are needed 

'''


def extract_feature_vgg_16(checkpoints_dir, input_image, layer):
    image_size = vgg.vgg_16.default_image_size
    processed_image = vgg_preprocessing.preprocess_image(input_image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)

    processed_images = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(vgg.vgg_arg_scope()):
        _, model_var = vgg.vgg_16(processed_images,
                                       num_classes=1000,
                                       is_training=False)

    init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
                                             slim.get_model_variables('vgg_16'))

    with tf.Session() as sess:
        init_fn(sess)
        features = sess.run(model_var)
        fea=features['vgg_16/'+layer]

    return fea


'''
extract specific layer feature of vgg_19, checkpoints_dir and layer name are needed

'''


def extract_feature_vgg_19(checkpoints_dir, input_image, layer):
    image_size = vgg.vgg_19.default_image_size
    processed_image = vgg_preprocessing.preprocess_image(input_image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)

    processed_images = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(vgg.vgg_arg_scope()):
        _, model_var = vgg.vgg_19(processed_images,
                                       num_classes=1000,
                                       is_training=False)

    init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
                                             slim.get_model_variables('vgg_16'))

    with tf.Session() as sess:
        init_fn(sess)
        features = sess.run(model_var)
        fea=features['vgg_19/'+layer]

    return fea

'''
extract specific layer feature of vgg, vgg_16 and vgg_16 are avaliable to chose
default net is vgg_16

'''


def extract_feature(checkpoints_dir, input_images, net='vgg_16', layer='conv5_'):
    if net=='vgg_16':
        fea=extract_feature_vgg_16(checkpoints_dir, input_images, layer)
        return fea
    elif net=='vgg_19':
        fea=extract_feature_vgg_19(checkpoints_dir, input_images, layer)
        return fea
    else:
        print('net type error, chose vgg_16 or vgg_19 to extract vgg feature')

