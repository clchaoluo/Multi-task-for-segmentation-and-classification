# -*- coding: utf-8 -*-
"""
Residual Attention Network
"""

import tensorflow as tf
import numpy as np
from keras.layers.convolutional import UpSampling2D


from .basic_layers import ResidualBlock
from .attention_module import AttentionModule
from .seg_module import SegModule



class ResidualAttentionNetwork(object):
    """
    Residual Attention Network
    URL: https://arxiv.org/abs/1704.06904
    """
    def __init__(self):
        """
        :param input_shape: the list of input shape (ex: [None, 28, 28 ,3]
        :param output_dim:
        """
        self.input_shape = [-1, 128, 128, 1]
        self.output_dim_seg = 2
        self.output_dim = 2

        self.attention_module = AttentionModule()
        self.residual_block = ResidualBlock()
        self.seg_module = SegModule()

    def f_prop(self, x, is_training=True, keep_prob = 0.5):
        """
        forward propagation
        :param x: input Tensor [None, row, line, channel]
        :return: outputs of probabilities
        """
        

        # x = [None, row, line, channel]
        # conv, x -> [None, row, line, 32]

        # attention 1
        # x = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=1, padding='SAME')
        # # max pooling, x -> [None, row, line, 32]
        # x = tf.nn.max_pool(x, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        #
        # # print('before x.shape:', x.shape)
        # # attention module, x -> [None, row, line, 32]
        # x = self.attention_module.f_prop(x, input_channels=32, scope="attention_module_1", is_training=is_training)
        #
        # # residual block, x-> [None, row, line, 64]
        # x = self.residual_block.f_prop(x, input_channels=32, output_channels=64, scope="residual_block_1",
        #                                is_training=is_training)
        # # max pooling, x -> [None, row, line, 64]
        # x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        # print('pool_1.shape', x.shape)

        seg_x = x
        print('x.shape', x.shape)
        # seg
        x_front_seg = self.seg_module.ConvBnRelu(seg_x, out_filters=64, kernel_size=1, strides=1,
                                           scope='conv_bn_relu_front_1')
        x_front_seg_1 = tf.nn.max_pool(x_front_seg, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

        x_front_seg_2 = self.seg_module.ConvBnRelu(x_front_seg_1, out_filters=64, kernel_size=3, strides=1,
                                             scope='conv_bn_relu_front_2')
        # x_front_seg_2 = x_front_seg_2 + x_front_seg_1
        x_front_seg_3 = tf.nn.max_pool(x_front_seg_2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        x_front_seg_4 = self.seg_module.ConvBnRelu(x_front_seg_3, out_filters=64, kernel_size=5, strides=1,
                                             scope='conv_bn_relu_front_3')
        # x_front_seg_4 = x_front_seg_4 + x_front_seg_3
        x_front_seg_5 = tf.nn.max_pool(x_front_seg_4, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        x_front_seg_6 = self.seg_module.ConvBnRelu(x_front_seg_5, out_filters=64, kernel_size=7, strides=1,
                                             scope='conv_bn_relu_front_4')
        # x_front_seg_6 = x_front_seg_6 + x_front_seg_5
        x_front_seg_7 = tf.nn.max_pool(x_front_seg_6, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME')

        # layer 1
        x_seg = self.seg_module.ConvBnRelu(x_front_seg_7, out_filters=64, kernel_size=1, strides=1, scope='conv_bn_relu_1')
        x_seg_1 = tf.nn.max_pool(x_seg, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        print('x_seg_1.shape', x_seg_1.shape)

        # layer 2
        x_seg_2 = self.seg_module.ConvBnRelu(x_seg_1, out_filters=128, kernel_size=1, strides=2, scope='conv_bn_relu_2')
        x_seg_2 = tf.nn.max_pool(x_seg_2, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
        # layer 2 branch
        x_seg_l2_1 = self.seg_module.ConvBnRelu(x_seg_2, out_filters=128, kernel_size=1, strides=1,
                                                scope='conv_bn_relu_2_branch')
        x_seg_l2_1 = tf.nn.max_pool(x_seg_l2_1, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
        # eltwise
        x_seg_2 = x_seg_2 + x_seg_l2_1
        x_seg_2 = tf.nn.max_pool(x_seg_2, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
        # x_seg_2 = tf.nn.atrous_conv2d(x_seg_2, filters=[3, 3, 128, 128], rate=3, padding='SAME', name='atr_conv_1')

        print('x_seg_2.shape', x_seg_2.shape)

        # layer 3
        x_seg_3 = self.seg_module.ConvBnRelu(x_seg_2, out_filters=256, kernel_size=3, strides=2, scope='conv_bn_relu_3')
        x_seg_3 = tf.nn.max_pool(x_seg_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        # layer 3 branch
        x_seg_l3_1 = self.seg_module.ConvBnRelu(x_seg_3, out_filters=256, kernel_size=3, strides=1,
                                                scope='conv_bn_relu_3_branch')
        x_seg_l3_1 = tf.nn.max_pool(x_seg_l3_1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        # eltwise
        x_seg_3 = x_seg_3 + x_seg_l3_1
        x_seg_3 = tf.nn.max_pool(x_seg_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        print('x_seg_3.shape', x_seg_3.shape)
        # layer 4
        x_seg_4 = self.seg_module.ConvBnRelu(x_seg_3, out_filters=512, kernel_size=5, strides=2, scope='conv_bn_relu_4')
        x_seg_4 = tf.nn.max_pool(x_seg_4, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        # layer 4 branch
        x_seg_l4_1 = self.seg_module.ConvBnRelu(x_seg_4, out_filters=512, kernel_size=5, strides=1,
                                                scope='conv_bn_relu_4_branch')
        x_seg_l4_1 = tf.nn.max_pool(x_seg_l4_1, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        # eltwise
        x_seg_4 = x_seg_l4_1 + x_seg_4
        x_seg_4 = tf.nn.max_pool(x_seg_4, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')

        print('x_seg_layer4.shape', x_seg_4.shape)

        # class attention ***************
        # attention module, x -> [None, row, line, 64]
        

        # x = self.seg_module.ConvBnReluValid(x, out_filters=256, kernel_size=4, strides=1, scope='t_2')
        # print ('x_class_start_v', x.shape)
        #
        # # # attention 1
        # # x = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=1, padding='SAME')
        # # max pooling, x -> [None, row, line, 32]
        # x = tf.nn.max_pool(x, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        #
        # # print('before x.shape:', x.shape)
        # # attention module, x -> [None, row, line, 32]
        # x = self.attention_module.f_prop(x, input_channels=256, scope="attention_module_1", is_training=is_training)
        #
        # # residual block, x-> [None, row, line, 64]
        # x = self.residual_block.f_prop(x, input_channels=256, output_channels=256, scope="residual_block_1",
        #                                is_training=is_training)
        # # max pooling, x -> [None, row, line, 64]
        # x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        # print('pool_1.shape', x.shape)
        # x = class_x + x
        # x = self.seg_module.ConvBnReluValid(x, out_filters=512, kernel_size=3, strides=1, scope='t_1')
        # class_x = self.seg_module.ConvBnRelu(x, out_filters=256, kernel_size=3, strides=1,scope='class_conv_1')

        x = self.attention_module.f_prop(x_seg_4, input_channels=512, scope="attention_module_2", is_training=is_training)

        # residual block, x-> [None, row, line, 128]
        x = self.residual_block.f_prop(x, input_channels=512, output_channels=256, scope="residual_block_2",
                                       is_training=is_training)
        # max pooling, x -> [None, row/2, line/2, 128]
        # x = x + class_x
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        # pool_2 = x
        print('pool_2.shape', x.shape)
        #
        #
        #
        # attention module, x -> [None, row/2, line/2, 64]

        x = self.attention_module.f_prop(x, input_channels=256, scope="attention_module_3", is_training=is_training)

        # residual block, x-> [None, row/2, line/2, 256]
        x = self.residual_block.f_prop(x, input_channels=256, output_channels=64, scope="residual_block_3",
                                       is_training=is_training)
        # max pooling, x -> [None, row/4, line/4, 256]
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        print('pool_3.shape', x.shape)
        # if is_training == True:
        #     x = tf.nn.dropout(x, keep_prob)
        print('x_class', x.shape)
        x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
        print('x_class_reshape', x.shape)
        # layer normalization
        x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
        print('x_class_norm', x.shape)
        # FC, softmax, class
        y_class = tf.layers.dense(x, self.output_dim, activation=tf.nn.softmax)
        print ('y_class.shape', y_class.shape)
        tf.add_to_collection('pre_class', y_class)
        return y_class

