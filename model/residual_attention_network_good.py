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
        self.output_dim = 2
        self.output_dim_seg = 2

        self.attention_module = AttentionModule()
        self.residual_block = ResidualBlock()
        self.seg_module = SegModule()

    def f_prop(self, x, is_training=True):
        """
        forward propagation
        :param x: input Tensor [None, row, line, channel]
        :return: outputs of probabilities
        """
        

        # x = [None, row, line, channel]
        # conv, x -> [None, row, line, 32]
        x = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=1, padding='SAME')
        # max pooling, x -> [None, row, line, 32]
        x = tf.nn.max_pool(x, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        # print('before x.shape:', x.shape)
        # attention module, x -> [None, row, line, 32]
        x = self.attention_module.f_prop(x, input_channels=32, scope="attention_module_1", is_training=is_training)
        
        # residual block, x-> [None, row, line, 64]
        x = self.residual_block.f_prop(x, input_channels=32, output_channels=64, scope="residual_block_1",
                                       is_training=is_training)
        # max pooling, x -> [None, row, line, 64]
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        print('x_before_seg',x.shape)
        tf.add_to_collection('activations_1', x)
        seg_x = x
        # seg
        # layer 1
        x_seg = self.seg_module.ConvBnRelu(seg_x, out_filters=64, kernel_size=1, strides=2, scope='conv_bn_relu_1')
        x_seg_1 = tf.nn.max_pool(x_seg, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
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

        # up sample
        # up layer 1
        x_seg_up1 = UpSampling2D([2, 2])(x_seg_4)
        x_seg_up1 = self.seg_module.ConvBnRelu(x_seg_up1, out_filters=256, kernel_size=5, strides=1,
                                               scope='conv_bn_relu_up_1')
        print('x_seg_up1.shape', x_seg_up1.shape)
        x_seg_up1 = x_seg_up1 + x_seg_3
        x_seg_up1 = self.seg_module.Relu(x_seg_up1, scope='up_relu_1')

        # up layer 2
        x_seg_up2 = UpSampling2D([2, 2])(x_seg_up1)
        x_seg_up2 = self.seg_module.ConvBnRelu(x_seg_up2, out_filters=128, kernel_size=3, strides=1,
                                               scope='conv_bn_relu_up_2')
        x_seg_up2 = x_seg_up2 + x_seg_2
        x_seg_up2 = self.seg_module.Relu(x_seg_up2, scope='up_relu_2')

        # up layer 3
        x_seg_up3 = UpSampling2D([2, 2])(x_seg_up2)
        x_seg_up3 = self.seg_module.ConvBnRelu(x_seg_up3, out_filters=64, kernel_size=1, strides=1,
                                               scope='conv_bn_relu_up_3')
        x_seg_up3 = x_seg_up3 + x_seg_1
        x_seg_up3 = self.seg_module.Relu(x_seg_up3, scope='up_relu_3')

        print('x_seg_up3.shape', x_seg_up3.shape)
        # up layer 4
        x_seg_up4 = UpSampling2D([2, 2])(x_seg_up3)
        x_seg_up4 = self.seg_module.Conv(x_seg_up4, out_filters=128, kernel_size=1, strides=1,
                                         scope='conv_up_1')
        x_seg_up4 = self.seg_module.Relu(x_seg_up4, scope='up_relu_4')
        print('x_seg_up4.shape', x_seg_up4.shape)
        
        # seg attention

        x_seg_attention_1 = self.seg_module.ConvBnRelu(x_seg_up4, out_filters=256, kernel_size=3, strides=1,
                                               scope='conv_seg_attention_1')
        # max pooling, x -> [None, row/2, line/2, 128]
        x_max_pool_1 = tf.nn.max_pool(x_seg_attention_1, ksize=[1, 3, 3, 1], strides=[1, 4, 4, 1], padding='SAME')
        print('x_max_pool_1.shape', x_max_pool_1.shape)
        
        # class attention ***************
        # attention module, x -> [None, row, line, 64]
        x = self.attention_module.f_prop(x, input_channels=64, scope="attention_module_2", is_training=is_training)

        # residual block, x-> [None, row, line, 128]
        x = self.residual_block.f_prop(x, input_channels=64, output_channels=128, scope="residual_block_2",
                                       is_training=is_training)
        # max pooling, x -> [None, row/2, line/2, 128]
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        print('class attention 2 shape', x.shape)

        # attention module, x -> [None, row/2, line/2, 64]
        x = self.attention_module.f_prop(x, input_channels=128, scope="attention_module_3", is_training=is_training)

        # residual block, x-> [None, row/2, line/2, 256]
        x = self.residual_block.f_prop(x, input_channels=128, output_channels=256, scope="residual_block_3",
                                       is_training=is_training)
        # max pooling, x -> [None, row/4, line/4, 256]
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        print('class attention 3 shape', x.shape)
        
        # Eltwise seg and class *******************
        # x = x_max_pool_1 + x
        # x = tf.concat([x_max_pool_1, x], -1)
        # print('after concat shape', x.shape)
        #
        # x = self.seg_module.ConvBnRelu(x, out_filters=256, kernel_size=3, strides=1,
        #                                                scope='conv_seg_attention_2')
        # # max pooling, x -> [None, row/2, line/2, 128]
        # x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')


        # seg up layer
        # x_seg_up4 = UpSampling2D([4, 4])(x)
        x_seg_up4 = self.seg_module.Conv(x_seg_up4, out_filters=self.output_dim_seg, kernel_size=1, strides=1,
                                         scope='after_conv_up_1')
        

        y_seg = tf.nn.softmax(x_seg_up4)
        print('y_seg.shape', y_seg.shape)
        tf.add_to_collection('pred_network', y_seg)  # 用于加载模型获取要预测的网络结构
        
        x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

        # layer normalization
        x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
        # FC, softmax, class
        y_class = tf.layers.dense(x, self.output_dim, activation=tf.nn.softmax)
        # FC, softmax, seg
        return y_class, y_seg

