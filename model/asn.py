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
        self.output_dim_seg = 3

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
        
        seg_x = x
        print('x.shape', x.shape)
        # seg
        x_front_seg = self.seg_module.ConvBnRelu(seg_x, out_filters=64, kernel_size=1, strides=1,
                                           scope='conv_bn_relu_front_1')
        x_front_seg_1 = tf.nn.max_pool(x_front_seg, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        x_front_seg_2 = self.seg_module.ConvBnRelu(x_front_seg_1, out_filters=64, kernel_size=3, strides=1,
                                             scope='conv_bn_relu_front_2')
        # x_front_seg_2 = x_front_seg_2 + x_front_seg_1
        x_front_seg_3 = tf.nn.max_pool(x_front_seg_2, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        x_front_seg_4 = self.seg_module.ConvBnRelu(x_front_seg_3, out_filters=64, kernel_size=5, strides=1,
                                             scope='conv_bn_relu_front_3')
        # x_front_seg_4 = x_front_seg_4 + x_front_seg_3
        x_front_seg_5 = tf.nn.max_pool(x_front_seg_4, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
        x_front_seg_6 = self.seg_module.ConvBnRelu(x_front_seg_5, out_filters=64, kernel_size=7, strides=1,
                                             scope='conv_bn_relu_front_4')
        # x_front_seg_6 = x_front_seg_6 + x_front_seg_5
        x_front_seg_7 = tf.nn.max_pool(x_front_seg_6, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        
        
        # layer 1
        x_seg = self.seg_module.ConvBnRelu(x_front_seg_7, out_filters=64, kernel_size=1, strides=1, scope='conv_bn_relu_1')
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

        # up sample
        # up layer 1
        x_seg_up1 = UpSampling2D([2, 2])(x_seg_4)
        x_seg_up1 = self.seg_module.ConvBnRelu(x_seg_up1, out_filters=256, kernel_size=5, strides=1,
                                               scope='conv_bn_relu_up_1')
        print('x_seg_up1.shape', x_seg_up1.shape)
        x_seg_up1 = self.seg_module.ConvBnReluValid(x_seg_up1, out_filters=256, kernel_size=2, strides=1)

        x_seg_up1 = x_seg_up1 + x_seg_3
        # x_seg_up1 = tf.concat((x_seg_up1, x_seg_3), 3)
        x_seg_up1 = self.seg_module.Relu(x_seg_up1, scope='up_relu_1')
        # add
        x_seg_up1_1 = self.seg_module.ConvBnRelu(x_seg_up1, out_filters=256, kernel_size=5, strides=1,
                                               scope='conv_bn_relu_up_1_1')
        x_seg_up1 = x_seg_up1 + x_seg_up1_1
        x_seg_up1 = tf.nn.max_pool(x_seg_up1, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        # up layer 2
        x_seg_up2 = UpSampling2D([2, 2])(x_seg_up1)
        x_seg_up2 = self.seg_module.ConvBnRelu(x_seg_up2, out_filters=128, kernel_size=3, strides=1,
                                               scope='conv_bn_relu_up_2')
        x_seg_up2 = x_seg_up2 + x_seg_2
        # x_seg_up2 = tf.concat((x_seg_up2, x_seg_2), 3)
        x_seg_up2 = self.seg_module.Relu(x_seg_up2, scope='up_relu_2')
        
        # add
        x_seg_up2_1 = self.seg_module.ConvBnRelu(x_seg_up2, out_filters=128, kernel_size=3, strides=1,
                                                 scope='conv_bn_relu_up_2_1')
        x_seg_up2 = x_seg_up2 + x_seg_up2_1
        x_seg_up2 = tf.nn.max_pool(x_seg_up2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        # up layer 3
        x_seg_up3 = UpSampling2D([2, 2])(x_seg_up2)
        x_seg_up3 = self.seg_module.ConvBnRelu(x_seg_up3, out_filters=64, kernel_size=1, strides=1,
                                               scope='conv_bn_relu_up_3')
        x_seg_up3 = x_seg_up3 + x_seg_1
        # x_seg_up3 = tf.concat((x_seg_up3, x_seg_1), 3)
        x_seg_up3 = self.seg_module.Relu(x_seg_up3, scope='up_relu_3')

        # add
        x_seg_up3_1 = self.seg_module.ConvBnRelu(x_seg_up3, out_filters=64, kernel_size=1, strides=1,
                                                 scope='conv_bn_relu_up_3_1')
        x_seg_up3 = x_seg_up3 + x_seg_up3_1
        x_seg_up3 = tf.nn.max_pool(x_seg_up3, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

        print('x_seg_up3.shape', x_seg_up3.shape)
        # up layer 4
        x_seg_up4 = UpSampling2D([2, 2])(x_seg_up3)
        x_seg_up4 = self.seg_module.Conv(x_seg_up4, out_filters=128, kernel_size=1, strides=1,
                                         scope='conv_up_1')
        x_seg_up4 = self.seg_module.Relu(x_seg_up4, scope='up_relu_4')
        print('x_seg_up4.shape', x_seg_up4.shape)
        tf.add_to_collection('activations_1', x)


        # seg up layer
        # x_seg_up4 = UpSampling2D([4, 4])(x)
        x_seg_up4 = self.seg_module.Conv(x_seg_up4, out_filters=self.output_dim_seg, kernel_size=1, strides=2,
                                         scope='after_conv_up_1')
        

        y_seg = tf.nn.softmax(x_seg_up4)
        print('y_seg.shape', y_seg.shape)
        tf.add_to_collection('pred_network', y_seg)  # 用于加载模型获取要预测的网络结构
        return y_seg

