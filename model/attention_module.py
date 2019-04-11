# -*- coding: utf-8 -*-
"""
attention module of Residual Attention Network
"""

import tensorflow as tf
from keras.layers.convolutional import UpSampling2D

from .basic_layers import ResidualBlock


class AttentionModule(object):
    """AttentionModuleClass"""
    def __init__(self, p=1, t=2, r=1):
        """
        :param p: the number of pre-processing Residual Units before splitting into trunk branch and mask branch
        :param t: the number of Residual Units in trunk branch
        :param r: the number of Residual Units between adjacent pooling layer in the mask branch
        """
        self.p = p
        self.t = t
        self.r = r

        self.residual_block = ResidualBlock()

    def f_prop(self, input, input_channels, scope="attention_module", is_training=True):
        """
        f_prop function of attention module
        :param input: A Tensor. input data [batch_size, height, width, channel]
        :param input_channels: dimension of input channel.
        :param scope: str, tensorflow name scope
        :param is_training: boolean, whether training step or not(test step)
        :return: A Tensor [batch_size, height, width, channel]
        """
        with tf.variable_scope(scope):

            # residual blocks(TODO: change this function)
            with tf.variable_scope("first_residual_blocks"):
                for i in range(self.p):
                    input = self.residual_block.f_prop(input, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)
            # before mask
            if scope == 'attention_module_2':
                tf.add_to_collection('activations_1', input)
            with tf.variable_scope("trunk_branch"):
                output_trunk = input
                for i in range(self.t):
                    output_trunk = self.residual_block.f_prop(output_trunk, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

            with tf.variable_scope("soft_mask_branch"):

                with tf.variable_scope("down_sampling_1"):
                    # max pooling
                    filter_ = [1, 2, 2, 1]
                    output_soft_mask = tf.nn.max_pool(input, ksize=filter_, strides=filter_, padding='SAME')

                    for i in range(self.r):
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

                with tf.variable_scope("skip_connection"):
                    # TODO(define new blocks)
                    # print('output_soft_mask.shape_1', output_soft_mask.shape)
                    output_skip_connection = self.residual_block.f_prop(output_soft_mask, input_channels, is_training=is_training)
                    # print('output_skip_connection.shape_1',output_skip_connection.shape)

                with tf.variable_scope("down_sampling_2"):
                    # max pooling
                    filter_ = [1, 2, 2, 1]
                    output_soft_mask = tf.nn.max_pool(output_soft_mask, ksize=filter_, strides=filter_, padding='SAME')
                    # print('output_soft_mask.shape_2', output_soft_mask.shape)
                    for i in range(self.r):
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)
                    # print('output_soft_mask.shape_3', output_soft_mask.shape)
                with tf.variable_scope("up_sampling_1"):
                    for i in range(self.r):
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)
                    # print('output_soft_mask.shape_4', output_soft_mask.shape)
                    # interpolation
                    output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)
                    # print('output_soft_mask.shape_5', output_soft_mask.shape)
                # add skip connection
                
                output_soft_mask += output_skip_connection

                with tf.variable_scope("up_sampling_2"):
                    for i in range(self.r):
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

                    # interpolation
                    output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)


                with tf.variable_scope("output"):
                    output_soft_mask = tf.layers.conv2d(output_soft_mask, filters=input_channels, kernel_size=1)
                    output_soft_mask = tf.layers.conv2d(output_soft_mask, filters=input_channels, kernel_size=1)
                    
                    
                    # print('soft mask output_soft_mask.shap',output_soft_mask.shape)
                    # sigmoid
                    output_soft_mask = tf.nn.sigmoid(output_soft_mask)

            # soft mask
            if scope == 'attention_module_2':
                tf.add_to_collection('activations_1', output_soft_mask)
            with tf.variable_scope("attention"):
                output = (1 + output_soft_mask) * output_trunk
            
            # print('after mask output.shape:', output.shape)
            if scope == 'attention_module_2':
                tf.add_to_collection('activations_1', output)
            with tf.variable_scope("last_residual_blocks"):
                for i in range(self.p):
                    output = self.residual_block.f_prop(output, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

            return output
