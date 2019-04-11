# -*- coding: utf-8 -*-
import tensorflow as tf
from .basic_layers import ResidualBlock

from keras.layers.convolutional import UpSampling2D

class SegModule(object):
    def __init__(self):
        self.residual_block = ResidualBlock()
        
    def ConvBnRelu(self, x, out_filters=32, kernel_size=5, strides=1, scope = 'conv_bn_relu'):
        with tf.variable_scope(scope):
            x = tf.layers.conv2d(x, out_filters, kernel_size, strides, padding='SAME')
            # x = self.residual_block.batch_norm(x, out_filters)
            x = self.residual_block._batch_norm(x)
            
        return x

    def Conv(self, x, out_filters=32, kernel_size=5, strides=1, scope='conv'):
        with tf.variable_scope(scope):
            x = tf.layers.conv2d(x, out_filters, kernel_size, strides, padding='SAME')
        return x

    def ConvBnReluValid(self, x, out_filters=32, kernel_size=5, strides=1, scope='conv_bn_relu_valid'):
        with tf.variable_scope(scope):
            x = tf.layers.conv2d(x, out_filters, kernel_size, strides, padding='VALID')
            # x = self.residual_block.batch_norm(x, out_filters)
            x = self.residual_block._batch_norm(x)
    
        return x
    def Relu(self, x, scope='relu'):
        with tf.variable_scope(scope):
            return tf.nn.relu(x)