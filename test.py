# -*- coding: utf-8 -*-
import tensorflow as tf

classes = 3
labels = tf.constant([0,1,2,3]) # 输入的元素值最小为0，最大为2
output = tf.one_hot(labels,classes)

sess = tf.Session()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(output)
    print("output of one-hot is : ",output)