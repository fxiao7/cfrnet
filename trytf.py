# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:50:56 2017

@author: Dsleviadm
"""

import tensorflow as tf
import numpy as np

x_data=np.float32(np.random.rand(2, 100))
y_data=np.dot([0.100, 0.200], x_data)+0.300

b=tf.Variable(tf.zeros([1]))
W=tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y=tf.matmul(W, x_data)+b

loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

for step in range(0, 201):
    sess.run(train)
    if step%20 ==0:
        print(step, sess.run(W), sess.run(b), sess.run(tf.square(y-y_data)))
        
        
        
        
matrix1=tf.constant([[2, 2]])
matrix2=tf.constant([[2],[2]])
product=tf.matmul(matrix1, matrix2)
sess=tf.Session()
result=sess.run(product)
print(result)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


import tensorflow.examples.tutorials.mnist.input_data
mnist = read_data_sets("MNIST_data/", one_hot=True)


x=tf.placeholder(tf.float32, [None, 784])

W=tf.Variable(tf.zeros([784, 10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x, W)+b)
y_=tf.placeholder("float", [None, 10])
cross_entropy=-tf.reduce_sum(y_ * tf.log(y))

train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for i in range(100):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction=tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


print(accuracy.eval)
correct_prediction