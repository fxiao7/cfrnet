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