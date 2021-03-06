# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:20:38 2017

@author: Dsleviadm
"""

import tensorflow as tf
import numpy as np


def linearfit():
    train_X = np.linspace(-1, 1, 100)
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    w = tf.Variable(0.0, name="weight")
    b = tf.Variable(0.0, name="bias")
    loss =tf.square(Y - tf.multiply(X, w) -b)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        epoch = 1
        for i in range(10):
            for (x, y) in zip(train_X, train_Y):
                _, w_value, b_value = sess.run([train_op, w, b], feed_dict={X:x, Y:y})
            print("Epoch: {}, w:{}, b:{}".format(epoch, w_value, b_value))
            epoch = epoch + 1


flags  = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data", "Directory for storing mnist data")
flags.DEFINE_string("distributed_mode", False, "Run in distributed mode or not")

FLAGS = flags.FLAGS

def main(unused_argv):
    print(FLAGS.data_dir)
    print(FLAGS.distributed_mode)
    
    
if __name__ == "__main__":
    tf.app.run()


data_test=load_data("data\\ihdp_npci_1-100.test.npz")
data_train=load_data("data\\ihdp_npci_1-100.train.npz")