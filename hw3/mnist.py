#!/usr/bin/python3.6

### Kevin Sheng
### ECE471 Selected Topics in Machine Learning - Assignment 3 

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.contrib.learn.python.learn.datasets import mnist

class Model():
    def __init__(self, data, epochs=200, learning_rate=.001): 
        batch_size = 100 
        dropout = 0.80
        reg = 1e-3
        stride = 1
        
        X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        Y = tf.placeholder(tf.float32, [None, 10])
        keep_prob = tf.placeholder(tf.float32)
        
        K = 4  # first convolutional layer output depth
        L = 8  # second convolutional layer output depth
        M = 12  # third convolutional layer
        N = 200  # fully connected layer

        w1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
        w2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
        w3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
        w4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
        w5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
        weights = (w1, w2, w3, w4, w5)

        b1 = tf.Variable(tf.ones([K])/10)
        b2 = tf.Variable(tf.ones([L])/10)
        b3 = tf.Variable(tf.ones([M])/10)
        b4 = tf.Variable(tf.ones([N])/10)
        b5 = tf.Variable(tf.ones([10])/10)
        
        y1 = tf.nn.relu(tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
        y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, strides=[1, 2, 2, 1], padding='SAME') + b2)
        y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, strides=[1, 2, 2, 1], padding='SAME') + b3)

        yy = tf.reshape(y3, shape=[-1, 7 * 7 * M])

        y4 = tf.nn.relu(tf.matmul(yy, w4) + b4)
        logits = tf.matmul(y4, w5) + b5
        pred = tf.nn.softmax(logits)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        l2 = tf.reduce_sum([tf.reduce_sum(tf.pow(w,2)) for w in weights])
        loss = loss + reg * l2
        
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        optim = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)
        with tqdm(range(epochs)) as t:
            for step in t:
                batch_X, batch_Y = data.train.next_batch(batch_size)
                a_train, c_train, _ = sess.run([accuracy, loss, optim],
                    feed_dict={X: batch_X, Y: batch_Y, keep_prob: dropout})
                if step % 100 == 0:
                    a_validation, c_validation, _ = sess.run([accuracy, loss, optim],
                        feed_dict={X: data.validation.images, Y: data.validation.labels, keep_prob: 1.0})
                t.set_postfix(
                    t_acc=a_train,
                    t_loss=c_train,
                    v_acc=a_validation,
                    v_loss=c_validation
                )

        a, c = sess.run([accuracy, loss],
            feed_dict={X:data.test.images, Y:data.test.labels, keep_prob: 1.0})
        print('test accuracy={}, test loss={}'.format(a, c))

if __name__ == '__main__':
    data = mnist.read_data_sets("data", one_hot=True, reshape=False)
    model = Model(data, epochs=2000)
