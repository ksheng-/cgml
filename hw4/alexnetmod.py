#!/usr/bin/python3.6

### Kevin Sheng
### ECE471 Selected Topics in Machine Learning - Assignment 3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py
import cifar10_input

class AlexNetMod():
    def __init__(self):
        pass

    def step(self, images, labels, global_step, learning_rate, momentum, weight_decay):
        X = images
        y = labels 
        
        # started with alexnet since its simple and old so it trains
        #   decently fast on a GTX860M
        # decrease learning rate over time
        # input -> 32x32x3
        # conv1 (f=5, s=1, k=64 relu) -> 32x32x64
        # pool1 (f=3, s=2) -> 16x16x64
        # conv2 (f=5, s=1, k=64 relu) -> 16x16x64
        # pool2 (f=3, s=2) 8x8x64
        # dropout (.5)
        # fc1 (384 relu) -> 1x384
        # fc2 (192 relu) -> 1x192
        # linear -> 1x10

        f = [5, 5]
        s = [1, 1]
        k = [64, 64, 384, 192]
        keep_prob = .5

        # conv1
        w1 = self.weight('w1', [f[0], f[0], 3, k[0]])
        b1 = self.bias('b1', [k[0]])
        X_ = self.conv(X, w1, s[0], b1)

        # pool1
        X_ = self.pool(X_, 3, 2)

        # conv2
        w2 = self.weight('w2', [f[1], f[1], k[0], k[1]])
        b2 = self.bias('b2', [k[1]])
        X_ = self.conv(X_, w2, s[1], b2)
        
        # pool2
        X_ = self.pool(X_, 3, 2)
        
        # dropout
        X_ = tf.nn.dropout(X_, keep_prob)

        # fc1
        w3 = self.weight('w3', [k[1], k[2]])
        b3 = self.bias('b3', [k[2]])
        X_ = tf.reshape(X_, shape=[-1, k[1]])
        X_ = self.fc(X_, w3, b3)
        
        # fc2
        w4 = self.weight('w4', [k[2], k[3]])
        b4 = self.bias('b4', [k[3]])
        
        # linear
        w5 = self.weight('w5', [k[3], 10])
        b5 = self.bias('b5', [10])
        logits = self.fc(X_, w4, b4)

        pred = tf.nn.softmax(logits)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        l2 = tf.reduce_sum([tf.reduce_sum(tf.pow(w,2)) for w in (w1, w2, w3, w4, w5)])
        loss = loss + weight_decay * l2

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        optim = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        
        return accuracy, loss, optim

    def train(self, epochs, learning_rate=.01, batch_size=32, early_stop=False):
        with tf.Graph().as_default():
            global_step = tf.train.get_or_create_global_step()

            num_train = 50000
            steps_per_epoch = num_train // batch_size
            learning_rate = tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    steps_per_epoch * 350,
                    0.1,
                    staircase=True)
            
            init = tf.global_variables_initializer()

            saver = tf.train.Saver()
            checkpoint = 'checkpoints/model.ckpt'

            sess = tf.Session()
            sess.run(init)
            with tqdm(range(epochs)) as t:
                # training data QueueRunner that returns (32, 24, 24, 3) training batches
                # 50000/10000 train/test split, no validation
                # Random crop, flip, standardization, no random brightness/contrast
                images, labels = cifar10_input.distorted_inputs('cifar-10-batches-bin', batch_size)
                best = 0
                for epoch in t:
                    for step in range(steps_per_epoch):
                        train_op = self.step(images, labels, global_step, learning_rate, .9, .004)
                        a_train, c_train, _ = sess.run(train_op) 
                    #  if epoch % 10 == 0:
                        #  a_validation, c_validation, _ = sess.run([accuracy, loss, optim],
                            #  feed_dict={X: data.validation.images, Y: data.validation.labels, keep_prob: 1.0})
                        #  if a_validation >= best:
                            #  saver.save(sess, checkpoint)
                    t.set_postfix(
                        t_acc=a_train,
                        t_loss=c_train,
                        v_acc=a_validation,
                        v_loss=c_validation
                    )
            if (early_stop):
                saver.restore(sess, checkpoint)
            sess.close

    def weight(self, name, shape):
        return tf.get_variable(
                name=name,
                shape=shape,
                initializer=tf.variance_scaling_initializer()
        )

    def bias(self, name, shape):
        return tf.get_variable(
                name=name,
                shape=shape,
                initializer=tf.constant_initializer(0.0)
        )

    def batch_norm(self, inputs, training=True):
        return tf.layers.batch_normalization(
            inputs=inputs,
            training=training,
            fused=True
        )

    def conv(self, inputs, filter, stride, bias):
        return tf.nn.relu(tf.nn.conv2d(
            input=inputs,
            filter=filter,
            strides=[1, stride, stride, 1],
            padding='SAME',
        ) + bias)

    def pool(self, inputs, kernel_size, stride):
        return tf.nn.max_pool(
            value=inputs,
            ksize=[1, kernel_size, kernel_size, 1],
            strides=[1, stride, stride, 1],
            padding='SAME'
        )

    def norm(self, inputs):
        return tf.nn.lrn(
            input=inputs,
            depth_radius=4,
            bias=1.0,
            alpha=0.001 / 9.0,
            beta=0.75
        )
    
    def fc(self, inputs, filter, bias):
        return tf.nn.relu(tf.matmul(inputs, filter) + bias)

if __name__ == '__main__':
    model = AlexNetMod()
    model.train(100)
