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


class ResNet():
    def __init__(self, epochs=1000, learning_rate=.01, batch_size=32,
            early_stop=False, momentum=.9, weight_decay=.0001, reg=0):

        num_train = 50000
        steps_per_epoch = num_train // batch_size

        X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        Y = tf.placeholder(tf.float32, [None, 10])

        # Resnet-18 as described in https://arxiv.org/pdf/1512.03385.pdf,
        # but using the blocks from v2 from https://arxiv.org/abs/1603.05027

        # https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
        # http://torch.ch/blog/2016/02/04/resnets.html
        # Static learning rate / momentum

        # input -> 32x32
        # conv1 (f=3, s=1, k=64) -> 32x32
        # conv2.x (f=3, s=2, k=64) x 2 -> 16x16
        # conv3.x (f=3, s=2, k=128) x 2 -> 8x8
        # conv4.x (f=3, s=2, k=256) x 2 -> 4x4
        # avgpool
        # fc
        # softmax

        f = [3, 3, 3, 3]
        s = [1, 1, 1, 1]
        k = [64, 64, 128, 256]

        # conv1
        w1 = self.weight('w1', [f[0], f[0], 3, k[0]])
        b1 = self.bias('b1', [k[0]])
        X_ = self.conv2d(X, w1, s[0], b1)
        X_ = self.batch_norm(X_)

        # conv2.x
        w2 = self.weight('w2', [f[1], f[1], k[0], k[1]])
        b2 = self.bias('b2', [k[1]])
        X_ = self.res_block(X_, w2, s[1], b2)

        # conv3.x
        w3 = self.weight('w3', [f[2], f[2], k[1], k[2]])
        b3 = self.bias('b3', [k[2]])
        X_ = self.res_block(X_, w3, s[2], b3)

        # conv4.x
        w4 = self.weight('w4', [f[3], f[3], k[2], k[3]])
        b4 = self.bias('b4', [k[3]])
        X_ = self.res_block(X_, w4, s[3], b4)

        # global average pooling
        X_ = tf.reduce_mean(x, axis=[1,2])

        # fc
        w5 = self.weight('w5', [k[3], 10])
        b5 = self.bias('b5', [10])
        logits = self.fc(X_, w5, b5, 4 * 4 * k[3])

        pred = tf.nn.softmax(logits)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        l2 = tf.reduce_sum([tf.reduce_sum(tf.pow(w,2)) for w in (w1, w2, w3, w4, w5)])
        loss = loss + reg * l2

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        optim = tf.train.AdamOptimizer(learning_rate).minimize(loss)
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
                    a_train, c_train, _ = sess.run([accuracy, loss, optim],
                        feed_dict={X: images, Y: labels, keep_prob: dropout})
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

        #  a, c = sess.run([accuracy, loss],
            #  feed_dict={X:data.test.images, Y:data.test.labels, keep_prob: 1.0})
        #  print('test accuracy={}, test loss={}'.format(a, c))

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

    def conv2d(self, inputs, filter, stride, bias):
        return tf.nn.conv2d(
            input=inputs,
            filter=filter,
            strides=[1, stride, stride, 1],
            padding='SAME',
        ) + bias

    def res_block(self, inputs, filter, stride, bias, training=True):
        # batch norm -> relu -> conv -> batch norm -> relu -> conv
        # using identity shortcut right now (type A)
        shortcut = inputs
        inputs = self.batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)
        inputs = self.conv2d(inputs, filter, stride, bias)
        inputs = self.batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)
        inputs = self.conv2d(inputs, filter, stride, bias)
        return inputs + shortcut

    def fc(self, inputs, filter, bias):
        fc = tf.reshape(inputs, shape=[-1, 1])
        return tf.nn.relu(tf.matmul(fc, filter) + bias)


#  class AlexNet():
        # started with alexnet since its simple and old so it trains
        #   decently fast on a GTX860M
        # decrease learning rate over time
        # input
        # conv1 (f=11, s=4, k=96 relu)
        # pool1 (f=3, s=2)
        # norm1
        # conv2 (f=5, s=1, k=256 relu)
        # pool2 (f=3, s=2)
        # norm2
        # conv3 (f=3, s=1, k=384 relu)
        # conv4 (f=3, s=1, k=384 relu)
        # conv5 (f=3, s=1, k=256 relu)
        # pool3 (f=3, s=2)
        # dropout (.5)
        # fc1 (relu)
        # fc2 (relu)
        # fc3 (relu)
        # filters in each convolutional layer

class Cifar10:
    def __init__(self, path, one_hot=True, batch_size=32):
        train_files = ['{}/data_batch_{}'.format(path, i) for i in range(1,5)]
        X_train, y_train = [], []
        for train_file in train_files:
            with open(train_file, 'rb') as f:
                ddict = pickle.load(f, encoding='bytes')
                X_train += ddict[b'data']
                y_train += ddict[b'labels']
        image, label = np.array(X_train), np.array(y_train)
        self.train = tf.train.shuffle_batch([image, label], batch_size=batch_size)
        image, label = np.array(X_train), np.array(y_train)
        self.train = tf.train.shuffle_batch([image, label], batch_size=batch_size)

        test_file = '{}/test_batch'.format(path)
        with open(test_file, 'rb') as f:
            ddict = pickle.load(f, encoding='bytes')
            self.test = np.array(ddict[b'data']), np.array(ddict[b'labels'])

    def next_batch(shuffle=True, batch_size=32):
        pass

if __name__ == '__main__':
    model = ResNet()
