#!/usr/bin/python3.6

### Kevin Sheng
### ECE471 Selected Topics in Machine Learning - Assignment 4 

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py
import cifar100_input

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class Cifar100():
    def __init__(self, eval=False):
        # pretty standard/simple DNN loosely based on alexnet since its
        #   simple and old so it trains decently fast on a GTX860M
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

        epochs=100
        learning_rate=.00000001
        batch_size=16
        early_stop=False
        num_train = 50000
        
        f = [3, 3, 3, 3]
        k = [32, 64, 128, 128, 512]
        
        # conv1
        w1 = self.weight('w1', [f[0], f[0], 3, k[0]])
        b1 = self.bias('b1', [k[0]])
        
        # conv2
        w2 = self.weight('w2', [f[1], f[1], k[0], k[1]])
        b2 = self.bias('b2', [k[1]])
        
        # conv3
        w3 = self.weight('w3', [f[2], f[2], k[1], k[2]])
        b3 = self.bias('b3', [k[2]])
        
        # conv4
        w4 = self.weight('w4', [f[3], f[3], k[2], k[3]])
        b4 = self.bias('b4', [k[3]])
        
        # fc1
        w5 = self.weight('w5', [8*8*k[3], k[4]])
        b5 = self.bias('b5', [k[4]])
        
        # fc2
        w6 = self.weight('w6', [k[4], 100])
        b6 = self.bias('b6', [100])
        
        # linear
        #  w5 = self.weight('w5', [k[3], 10])
        #  b5 = self.bias('b5', [10])
        self.params = (w1, w2, w3, w4, w5, w6)

        if eval:
            images, labels = cifar100_input.inputs(
                True, 'cifar-100-binary', 1000
            )
        else:
            images, labels = cifar100_input.distorted_inputs(
                'cifar-100-binary', batch_size
            )
        s = [1, 1, 1, 1]

        global_step = tf.train.get_or_create_global_step()
        X_ = self.conv(images, w1, s[0], b1)
        X_ = self.conv(X_, w2, s[1], b2)
        X_ = self.pool(X_, 2, 2)
        #  X_ = tf.nn.dropout(X_, .25)
        #  X_ = self.batch_norm(X_)
        X_ = self.conv(X_, w3, s[2], b3)
        X_ = self.conv(X_, w4, s[3], b4)
        X_ = self.pool(X_, 2, 2)
        #  X_ = tf.nn.dropout(X_, .25)
        #  X_ = self.batch_norm(X_)
        X_ = tf.nn.relu(self.fc(X_, w5, b5))
        X_ = tf.nn.dropout(X_, .5)
        logits = self.fc(X_, w6, b6)
        
        pred = tf.nn.softmax(logits)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        #  l2 = tf.reduce_sum([tf.reduce_sum(tf.pow(w,2)) for w in self.params])
        #  loss = loss + weight_decay * l2

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        optim = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        steps_per_epoch = num_train // batch_size
        
        saver = tf.train.Saver()
        checkpoint = 'checkpoints/model.cifar100.v5.ckpt'
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
            try: 
                saver.restore(sess, checkpoint)
            except:
                pass
            if eval:
                total_accuracy = 0
                total_loss = 0 
                for batch in range(10):
                    a, l, g = sess.run([accuracy, loss, global_step])
                    total_accuracy += a
                    total_loss += l
                print('global_step {} (epoch {}): test accuracy={}, test loss={}'.format(g, g // steps_per_epoch, total_accuracy / 10, total_loss / 10))
                coord.request_stop()
                coord.join(threads)
                return
            
            with tqdm(range(steps_per_epoch * epochs)) as t:
                best = 0
                for step in t:
                    epoch, step_in_epoch = divmod(step, steps_per_epoch)
                    if step_in_epoch == 0:
                        saver.save(sess, checkpoint)
                        total_accuracy = 0
                        total_loss = 0
                    
                    a, l, o, g = sess.run([accuracy, loss, optim, global_step]) 
                    total_accuracy += a
                    total_loss += l
                    
                    t.set_postfix(
                        epoch=g // steps_per_epoch,
                        step=step_in_epoch,
                        acc=total_accuracy / step_in_epoch,
                        loss=total_loss / step_in_epoch,
                    )

            if early_stop:
                saver.restore(sess, checkpoint)
            coord.request_stop()
            coord.join(threads)
    
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
        inputs = tf.reshape(inputs, [inputs.get_shape().as_list()[0], -1])
        return tf.matmul(inputs, filter) + bias

if __name__ == '__main__':
    model = Cifar100(eval=True)
