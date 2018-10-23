#!/usr/bin/python3.6

### Kevin Sheng
### ECE471 Selected Topics in Machine Learning - Midterm Project

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.contrib.learn.python.learn.datasets import mnist

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class Model():
    def __init__(self, data, eval=False):
        epochs = 200
        learning_rate = .001
        momentum = .9
        weight_decay = .0005
        batch_size = 16 
        early_stop = False
        num_train = 55000
        
        f = [5, 5]
        k = [20, 50, 500]
        
        X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        y = tf.placeholder(tf.float32, [None, 10])
        #  learning_rate = tf.Variable(learning_rate_base, tf.float32)
        # conv1
        w1 = self.weight('w1', [f[0], f[0], 1, k[0]])
        b1 = self.bias('b1', [k[0]])

        # conv2
        w2 = self.weight('w2', [f[1], f[1],  k[0], k[1]])
        b2 = self.bias('b2', [k[1]])

        # fc1, after 2 maxpools
        w3 = self.weight('w3', [7*7*k[1], k[2]])
        b3 = self.bias('b3', [k[2]])
        
        # fc2
        w4 = self.weight('w4', [k[2], 10])
        b4 = self.bias('b4', 10)
       
        decay_params = (w1, w2, w3, w4)

        global_step = tf.train.get_or_create_global_step()
        X_ = self.conv(X, w1, 1, b1)
        X_ = self.pool(X_, 2, 2)
        X_ = self.conv(X_, w2, 1, b2)
        X_ = self.pool(X_, 2, 2)
        X_ = self.fc(X_, w3, b3)
        logits = self.fc(X_, w4, b4, activation=False)
        
        pred = tf.nn.softmax(logits)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        # 1.26 million weights
        l0 = [tf.count_nonzero(w, dtype=tf.float32) for w in decay_params]
        l2 = tf.reduce_sum([tf.reduce_sum(tf.pow(w,2))
                for w in decay_params])
        loss = loss + weight_decay * l2

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        #  learning_rate.assign(learning_rate_base*(1+.0001*global_step)^-.75)
        optim = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        saver = tf.train.Saver()
        checkpoint = 'checkpoints/model.ckpt'
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            try:
                saver.restore(sess, checkpoint)
            except:
                pass

            if eval:
                a_test, l_test = sess.run([accuracy, loss],
                feed_dict={X:data.test.images, y:data.test.labels})
                print('test accuracy: {} / test loss: {}.'.format(a_test, l_test))
                return

            best = 0 
            current_epoch = 0
            step_in_epoch = 0
            a_total, l_total = 0, 0
            a_val, l_val = 0, 0
            with tqdm(total=epochs) as t:
                t.update(0)
                while True:
                    data_train, labels_train = data.train.next_batch(batch_size)
                    a, l, o, s, params = sess.run([accuracy, loss, optim, global_step, l0],
                            feed_dict={X: data_train, y: labels_train}) 
                    
                    epochs_completed = data.train.epochs_completed
                    # grab the next batch of data
                    t.update(epochs_completed)
                    a_total += a
                    l_total += l
                    step_in_epoch += 1

                    t.set_postfix(
                        step=s,
                        epoch=epochs_completed,
                        l0=params,
                        t_acc=a_total / step_in_epoch,
                        t_loss=l_total / step_in_epoch,
                        v_acc=a_val,
                        v_loss=l_val,
                    )
                    

                    # check validation loss every complete epoch
                    if epochs_completed > current_epoch:
                        a, l  = sess.run([accuracy, loss],
                        feed_dict={X: data.validation.images, y: data.validation.labels})
                        if a >= best:
                            saver.save(sess, 'checkpoints/model.ckpt.best')
                            best = a
                        saver.save(sess, checkpoint)
                        #  saver.save(sess, 'model.{}.ckpt'.format(current_epoch))
                        t.set_postfix(
                            step=s,
                            epoch=epochs_completed,
                            l0=l0,
                            t_acc=a_total / data.train._index_in_epoch,
                            t_loss=l_total / data.train._index_in_epoch,
                            v_acc=a_val,
                            v_loss=l_val,
                        )
                        
                        a_total = 0
                        l_total = 0
                        step_in_epoch = 0
                        current_epoch = epochs_completed 

                        if epochs_completed >= epochs:
                            break
    
    def weight(self, name, shape):
        return tf.get_variable(
                name=name,
                shape=shape,
                initializer=tf.contrib.layers.xavier_initializer()
        )

    def bias(self, name, shape):
        return tf.get_variable(
                name=name,
                shape=shape,
                initializer=tf.constant_initializer(0.0)
        )

    def conv(self, inputs, filter, stride, bias):
        return tf.nn.conv2d(
            input=inputs,
            filter=filter,
            strides=[1, stride, stride, 1],
            padding='SAME',
        ) + bias

    def pool(self, inputs, kernel_size, stride):
        return tf.nn.max_pool(
            value=inputs,
            ksize=[1, kernel_size, kernel_size, 1],
            strides=[1, stride, stride, 1],
            padding='SAME'
        )
 
    def fc(self, inputs, filter, bias, activation=True):
        inputs = tf.layers.flatten(inputs)
        if activation:
            return tf.nn.relu(tf.matmul(inputs, filter) + bias)
        else:
            return tf.matmul(inputs, filter) + bias

if __name__ == '__main__':
    data = mnist.read_data_sets("data", one_hot=True, reshape=False)
    model = Model(data, eval=False)
