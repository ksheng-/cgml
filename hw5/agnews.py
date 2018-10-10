#!/usr/bin/python3.6

### Kevin Sheng
### ECE471 Selected Topics in Machine Learning - Assignment 4 

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import agnews_input

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class Model():
    def __init__(self, dataset, eval=False):

        epochs = 10
        learning_rate = .1
        momentum =.9
        batch_size = 128
        early_stop = False
        num_train = dataset.num_train 
        
        f = [256, 256, 256, 256, 256, 256]
        k = [7, 7, 3, 3, 3, 3, 1024, 1024]
        
        X = tf.placeholder(tf.float32, [batch_size, 1014, 26])
        y = tf.placeholder(tf.float32, [batch_size, 4])
        
        # conv1
        w1 = self.weight('w1', [f[0], 26, k[0]])
        b1 = self.bias('b1', [k[0]])
        # conv2
        w2 = self.weight('w2', [f[1], k[0], k[1]])
        b2 = self.bias('b2', [k[1]])
        # conv3
        w3 = self.weight('w3', [f[2], k[1], k[2]])
        b3 = self.bias('b3', [k[2]])
        # conv4
        w4 = self.weight('w4', [f[3], k[2], k[3]])
        b4 = self.bias('b4', [k[3]])
        # conv5
        w5 = self.weight('w5', [f[4], k[3], k[4]])
        b5 = self.bias('b5', [k[4]])
        # conv6
        w6 = self.weight('w6', [f[5], k[4], k[5]])
        b6 = self.bias('b6', [k[5]])
        # fc1, after 3 maxpools
        w7 = self.weight('w7', [37*k[5], k[6]])
        b7 = self.bias('b7', [k[6]])
        # fc2
        w8 = self.weight('w8', [k[6], k[7]])
        b8 = self.bias('b8', k[7])
        # fc3
        w9 = self.weight('w9', [k[7], 4])
        b9 = self.bias('b9', [4])
        
        self.params = (w1, w2, w3, w4, w5, w6)

        global_step = tf.train.get_or_create_global_step()
        X_ = self.conv(X, w1, 1, b1)
        X_ = self.pool(X_, 3, 3)
        X_ = self.conv(X_, w2, 1, b2)
        X_ = self.pool(X_, 3, 3)
        X_ = self.conv(X_, w3, 1, b3)
        X_ = self.conv(X_, w4, 1, b4)
        X_ = self.conv(X_, w5, 1, b5)
        X_ = self.conv(X_, w6, 1, b6)
        X_ = self.pool(X_, 3, 3)
        X_ = self.fc(X_, w7, b7)
        X_ = tf.nn.dropout(X_, .5)
        X_ = self.fc(X_, w8, b8)
        X_ = tf.nn.dropout(X_, .5)
        logits = self.fc(X_, w9, b9)
        
        pred = tf.nn.softmax(logits)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        #  l2 = tf.reduce_sum([tf.reduce_sum(tf.pow(w,2)) for w in self.params])
        #  loss = loss + weight_decay * l2

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        optim = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step=global_step)

        saver = tf.train.Saver()
        checkpoint = 'checkpoints/model.ckpt'
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            try:
                saver.restore(sess, checkpoint)
            except:
                pass

            if eval:
                data_test, labels_test = dataset.get_test_set()
                a_test, l_test, _ = sess.run([accuracy, loss, optim],
                    feed_dict={X: data_test, y: labels_test})
                print('test accuracy: {} / test loss: {}.'.format(a_test, l_test))

            best = np.inf 
            step_in_epoch = 0
            current_epoch = 0
            a_total, l_total = 0, 0
            a_val, l_val = 0, 0
            data_val, labels_val = dataset.get_validation_set()
            with tqdm(total=epochs) as t:
                while True:
                    # grab the next batch of data
                    data_train, labels_train, training_epoch = dataset.next_batch(batch_size)

                    # validate at the end of each epoch
                    if training_epoch > current_epoch:
                        a_val, l_val, _ = sess.run([accuracy, loss, optim],
                            feed_dict={X: data_val, y: labels_val})
                        if l_val < best:
                            saver.save(sess, checkpoint)
                            best = l_val
                        saver.save(sess, 'model.{}.ckpt'.format(current_epoch))
                        t.set_postfix(
                            step=step,
                            t_acc=a_total / step_in_epoch,
                            t_loss=l_total / step_in_epoch,
                            v_acc=a_val,
                            v_loss=l_val,
                        )
                        
                        a_total = 0
                        l_total = 0
                        step_in_epoch = 0
                        data_val, labels_val = dataset.get_validation_set()
                        current_epoch = training_epoch
                        t.update(current_epoch)
                    
                    if current_epoch >= epochs:
                        break

                    a, l, o, s = sess.run([accuracy, loss, optim, global_step],
                        feed_dict={X: data_train, y: labels_train}) 
                    a_total += a
                    l_total += l
                    step_in_epoch += 1

                    t.set_postfix(
                        step=s,
                        t_acc=a_total / step_in_epoch,
                        t_loss=l_total / step_in_epoch,
                        v_acc=a_val,
                        v_loss=l_val,
                    )
    
    def weight(self, name, shape):
        return tf.get_variable(
                name=name,
                shape=shape,
                initializer=tf.truncated_normal_initializer(0.0, 0.05)
                #  initializer=tf.variance_scaling_initializer()
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

    def conv(self, value, filters, stride, bias):
        return tf.nn.relu(tf.nn.conv1d(
            value=value,
            filters=filters,
            stride=stride,
            padding='SAME',
        ) + bias)

    def pool(self, inputs, pool_size, strides):
        return tf.layers.max_pooling1d(
            inputs=inputs,
            pool_size=pool_size,
            strides=strides,
            padding='VALID'
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
        return tf.nn.relu(tf.matmul(inputs, filter) + bias)

if __name__ == '__main__':
    model = Model(agnews_input.DataSet())