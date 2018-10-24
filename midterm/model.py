#!/usr/bin/python3.6

### Kevin Sheng
### ECE471 Selected Topics in Machine Learning - Midterm Project

import os
import argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
import blocks


class Model():
    def __init__(self, data, train=False, save=False):
        epochs = 200
        learning_rate = .001
        momentum = .9
        weight_decay = .0005
        batch_size = 100 
        early_stop = False
        num_train = 55000
        
        f = [5, 5]
        k = [20, 50, 500]
        
        X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        y = tf.placeholder(tf.float32, [None, 10])
        
        # conv1
        conv1 = blocks.L0Conv2d('conv1', [f[0], f[0], 1, k[0]], weight_decay=weight_decay)
        # conv2
        conv2 = blocks.L0Conv2d('conv2', [f[1], f[1], k[0], k[1]], weight_decay=weight_decay)
        # fc1, after 2 maxpools
        fc1 = blocks.L0Dense('fc1', [7*7*k[1], k[2]], weight_decay=weight_decay)
        # fc2
        fc2 = blocks.L0Dense('fc2', [k[2], 10], weight_decay=weight_decay)
       
        layers = (conv1, conv2, fc1, fc2)

        global_step = tf.train.get_or_create_global_step()
        # the paper doesn't bias on convs for some reason
        X_ = blocks.conv(X, conv1.sample_weights(), 1, conv1.bias)
        X_ = blocks.pool(X_, 2, 2)
        X_ = blocks.conv(X_, conv2.sample_weights(), 1, conv2.bias)
        X_ = blocks.pool(X_, 2, 2)
        X_ = blocks.dense(X_, fc1.sample_weights(), fc1.bias)
        logits = blocks.dense(X_, fc2.sample_weights(), fc2.bias, activation=False)
        
        pred = tf.nn.softmax(logits)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        # 1.26 million weights
        # l0 = [tf.count_nonzero(w, dtype=tf.float32) for w in layers]
        #  l2 = tf.reduce_sum([tf.reduce_sum(tf.pow(w,2))
                #  for w in decay_params])
        l0 = [l.count_l0() for l in layers]
        reg = tf.reduce_sum([- (1/num_train) * l.regularization() for l in layers])
        loss = loss + reg

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        #  learning_rate.assign(learning_rate_base*(1+.0001*global_step)^-.75)
        optim = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        saver = tf.train.Saver()
        checkpoint = 'checkpoints/model.ckpt'
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            try:
                saver.restore(sess, 'checkpoints/model.ckpt.best')
            except:
                pass

            if not train:
                a_test, l_test = sess.run([accuracy, loss],
                feed_dict={X:data.test.images, y:data.test.labels})
                print('test accuracy: {} / test loss: {}.'.format(a_test, l_test))
                return

            best = 0 
            current_epoch = 0
            step_in_epoch = 0
            a_total, l_total = 0, 0
            a_val, l_val = 0, 0
            with tqdm(total=epochs * num_train // batch_size) as t:
                t.update(0)
                while True:
                    data_train, labels_train = data.train.next_batch(batch_size)
                    a, l, o, s, params = sess.run([accuracy, loss, optim, global_step, l0],
                            feed_dict={X: data_train, y: labels_train}) 
                    for layer in layers:
                        layer.constrain_parameters()
                    epochs_completed = data.train.epochs_completed
                    total_epochs = s * batch_size // num_train
                    # grab the next batch of data
                    t.update(s - t.n)
                    a_total += a
                    l_total += l
                    step_in_epoch += 1

                    t.set_postfix(
                        step=s,
                        epoch=total_epochs,
                        l0=params,
                        t_acc=a_total / step_in_epoch,
                        t_loss=l_total / step_in_epoch,
                        v_acc=a_val,
                        v_loss=l_val,
                    )
                    
                    # check validation loss every complete epoch
                    if epochs_completed > current_epoch:
                        a, l = sess.run([accuracy, loss],
                        feed_dict={X: data.validation.images, y: data.validation.labels})
                        if save:
                            if a >= best:
                                saver.save(sess, 'checkpoints/model.ckpt.best')
                                best = a
                            if epochs_completed % 10 == 0:
                                saver.save(sess, 'checkpoints/model.ckpt.' + str(epochs_completed))
                            saver.save(sess, checkpoint)
                        #  saver.save(sess, 'model.{}.ckpt'.format(current_epoch))
                        t.set_postfix(
                            step=s,
                            epoch=total_epochs,
                            l0=params,
                            t_acc=a_total / data.train._index_in_epoch,
                            t_loss=l_total / data.train._index_in_epoch,
                            v_acc=a,
                            v_loss=l,
                        )
                        
                        a_total = 0
                        l_total = 0
                        step_in_epoch = 0
                        current_epoch = epochs_completed 

                        if total_epochs >= epochs:
                            break
    

if __name__ == '__main__':
    # some cuda issues
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='')
    parser.add_argument('--save', action='store_true', help='')

    args = parser.parse_args()

    data = mnist.read_data_sets("data", one_hot=True, reshape=False)
    model = Model(data, train=args.train, save=args.save)
