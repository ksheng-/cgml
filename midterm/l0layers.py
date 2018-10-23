import numpy as np
import tensorflow as tf

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

def l0_dense():

def l0_conv2d():
    pass

def cdf_qz(x):
    xn = (x - limit_a) / (limit_b - limit_a)
    logits = tf.log(xn) - tf.log(1 - xn)
    return tf.clip_by_value(tf.nn.sigmoid(logits * self.temperature - self.qz_loga), epsilon, 1 - epsilon)

def quantile_concrete(x):
    y = tf.nn.sigmoid(tf.log(x) - tf.log(1-x) + qz_loga) / temperature)
    return y * (limit_b - limit_a) + limit_a
