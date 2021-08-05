import numpy as np
import tensorflow as tf


def conv2d(input, out_c, k, s, name, padding='SAME', norm=None, activation=None, is_train=False, regularizer=None):
    with tf.variable_scope(name):
        w = _weight('weights', [k, k, input.get_shape()[3], out_c], regularizer=regularizer)
        b = _bias('bias', [out_c,])
        conv = tf.nn.conv2d(input, w, [1, s, s, 1], padding, name='conv') + b

        norm_conv = _norm(norm, conv, is_train=is_train)

        if activation is None:
            act_conv = norm_conv
        elif activation == 'relu':
            act_conv = tf.nn.relu(norm_conv, name='conv')
        elif activation == 'lrelu':
            act_conv = tf.nn.leaky_relu(norm_conv, name='conv')
        elif activation == 'relu6':
            act_conv = tf.nn.relu6(norm_conv, name='conv')
        elif activation == 'elu':
            act_conv = tf.nn.elu(norm_conv, name='conv')

    return act_conv


def pool(input, k, s, name, type='max', padding='SAME'):
    with tf.variable_scope(name):
        if type=='max':
            pool = tf.nn.max_pool(input, [1, k, k, 1], [1, s, s, 1], padding=padding, name='max_pool')
        elif type=='avg':
            pool = tf.nn.avg_pool(input, [1, k, k, 1], [1, s, s, 1], padding=padding, name='avg_pool')
        elif type=='globalavg':
            pool = tf.layers.average_pooling2d(input, k, s, padding=padding, name=name)

    return pool



def fc(input, out_c, name, norm=None, activation=None, is_train=False, regularizer=None):
    with tf.variable_scope(name):
        w = _weight('weights', [input.get_shape()[1], out_c], regularizer=regularizer)
        b = _bias('bias', [out_c])
        fc = tf.matmul(input, w) + b

        norm_fc = _norm(norm, fc, is_train=is_train)

        if activation is None:
            act_fc = fc
        elif activation == 'sigmoid':
            act_fc = tf.nn.sigmoid(fc, name='fc')
        elif activation == 'tanh':
            act_fc = tf.nn.tanh(fc, name='fc')
        elif activation == 'softmax':
            act_fc = tf.nn.softmax(fc, name='fc')
        elif activation == 'relu':
            act_fc = tf.nn.relu(fc, name='fc')
        elif activation == 'lrelu':
            act_fc = tf.nn.leaky_relu(fc, name='fc')
        elif activation == 'relu6':
            act_fc = tf.nn.relu6(fc, name='fc')

    return act_fc


def flatten(input, dim):
    return tf.reshape(input, [-1, dim])


def _weight(name, shape, regularizer=None):
    var = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=False), regularizer=regularizer)

    return var


def _bias(name, shape, constant=0.0):
    var = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(constant))

    return var


def dropout(x, keep_prob=None, noise_shape=None, seed=None, name=None, rate=None):
    return tf.nn.dropout(x, keep_prob=keep_prob, noise_shape=noise_shape, seed=seed, name=name, rate=rate)


def _norm(norm, x, is_train=True, group_num=32, momentum=0.99, epsilon=0.001):
    if norm is None:
        return x
    elif norm == 'batch':
        norm = tf.layers.batch_normalization(x, training=is_train, momentum=momentum, epsilon=epsilon, name='norm')
    elif norm == 'instance':
        norm = tf.contrib.layers.instance_norm(x, scope='instance_norm') #reuse..?how..?
    elif norm == 'group':
        norm = tf.contrib.layers.group_norm(x, group=group_num, scope='group_norm') #reuse..?
    elif norm == 'batch_instance':
        norm = batch_instance_norm(x, scope='batch_instance_norm')

    return norm