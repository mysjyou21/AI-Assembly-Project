import tensorflow as tf
import numpy as np


def drop_out(input, keep_prob, is_train):
    if is_train:
        out = tf.nn.dropout(input, keep_prob)
    else:
        keep_prob = 1
        out = tf.nn.dropout(input, keep_prob)
    return out


def _norm(input, is_train, reuse=True, norm=None):
    assert norm in ['instance', 'batch', None]
    if norm == 'instance':
        with tf.variable_scope('instance_norm', reuse=reuse):
            eps = 1e-5
            mean, sigma = tf.nn.moments(input, [1, 2], keep_dims=True)
            normalized = (input - mean) / (tf.sqrt(sigma) + eps)
            out = normalized
    elif norm == 'batch':
        with tf.variable_scope('batch_norm', reuse=reuse):
            out = tf.layers.batch_normalization(inputs=input, training=is_train, reuse=reuse)
    else:
        out = input
    return out


def _activation(input, activation=None):
    assert activation in ['relu', 'leaky', 'tanh', 'sigmoid', None]
    if activation == 'relu':
        return tf.nn.relu(input)
    elif activation == 'leaky':
        return tf.contrib.keras.layers.LeakyReLU(0.1)(input)
    elif activation == 'tanh':
        return tf.tanh(input)
    elif activation == 'sigmoid':
        return tf.sigmoid(input)
    elif activation == 'prelu':
        alphas = tf.get_variable('alpha',
                                 input.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(input)
        neg = alphas * (input - abs(input)) * 0.5
        return pos + neg
    else:
        return input


def pooling(input, k_size, stride, mode):
    assert mode in ['MAX', 'AVG']
    result = None
    if mode == 'MAX':
        result = tf.nn.max_pool(value=input,
                                ksize=[1, k_size[0], k_size[1], 1],
                                strides=[1, stride[0], stride[1], 1],
                                padding='SAME',
                                name='max_pooling')
    elif mode == 'AVG':
        result = tf.nn.avg_pool(value=input,
                                ksize=[1, k_size[0], k_size[1], 1],
                                strides=[1, stride[0], stride[1], 1],
                                padding='SAME',
                                name='avg_pooling')

    return result


def flatten(input):
    return tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])


def conv2d(input, num_filters, filter_size, stride, pad='SAME', dtype=tf.float32, bias=True):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, input.get_shape()[3], num_filters]
    w = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    if pad == 'REFLECT':
        p = (filter_size - 1) // 2
        x = tf.pad(input, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        conv = tf.nn.conv2d(x, w, stride_shape, padding='VALID')
    else:
        assert pad in ['SAME', 'VALID']
        conv = tf.nn.conv2d(input, w, stride_shape, padding=pad)

    if bias:
        b = tf.get_variable('b', [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0))
        conv = conv + b
    return conv


def conv_block(input, name, num_filters, k_size, stride, is_train, reuse, norm, activation, pad='SAME', bias=True):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d(input, num_filters, k_size, stride, pad, bias=bias)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out


def mlp(input, out_dim, name, is_train, reuse, norm=None, activation=None, dtype=tf.float32, bias=True):
    with tf.variable_scope(name, reuse=reuse):
        _, n = input.get_shape()
        w = tf.get_variable('w', [n, out_dim], dtype, tf.random_normal_initializer(0.0, 0.02))
        out = tf.matmul(input, w)
        if bias:
            b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))
            out = out + b
        out = _activation(out, activation)
        out = _norm(out, is_train, reuse, norm)
        return out

