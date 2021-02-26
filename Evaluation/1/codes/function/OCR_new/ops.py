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


def conv2d_transpose(input, num_filters, filter_size, stride, pad='SAME', dtype=tf.float32):
    n, h, w, c = input.get_shape().as_list()
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, num_filters, c]
    w = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))

    input_shape = tf.shape(input)
    try:  # tf pre-1.0 (top) vs 1.0 (bottom)
        output_shape = tf.pack([input_shape[0], stride * input_shape[1], stride * input_shape[2], num_filters])
    except Exception as e:
        output_shape = tf.stack([input_shape[0], stride * input_shape[1], stride * input_shape[2], num_filters])

    deconv = tf.nn.conv2d_transpose(input, w, output_shape, stride_shape, pad)
    return deconv


def bilstm(bottom_sequence, sequence_length, cell_size):
    """Build bidirectional (concatenated output) RNN layer"""
    weight_initializer = tf.truncated_normal_initializer(stddev=0.01)
    cell_fw = tf.contrib.rnn.LSTMCell(cell_size, initializer=weight_initializer)
    cell_bw = tf.contrib.rnn.LSTMCell(cell_size, initializer=weight_initializer)
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, bottom_sequence,
        sequence_length=sequence_length,
        time_major=True,
        dtype=tf.float32)
    rnn_output_stack = tf.concat(rnn_output, 2, name='output_stack')
    return rnn_output_stack


def bigru(bottom_sequence, sequence_length, cell_size):
    """Build bidirectional (concatenated output) RNN layer"""
    weight_initializer = tf.truncated_normal_initializer(stddev=0.01)
    cell_fw = tf.contrib.rnn.GRUCell(cell_size)
    cell_bw = tf.contrib.rnn.GRUCell(cell_size)
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, bottom_sequence,
        sequence_length=sequence_length,
        time_major=True,
        dtype=tf.float32)
    rnn_output_stack = tf.concat(rnn_output, 2, name='output_stack')
    return rnn_output_stack


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

def conv_block(input, name, num_filters, k_size, stride, is_train, reuse, norm, activation, pad='SAME', bias=True):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d(input, num_filters, k_size, stride, pad, bias=bias)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out


def residual_block(input,  name, num_filters,  is_train, reuse, norm, activation, pad='SAME', bias=True):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('res1', reuse=reuse):
            out = conv2d(input, num_filters, 3, 1, pad, bias=bias)
            out = _norm(out, is_train, reuse, norm)
            out = _activation(out, activation)

        with tf.variable_scope('res2', reuse=reuse):
            out = conv2d(out, num_filters, 3, 1, pad, bias=bias)
            out = _norm(out, is_train, reuse, norm)
            out = _activation(out + input, activation)
        return out

def inception_block(input, name, num_filters, is_train, reuse):
    num_filters = int(num_filters / 4)
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('incep_1', reuse=reuse):
            out_1_1 = conv_block(input, 'incep_1_1', num_filters, 1, 1, is_train, reuse, None, 'relu', pad='SAME')

        with tf.variable_scope('incep_3', reuse=reuse):
            out_3_1 = conv_block(input, 'incep_3_1', num_filters, 1, 1, is_train, reuse, None, 'relu', pad='SAME')
            out_3_3 = conv_block(out_3_1, 'incep_3_3', num_filters, 3, 1, is_train, reuse, None, 'relu', pad='SAME')

        with tf.variable_scope('incep_5', reuse=reuse):
            out_5_1 = conv_block(input, 'incep_5_1', num_filters, 1, 1, is_train, reuse, None, 'relu', pad='SAME')
            out_5_5 = conv_block(out_5_1, 'incep_5_5', num_filters, 5, 1, is_train, reuse, None, 'relu', pad='SAME')

        with tf.variable_scope('incep_max', reuse=reuse):
            out_max = pooling(input, [3, 3], [1, 1], 'MAX')
            out_max_1 = conv_block(out_max, 'incep_max', num_filters, 1, 1, is_train, reuse, None, 'relu', pad='SAME')

        out = tf.concat([out_1_1, out_3_3, out_5_5, out_max_1], 3)

    return out

def transposed_conv_block(input, name, num_filters, k_size, stride, is_train, reuse, norm, activation):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d_transpose(input, num_filters, k_size, stride)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out


def bilstm_block(input, name, sequence_length, cell_size, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        out = bilstm(input, sequence_length, cell_size)
        return out


def bigru_block(input, name, sequence_length, cell_size, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        out = bigru(input, sequence_length, cell_size)
        return out
