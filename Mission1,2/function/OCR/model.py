from OCR.ops import *


def digit_recognizer(x, is_train, reuse=False):
    """Simplest model for assembly number classification"""
    # input: [batch_size, None, None, 1]
    # architecture: 512 fc - 512 fc - 10 fc
    with tf.variable_scope('model1'):
        x = flatten(x)
        out = mlp(x, 512, 'fc1', is_train, reuse, norm=None, activation='relu')
        out = mlp(out, 512, 'fc2', is_train, reuse, norm=None, activation='relu')
        out = mlp(out, 10, 'fc3', is_train, reuse, norm=None, activation=None)
    return out

'''
def digit_recognizer(x, is_train, reuse=False):
    """Simplest model for assembly number classification"""
    # input: [batch_size, None, None, 1]\
    with tf.variable_scope('model1'):
        out = conv_block(x, 'conv1', 32, 3, 1, is_train, False, None, 'relu')
        out = pooling(out, [2, 2], [2, 2], 'MAX')

        out = conv_block(out, 'conv2', 64, 3, 1, is_train, False, 'batch', 'relu')
        out = pooling(out, [2, 2], [2, 2], 'MAX')

        out = conv_block(out, 'conv3', 128, 3, 1, is_train, False, 'batch', 'relu')
        out = pooling(out, [2, 2], [2, 2], 'MAX')

        out = conv_block(out, 'conv4', 256, 3, 1, is_train, False, 'batch', 'relu')
        out = pooling(out, [2, 2], [2, 2], 'MAX')

        out = flatten(out)
        out = mlp(out, 512, 'fc9', is_train, False, 'batch', 'relu')
        out = mlp(out, 10, 'fc10', is_train, False, None, None)

    return out
'''


def CE_loss(logit, label):
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(label, logit))
    return loss


def focal_loss(logit, label, gamma):
    logit = tf.nn.softmax(logit)
    eps = 1e-9
    weight = tf.pow(1. - logit, gamma)
    loss = -tf.reduce_sum(weight * label * tf.log(logit + eps))
    return loss

def MSE_loss(logit, label):
    #logit = tf.nn.softmax(logit)
    loss = tf.reduce_mean(tf.square(logit - label))
    return loss

def tf_get_accuracy(logit, label):
    #logit = tf.nn.softmax(logit)
    pred = tf.argmax(logit, axis=1)
    ans = tf.argmax(label, axis=1)
    equal = tf.cast(tf.equal(pred, ans), dtype=tf.float32)
    acc = tf.reduce_mean(equal)
    return acc
