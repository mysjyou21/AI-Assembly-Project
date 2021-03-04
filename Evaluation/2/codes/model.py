import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import numpy as np
import sys

sys.path.append('./miscc')
from ops import *
from ops import _weight

H,W = 224,224

def CNN(x, name='image', reuse=False, is_train=False, regularizer=None):
    with tf.variable_scope(name+'_CNN', reuse=reuse):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            nets, end_points = resnet_v1.resnet_v1_50(x, is_training=is_train)  # nets: after pool5
        feature = flatten(nets, 2048)
        return feature

def METRIC(x, ft_channel=32, name='image', reuse=False, is_train=False, regularizer=None):
    with tf.variable_scope(name+'_METRIC', reuse=reuse):
        in_channel = 2048
        x = fc(x, in_channel//2, 'fc1', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
        x = fc(x, in_channel//4, 'fc2', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
        x = fc(x, in_channel//8, 'fc3', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
        x = fc(x, in_channel//16, 'fc4', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
        x = fc(x, ft_channel, 'fc5', norm=None, activation=None, is_train=is_train, regularizer=regularizer)

        return x

def CLASSIFIER(x, class_num, name='image', reuse=False, is_train=False, regularizer=None):
    with tf.variable_scope(name+'_CLASSIFIER', reuse=reuse):
        in_channel=32
        x = fc(x, class_num, 'fc1', norm=None, activation=None, is_train=is_train, regularizer=regularizer)

        return x

def VIEWPOOL(views, ft_channel, trans_feature, batch_size=5, name='view', _type='mean', reuse=False, view_num=24):
    """ views: a list of features [Bx2048, Bx2048, ...]"""
    in_channel = 2048
    with tf.variable_scope('VIEWPOOL', reuse=reuse):
        vp = tf.expand_dims(views[0], 0) # 1xBx2048
        for v in views[1:]:
            v = tf.expand_dims(v, 0)
            vp = tf.concat([vp, v], 0) # VxBx2048
        if _type=='mean':
            vp = tf.reduce_mean(vp, 0, name=name) # Bx2048
        elif _type=='max':
            vp = tf.reduce_max(vp, 0, name=name)
        elif _type=='attention':
            # Loung attention 
            vpp = tf.transpose(vp, [1,0,2])
            Wh = _weight('attention_key', [in_channel, ft_channel], initialize='constant', constant=1/ft_channel)
            h = tf.matmul(vpp, tf.tile(tf.expand_dims(Wh, 0), [batch_size,1,1])) # VxBxft_channel (key)
            scores = tf.matmul(tf.expand_dims(trans_feature,1), h, transpose_b=True)
            scores = tf.squeeze(scores, 1)
            scores = tf.nn.softmax(scores, axis=1)
            scores = tf.tile(tf.expand_dims(scores,2), [1,1,in_channel])
            scores = tf.transpose(scores, [1,0,2])
            vp = tf.multiply(vp, scores)
            vp = tf.reduce_sum(vp, 0, name=name)
    return vp

def TRANSFORM(x, ft_channel, name='trans', reuse=False, regularizer=None):
    with tf.variable_scope(name, reuse=reuse):
        x = fc(x, ft_channel//2, 'fc1', activation='relu', regularizer=regularizer)
        x = fc(x, ft_channel//4, 'fc2', activation='relu', regularizer=regularizer)
        x = fc(x, ft_channel//2, 'fc3', activation='relu', regularizer=regularizer) ##
        x = fc(x, ft_channel, 'fc4', activation=None)

        return x


def VIEWS(views, ft_channel=0, trans_feature=None, batch_size=5, name='view', _type='mean', reuse_t=False, is_train=False, regularizer=None, view_num=24):
    """ views: BxVxHxWx3 (B: batch size, V: # of views for each model """
    V = views.get_shape().as_list()[1]

    # BxVxHxWx3 -> VxBxHxWx3
    views = tf.transpose(views, perm=[1,0,2,3,4])

    vp = []
    for i in range(V):
        reuse=(i!=0) or reuse_t
        view = tf.gather(views, i) # BxHxWx3 for i-th view

        view_feature = CNN(view, name=name, reuse=reuse, is_train=is_train, regularizer=regularizer) # Bx2048

        vp.append(view_feature)

    vp = VIEWPOOL(vp, ft_channel, trans_feature, batch_size, name=name, _type=_type, reuse=reuse_t, view_num=view_num)

    return vp


