import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import numpy as np
import sys

sys.path.append('./miscc')
from ops import *

H,W = 224,224

def CNN(x, name='image', reuse=False, is_train=False):
    with tf.variable_scope(name+'_CNN', reuse=reuse):
#        feature_extractor = tf.keras.applications.ResNet50(include_top=False, input_shape=(H,W,3))
#        feature_extractor.trainable = True
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            nets, end_points = resnet_v1.resnet_v1_50(x, is_training=is_train)  # nets: after pool5
#        feature = end_points['resnet_v1_50/block4']  # None,7,7,2048
        feature = flatten(nets, 2048)  # before pooling 5..? but start channel-number is 2048

        return feature

def METRIC(x, name='image', reuse=False, is_train=False):
    with tf.variable_scope(name+'_METRIC', reuse=reuse):
        x = fc(x, 1024, 'fc1', norm='batch', activation='relu', is_train=is_train)
        x = fc(x, 512, 'fc2', norm='batch', activation='relu', is_train=is_train)
        x = fc(x, 256, 'fc3', norm='batch', activation='relu', is_train=is_train)
        x = fc(x, 128, 'fc4', norm=None, activation='tanh', is_train=is_train)

        return x

def VIEWPOOL(views, name='view', _type='mean', reuse=False):
    """ views: a list of features [Bx2048, Bx2048, ...]"""
    with tf.variable_scope(name+'_VIEWPOOL', reuse=reuse):
        vp = tf.expand_dims(views[0], 0) # 1xBx2048
        for v in views[1:]:
            v = tf.expand_dims(v, 0)
            vp = tf.concat([vp, v], 0) # VxBx2048
        if _type=='mean':
            vp = tf.reduce_mean(vp, 0, name=name) # Bx2048
        elif _type=='max':
            vp = tf.reduce_max(vp, 0, name=name)
    return vp

def TRANSFORM(x, name='trans', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        x = fc(x, 128, 'fc1', activation='relu')
        x = fc(x, 64, 'fc2', activation='relu')
        x = fc(x ,32, 'fc3', activation='relu')
        x = fc(x, 64, 'fc4', activation='relu')
        x = fc(x, 128, 'fc5', activation='tanh')

        return x

def DISCRIMINATOR(x, name='discriminator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        x = fc(x, 64, 'fc1', activation='relu')
        x = fc(x, 1, 'fc2')

        return x

def VIEWS(views, name='view', _type='mean', reuse_t=False, is_train=False):
    """ views: BxVxHxWx3 (B: batch size, V: # of views for each model """
    V = views.get_shape().as_list()[1]

    # BxVxHxWx3 -> VxBxHxWx3
    views = tf.transpose(views, perm=[1,0,2,3,4])

    vp = []
    for i in range(V):
        reuse=(i!=0) or reuse_t
        view = tf.gather(views, i) # BxHxWx3 for i-th view

        view_feature = CNN(view, name=name, reuse=reuse, is_train=is_train) # Bx2048

        vp.append(view_feature)

    vp = VIEWPOOL(vp, name=name, _type=_type, reuse=reuse_t)

    return vp

