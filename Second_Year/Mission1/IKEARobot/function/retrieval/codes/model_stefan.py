from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
# from tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import vgg

import numpy as np
import sys
# sys.path.append('./miscc')
from miscc.ops import *

slim = contrib_slim
H,W = 224,224
""" VGG Network Based"""
def vgg_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

def vgg_16(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          reuse=None,
          scope='vgg_16',
          fc_conv_padding='VALID',
          global_pool=False):
  with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      net = slim.repeat(net, 1, slim.conv2d, 32, [3, 3], scope='conv6')

      return net

"""Make CNN Architecture"""
def CNN(x, name='image', reuse=False, is_train=False, regularizer=None):
    with tf.variable_scope(name+'_CNN', reuse=reuse):

        with slim.arg_scope(vgg_arg_scope()):
            nets = vgg_16(x , is_training=is_train)
        # print("nets shape: {}".format(nets.get_shape()))
        feature = flatten(nets, 1568)
        return feature

def METRIC(x, name='image', reuse=False, is_train=False, regularizer=None):
    with tf.variable_scope(name+'_METRIC', reuse=reuse):
        x = fc(x, 1024, 'fc1', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
        x = fc(x, 512, 'fc2', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
        x = fc(x, 256, 'fc3', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
        x = fc(x, 128, 'fc4', norm=None, activation=None, is_train=is_train, regularizer=regularizer)
        # print("feature shape: {}".format(x.get_shape()))
        return x

def CLASSIFIER(x, class_num, name='image', reuse=False, is_train=False, regularizer=None):
    with tf.variable_scope(name+'_CLASS', reuse=reuse):
        x = fc(x, class_num, 'fc1', norm=None, activation=None, is_train=is_train, regularizer=regularizer)

        return x

def VIEWPOOL(views, class_num, trans_feature, batch_size=5, name='view', _type='mean', reuse=False, view_num=24):
    """ views: a list of features [Bx2048, Bx2048, ...]"""
    in_channel = 1568
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
            Wh = _weight('attention_key', [in_channel, class_num], initialize='constant', constant=1/class_num)
            h = tf.matmul(vpp, tf.tile(tf.expand_dims(Wh, 0), [batch_size,1,1])) # VxBxclass_num (key)
            # Watt = _weight('attention_score', [class_num, class_num])
            scores = tf.matmul(tf.expand_dims(trans_feature,1), h, transpose_b=True)
#            scores = tf.matmul(tf.expand_dims(trans_feature,1), tf.matmul(tf.tile(tf.expand_dims(Watt,0), [batch_size,1,1]), h, transpose_b=True)) # BxV
            scores = tf.squeeze(scores, 1)
            scores = tf.nn.softmax(scores, axis=1)
            scores = tf.tile(tf.expand_dims(scores,2), [1,1,in_channel])
            scores = tf.transpose(scores, [1,0,2])
            vp = tf.multiply(vp, scores)
            vp = tf.reduce_sum(vp, 0, name=name)
    return vp

def TRANSFORM(x, name='trans', reuse=False, regularizer=None):
    with tf.variable_scope(name, reuse=reuse):
        x = fc(x, 64, 'fc1', activation='lrelu', regularizer=regularizer)
        x = fc(x, 32, 'fc2', activation='lrelu', regularizer=regularizer)
        x = fc(x, 64, 'fc3', activation='lrelu', regularizer=regularizer)
        x = fc(x, 128, 'fc4', activation=None, regularizer=regularizer)

        return x

def DISCRIMINATOR(x, name='discriminator', reuse=False, regularizer=None):
    with tf.variable_scope(name, reuse=reuse):
        x = fc(x, 128, 'fc1', activation='lrelu', regularizer=regularizer)
        x = fc(x, 64, 'fc2', activation='lrelu', regularizer=regularizer)
        x = fc(x, 1, 'fc3')

        return x

def VIEWS(views, class_num=0, trans_feature=None, batch_size=5, name='view', _type='mean', reuse_t=False, is_train=False, regularizer=None, view_num=24):
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

    vp = VIEWPOOL(vp, class_num, trans_feature, batch_size, name=name, _type=_type, reuse=tf.AUTO_REUSE, view_num=view_num)
    # print(vp)

    return vp

def map_ftn(ckpt_path, graph_vars):
    """ Load pre-trained Resnet-50 model using mapping function {ckpt_vars(old): graph_vars(new)} """
    ckpt_vars = tf.train.list_variables(ckpt_path)
    ckpt_vars = [v[0] for v in ckpt_vars if ('resnet_v1_50' in v[0] and 'logits' not in v[0] and 'mean_rgb' not in v[0])]

    assert len(ckpt_vars) == len(graph_vars)

    for v in graph_vars:
        map_v = ''
        for w in ckpt_vars:
            if w in v.op.name:
                map_v = w
                break
        tf.train.init_from_checkpoint(ckpt_path, {map_v: v})
