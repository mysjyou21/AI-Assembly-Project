import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import vgg
from pose_ops import * 


#=================
#     MODEL
#=================

VGG = True

if VGG:

    # --------- minwoo ------------
    def vgg_arg_scope(weight_decay=0.1):
      with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          biases_initializer=tf.zeros_initializer(),
                          ):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
          return arg_sc

    def vgg_16(inputs, is_training=True,
              reuse=False,
              scope='vgg_16',):
      with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
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
          
          return net

    def PRETRAINED_CNN_F(x, name='img_n_views', reuse=False, is_train=False, regularizer=None):
        with tf.variable_scope(name + '_PRETRAINED_CNN', reuse=reuse):
            with slim.arg_scope(vgg_arg_scope()):
                nets = vgg_16(x, is_training=is_train)
        return nets

    
    # --------- slim ------------    
    def PRETRAINED_CNN_F(x, name='img_n_views', reuse=False, is_train=False, regularizer=None):
        with tf.variable_scope(name + '_PRETRAINED_CNN', reuse=reuse):
            with slim.arg_scope(vgg.vgg_arg_scope()):
                nets, end_points = vgg.vgg_16(x, is_training=is_train)
        feature = end_points[name + '_PRETRAINED_CNN/vgg_16/conv5/conv5_3']
        return feature

    # --------- custom ------------
    def PRETRAINED_CNN(x, name='img_n_views', reuse=False, is_train=False, regularizer=None):
        with tf.variable_scope(name + '_PRETRAINED_CNN', reuse=reuse):
            x = conv2d(x, 64, 3, 1, name='conv1_1', norm='batch', activation='relu', is_train=is_train, regularizer=regularizer)
            x = conv2d(x, 64, 3, 1, name='conv1_2', norm='batch', activation='relu', is_train=is_train, regularizer=regularizer)
            x = pool(x, 2, 2, name='pool1', type='max')
            x = conv2d(x, 128, 3, 1, name='conv2_1', norm='batch', activation='relu', is_train=is_train, regularizer=regularizer)
            x = conv2d(x, 128, 3, 1, name='conv2_2', norm='batch', activation='relu', is_train=is_train, regularizer=regularizer)
            x = pool(x, 2, 2, name='pool2', type='max')
            x = conv2d(x, 256, 3, 1, name='conv3_1', norm='batch', activation='relu', is_train=is_train, regularizer=regularizer)
            x = conv2d(x, 256, 3, 1, name='conv3_2', norm='batch', activation='relu', is_train=is_train, regularizer=regularizer)
            x = pool(x, 2, 2, name='pool3', type='max')
            x = conv2d(x, 512, 3, 1, name='conv4_1', norm='batch', activation='relu', is_train=is_train, regularizer=regularizer)
            x = conv2d(x, 512, 3, 1, name='conv4_2', norm='batch', activation='relu', is_train=is_train, regularizer=regularizer)
            x = pool(x, 2, 2, name='pool4', type='max')
            x = conv2d(x, 512, 3, 1, name='conv5_1', norm='batch', activation='relu', is_train=is_train, regularizer=regularizer)
            x = conv2d(x, 512, 3, 1, name='conv5_2', norm='batch', activation='relu', is_train=is_train, regularizer=regularizer)
            x = pool(x, 2, 2, name='pool5', type='max')
        return x

    def ONE_BY_ONE(x, name='img_n_views', reuse=False, is_train=False, regularizer=None):
        with tf.variable_scope(name + '_ONE_BY_ONE', reuse=reuse):
            x = conv2d(x, 128, 1, 1, name='conv1', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
        return x

    def ONE_BY_ONE_AGAIN(x, name='img_n_views', reuse=False, is_train=False, regularizer=None):
        with tf.variable_scope(name + '_ONE_BY_ONE_AGAIN', reuse=reuse):
            x = conv2d(x, 128, 1, 1, name='conv1', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
        return x

    def FC(x, name='img_n_views', reuse=False, is_train=False, regularizer=None):
        with tf.variable_scope(name + '_FC', reuse=reuse):
            x = tf.layers.flatten(x)
            x = fc(x, 128, 'fc4', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
            x = fc(x, 1, 'fc5', norm=None, activation=None, is_train=is_train, regularizer=regularizer)
        return x

    def MODEL(x, view_num=48, name='img_n_views', is_train=False, regularizer=None):
        for i in range(2):
            input_image = tf.gather(x, i)
            input_image = tf.expand_dims(input_image, axis=0)
            reuse = (i!=0)
            image_feature = PRETRAINED_CNN(input_image, name=name, reuse=tf.AUTO_REUSE, is_train=is_train, regularizer=regularizer)
            image_feature = ONE_BY_ONE(image_feature, name=name, reuse=tf.AUTO_REUSE, is_train=is_train, regularizer=regularizer)
            t = image_feature if i==0 else tf.concat([t, image_feature], axis=-1)
        t = ONE_BY_ONE_AGAIN(t, name=name, reuse=tf.AUTO_REUSE, is_train=is_train, regularizer=regularizer)
        c = FC(t, name=name, reuse=tf.AUTO_REUSE, is_train=is_train, regularizer=regularizer)
        return c

    def map_ftn(ckpt_path, graph_vars):
        """ Load pre-trained Resnet-50 model using mapping function {ckpt_vars(old): graph_vars(new)} """
        ckpt_vars = tf.train.list_variables(ckpt_path)
        ckpt_vars = [v[0] for v in ckpt_vars if ('vgg_16' in v[0] and 'logits' not in v[0] and 'mean_rgb' not in v[0] and 'fc' not in v[0])]

        # assert len(ckpt_vars) == len(graph_vars), f'{len(ckpt_vars)} {len(graph_vars)}'
        
        for v in graph_vars:
            map_v = ''
            for w in ckpt_vars:
                if w in v.op.name:
                    map_v = w
                    break
            with tf.device('/cpu:0'):
                tf.train.init_from_checkpoint(ckpt_path, {map_v: v})

else:

    def PRETRAINED_CNN(x, name='img_n_views', reuse=False, is_train=False, regularizer=None):
        with tf.variable_scope(name + '_PRETRAINED_CNN', reuse=reuse):
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                nets, end_points = resnet_v1.resnet_v1_50(x, is_training=is_train)
        feature = end_points[name + '_PRETRAINED_CNN/resnet_v1_50/block3']
        return feature


    def ONE_BY_ONE(x, name='img_n_views', reuse=False, is_train=False, regularizer=None):
        with tf.variable_scope(name + '_ONE_BY_ONE', reuse=reuse):
            x = conv2d(x, 128, 1, 1, name='conv1', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
        return x

    def ONE_BY_ONE_AGAIN(x, name='img_n_views', reuse=False, is_train=False, regularizer=None):
        with tf.variable_scope(name + '_ONE_BY_ONE_AGAIN', reuse=reuse):
            x = conv2d(x, 128, 1, 1, name='conv1', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
        return x

    def FC(x, name='img_n_views', reuse=False, is_train=False, regularizer=None):
        with tf.variable_scope(name + '_FC', reuse=reuse):
            x = tf.layers.flatten(x)
            x = fc(x, 128, 'fc4', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
            x = fc(x, 2, 'fc5', norm=None, activation=None, is_train=is_train, regularizer=regularizer)
        return x


    def MODEL(x, view_num=48, name='img_n_views', reuse=False, is_train=False, regularizer=None):
        for i in range(3):
            input_image = tf.gather(x, i)
            input_image = tf.expand_dims(input_image, axis=0)
            reuse = (i!=0)
            image_feature = PRETRAINED_CNN(input_image, name=name, reuse=reuse, is_train=is_train, regularizer=regularizer)
            image_feature = ONE_BY_ONE(image_feature, name=name, reuse=reuse, is_train=is_train, regularizer=regularizer)
            t = image_feature if i==0 else tf.concat([t, image_feature], axis=-1)
        t = ONE_BY_ONE_AGAIN(t, name=name, reuse=False, is_train=is_train, regularizer=regularizer)
        c = FC(t, name=name, reuse=False, is_train=is_train, regularizer=regularizer)
        return c

    def map_ftn(ckpt_path, graph_vars):
        """ Load pre-trained Resnet-50 model using mapping function {ckpt_vars(old): graph_vars(new)} """
        ckpt_vars = tf.train.list_variables(ckpt_path)
        ckpt_vars = [v[0] for v in ckpt_vars if ('resnet_v1_50' in v[0] and 'logits' not in v[0] and 'mean_rgb' not in v[0])]

        # assert len(ckpt_vars) == len(graph_vars)

        for v in graph_vars:
            map_v = ''
            for w in ckpt_vars:
                if w in v.op.name:
                    map_v = w
                    break
            with tf.device('/cpu:0'):
                tf.train.init_from_checkpoint(ckpt_path, {map_v: v})




