import os
import sys
import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
import shutil
import cv2

from model_stefan import CNN, METRIC, VIEWPOOL, TRANSFORM, VIEWS, DISCRIMINATOR, CLASSIFIER
from model_stefan import map_ftn
from dataset import Dataset
# sys.path.append('./miscc')
from ops import *
from miscc.utils import normalize_feature, calculate_distance

class DCA(object):

    def __init__(self, args, max_epoch=1, C=16, K=1, view_num=12, size=(224,224), is_train=False, mode=-1, testmode=0):
        self.DATA_DIR = args.opt.assembly_path #'./input/stefan'
        self.checkpoint_dir = args.opt.retrieval_model_path #'./model/retrieval/ckpt'
        self.args = args

        self.max_epoch = max_epoch #1
        self.size = size
        self.H = self.W = size[0] #img size 224
        self.C = C #16
        self.K = K #1
        self.V = view_num #12
        self.is_train = False
        self.mode = -1
        self.testmode = 0

        self.img = tf.placeholder(tf.float32, [None, self.H, self.W, 3])
        self.view = tf.placeholder(tf.float32, [None, self.V, self.H, self.W, 3])
        self.view_lab = tf.placeholder(tf.int32, [None])

        imgf, transf, viewf = self.build_model(args, self.img, self.view)

        # [B], matched_indices[i] = the most appropriate (to i-th transf feature) feature among viewf features
        self.test_f1 = imgf if self.testmode == 1 else viewf if self.testmode == 2 else transf
        self.test_f2 = imgf if self.testmode == 1 else viewf if self.testmode == 2 else transf if self.testmode == 3 else viewf
        same = False if self.testmode == 0 else True
        self.matched_indices, _, _ = self.matching(self.test_f1, self.test_f2, same=same)

        # restore ckpt & graph
        t_vars = tf.trainable_variables()
        trans_vars = [v for v in t_vars if 'trans' in v.name]
        saver_I_CNN_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='image_CNN')
        saver_I_metric_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='image_METRIC')
        saver_I = tf.train.Saver(var_list=saver_I_CNN_vars + saver_I_metric_vars, max_to_keep=2)
        saver_V_CNN_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='view_CNN')
        saver_V_metric_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='view_METRIC')
        saver_V = tf.train.Saver(var_list=saver_V_CNN_vars + saver_V_metric_vars, max_to_keep=2)
        saver_T = tf.train.Saver(var_list=trans_vars, max_to_keep=5)

        dir_I = self.checkpoint_dir + '/Image'
        dir_V = self.checkpoint_dir + '/View'
        dir_T = self.checkpoint_dir + '/Trans'


        # restore models
        sess = args.sess_retrieval
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        if tf.train.latest_checkpoint(dir_I) is not None:
            print('RETRIEVAL MODEL : Loading saved model from %s' % tf.train.latest_checkpoint(dir_I))
            saver_I.restore(sess, tf.train.latest_checkpoint(dir_I))

        if tf.train.latest_checkpoint(dir_V) is not None:
            print('RETRIEVAL MODEL : Loading saved model from %s' % tf.train.latest_checkpoint(dir_V))
            saver_V.restore(sess, tf.train.latest_checkpoint(dir_V))

        if tf.train.latest_checkpoint(dir_T) is not None:
            print('RETRIEVAL MODEL : Loading saved model from %s' % tf.train.latest_checkpoint(dir_T))
            saver_T.restore(sess, tf.train.latest_checkpoint(dir_T))


    def build_model(self, args, img, views):
        """ img: BxHxWx3
            views: BxVxHxWx3 """
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

        # img feature
        img_feature_cnn = CNN(img, name='image', is_train=self.is_train, regularizer=self.regularizer, reuse=tf.AUTO_REUSE)
        img_feature_logit = METRIC(img_feature_cnn, name='image', is_train=self.is_train, regularizer=self.regularizer, reuse=tf.AUTO_REUSE)
        with tf.variable_scope('image_METRIC'):
            img_feature = tf.nn.tanh(img_feature_logit)

        # img2view transformation(generator)
        trans_feature_logit = TRANSFORM(img_feature, name='trans', regularizer=self.regularizer, reuse=tf.AUTO_REUSE)
        with tf.variable_scope('trans'):
            trans_feature = tf.nn.tanh(trans_feature_logit)

        # view features
        view_feature_cnn = VIEWS(views, name='view', _type='mean', is_train=self.is_train, regularizer=self.regularizer, reuse_t=tf.AUTO_REUSE)
        view_feature_logit = METRIC(view_feature_cnn, name='view', is_train=self.is_train, regularizer=self.regularizer, reuse=tf.AUTO_REUSE)
        with tf.variable_scope('view_METRIC'):
            view_feature = tf.nn.tanh(view_feature_logit)

        return img_feature, trans_feature, view_feature


    def matching(self, features1, features2, same=False):
        """ find the most closest feature of features1 among features2
            feature1: BxC
            feature2: BxC """
        norm_features1 = normalize_feature(features1)
        norm_features2 = normalize_feature(features2)
        dist = calculate_distance(features1, features2, same)

        matched_indices = tf.argmin(dist, axis=1)

        return matched_indices, norm_features1, norm_features2


    def test(self, args, step_num, candidate_classes):
        self.step = step_num
        retrieved_class_names = []
        init_candidate_classes = candidate_classes.copy()
        # print("Init Candidate Classe: ", init_candidate_classes)
        for img_index in range(len(self.args.parts[step_num])):
            self.DataProvider = Dataset(args, self.step, candidate_classes, self.DATA_DIR, self.size, self.V)
            BatchProvider = self.DataProvider.test_batch(img_index)

            with args.graph_retrieval.as_default():
                sess = args.sess_retrieval
                global_step = 0
                b_imgs, b_views, b_views_lab, step_class_id = next(BatchProvider)
                feed_dict = {self.img: b_imgs, self.view: b_views, self.view_lab: b_views_lab}
                b_transf, b_viewf, indices = sess.run([self.test_f1, self.test_f2, self.matched_indices], feed_dict=feed_dict)

                for i in range(len(indices)):
                    retrieved_class_name = step_class_id[indices[i]]
                    retrieved_class_names.append(step_class_id[indices[i]])
                    candidate_classes.remove(retrieved_class_name)
                    # print("Updated Candidate Classe: ", candidate_classes)
        return retrieved_class_names
