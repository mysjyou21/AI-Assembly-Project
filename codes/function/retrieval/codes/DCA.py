import os
import sys
import matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf
import shutil
import cv2

from .model_stefan import CNN, METRIC, VIEWPOOL, TRANSFORM, VIEWS, DISCRIMINATOR, CLASSIFIER
from .model_stefan import map_ftn
from function.ops import *
from function.utils import normalize_feature, calculate_distance
import json

class DCA(object):

    def __init__(self, args, max_epoch=1, C=10, K=1, view_num=12, size=(224,224), is_train=False, mode=-1, testmode=0):
        self.args = args
        self.checkpoint_dir = self.args.args.model_path + '/retrieval/ckpt'

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
        self.test_f1 = transf
        self.test_f2 = viewf
        same = False
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


    def accuracy(self, img_lab, view_lab, matched_indices):
        gt = np.equal(np.expand_dims(img_lab, 1), np.expand_dims(view_lab, 0))  # BxB

        img_indices = np.asarray([x for x in range(0, gt.shape[0])])
        pred = np.zeros_like(gt)
        pred[img_indices, matched_indices] = 1

        accurate = gt * pred
        accurate = np.sum(accurate, axis=1)

        return np.sum(accurate), accurate.shape[0]


    def matching(self, features1, features2, same=False):
        """ find the most closest feature of features1 among features2
            feature1: BxC
            feature2: BxC """
        norm_features1 = normalize_feature(features1)
        norm_features2 = normalize_feature(features2)
        dist = calculate_distance(features1, features2, same)

        matched_indices = tf.argmin(dist, axis=1)

        return matched_indices, norm_features1, norm_features2


    def test(self, args, page_num, test_batch, retrieval_gt, lab2name_gt):
        NPY_DIR = self.args.args.data_path + '/retrieval'
        self.test_view = np.load(NPY_DIR + '/test_view.npy')
        self.test_view_lab = np.load(NPY_DIR + '/test_view_lab.npy')
        with open(NPY_DIR + '/lab2name.json') as json_file:
            self.lab2name = json.load(json_file)

        b_imgs = test_batch
        b_views = self.test_view
        b_views_lab = self.test_view_lab

        with args.graph_retrieval.as_default():
            sess = args.sess_retrieval
            feed_dict = {self.img: b_imgs, self.view: b_views, self.view_lab: b_views_lab}
            b_transf, b_viewf, indices = sess.run([self.test_f1, self.test_f2, self.matched_indices], feed_dict=feed_dict)
            results_indices = self.test_view_lab[indices]
            
            # Label - Name matching correction
            retrieval_results_num = []
            retrieval_gt_num = []
            correct = 0

            for j in range(len(indices)):
                result_label = self.lab2name[str(results_indices[j])]
                result_idx = lab2name_gt.index(result_label)
                retrieval_results_num.append(result_idx)
            retrieval_results_name = [lab2name_gt[x] for x in retrieval_results_num]
            
            if len(retrieval_gt) != 0:
                for j in range(len(indices)):
                    gt_label = retrieval_gt[page_num][j]
                    gt_idx = lab2name_gt.index(gt_label)
                    retrieval_gt_num.append(gt_idx)
                correct = np.sum(np.equal(np.array(retrieval_results_num),np.array(retrieval_gt_num)).astype(int))
            if self.args.args.retrieval_visualization:

                if len(retrieval_gt) != 0:
                    FIG_DIR = self.args.args.intermediate_results_path + '/retrieval/unit_test'
                    if not os.path.exists(FIG_DIR):
                        os.makedirs(FIG_DIR)
                        
                    rows = 1
                    cols = 2

                    for idx in range(len(retrieval_results_num)):
                        img_idx = retrieval_gt_num[idx]
                        view_idx = retrieval_results_num[idx]
                        img_label = lab2name_gt[img_idx]
                        view_label = lab2name_gt[view_idx]
                        fig = plt.figure()
                        i = 1
                        sample_img = b_imgs[idx,:,:,:]
                        ax1 = fig.add_subplot(rows,cols,i)
                        ax1.set_xlabel(img_label)
                        plt.imshow(sample_img)

                        i += 1
                        sample_view = b_views[indices[idx],0,:,:,:]
                        ax2 = fig.add_subplot(rows,cols,i)
                        ax2.set_xlabel(view_label)
                        plt.imshow(sample_view)

                        if img_idx == view_idx:
                            fig_path = FIG_DIR + '/correct_' + str(page_num) + '_' + str(idx) + '.png'
                        else:
                            fig_path = FIG_DIR + '/wrong_' + str(page_num) + '_' + str(idx) + '.png'

                        plt.savefig(fig_path,dpi=300)

                else:
                    FIG_DIR = self.args.args.intermediate_results_path + '/retrieval'
                    if not os.path.exists(FIG_DIR):
                        os.makedirs(FIG_DIR)
                        
                    rows = 1
                    cols = 2

                    for idx in range(len(retrieval_results_num)):
                        view_idx = retrieval_results_num[idx]
                        view_label = lab2name_gt[view_idx]
                        fig = plt.figure()
                        i = 1
                        sample_img = b_imgs[idx,:,:,:]
                        ax1 = fig.add_subplot(rows,cols,i)
                        plt.imshow(sample_img)

                        i += 1
                        sample_view = b_views[indices[idx],0,:,:,:]
                        ax2 = fig.add_subplot(rows,cols,i)
                        ax2.set_xlabel(view_label)
                        plt.imshow(sample_view)
                        fig_path = FIG_DIR + '/' + str(page_num) + '_' + str(idx) + '.png'

                        plt.savefig(fig_path,dpi=300)
                        plt.close()

            return retrieval_results_name, retrieval_results_num, correct

    def resize_and_pad(self, comp, a=150):
        # find object region
        non_zero = np.nonzero(255 - comp)
        y_min = np.min(non_zero[0])
        y_max = np.max(non_zero[0])
        x_min = np.min(non_zero[1])
        x_max = np.max(non_zero[1])
        comp = comp[y_min:y_max + 1, x_min:x_max + 1]

        ## change data_type from uint8 to float? ##
        # resize to 150, 150
        long_side = np.max(comp.shape)
        ratio = a / long_side
        comp = cv2.resize(comp, dsize=(0, 0), fx=ratio, fy=ratio, interpolation = cv2.INTER_AREA)
        # pad to 224, 224
        pad_left = int(np.ceil((224 - comp.shape[1]) / 2))
        pad_right = int(np.floor((224 - comp.shape[1]) / 2))
        pad_top = int(np.ceil((224 - comp.shape[0]) / 2))
        pad_bottom = int(np.floor((224 - comp.shape[0]) / 2))
        comp = cv2.copyMakeBorder(comp, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, None, [255, 255, 255])
        ## change data_type from uint8 to float? ##
        return comp
