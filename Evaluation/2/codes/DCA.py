import tensorflow as tf
import sys
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import shutil
import cv2
from sklearn.metrics import confusion_matrix

from model import CNN, METRIC, VIEWPOOL, TRANSFORM, VIEWS, CLASSIFIER
from dataset import Dataset
sys.path.append('./miscc')
from ops import *
from utils import normalize_feature, calculate_distance

class DCA(object):
    def __init__(self, FLAGS):
        self.DATA_DIR = FLAGS.data_dir
        if len(FLAGS.img_size.split(',')) > 1:
            self.H, self.W = FLAGS.img_size.split(',')
            self.H = int(self.H)
            self.W = int(self.W)
        else:
            self.H = self.W = int(FLAGS.img_size)
        self.DataProvider = Dataset(self.DATA_DIR, view_num=FLAGS.view_num, mode='test')
        self.len = self.DataProvider.class_len
        self.C = FLAGS.C
        self.ft_channel = FLAGS.feature_channel
        self.is_train = False
        self.mode = -1
        self.max_epoch = FLAGS.max_epoch

        self.V = FLAGS.view_num

        self.checkpoint_dir = FLAGS.checkpoint_dir
        self.testmode = FLAGS.testmode
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
        config.gpu_options.allow_growth = True


    def build_model(self, img, views):
        """ img: BxHxWx3
            views: BxVxHxWx3 """
        class_num = self.DataProvider.class_len
        ft_channel = self.ft_channel
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        # img feature
        img_feature_logit = CNN(img, name='image', is_train=self.is_train, regularizer=None)
        img_feature_logit = METRIC(img_feature_logit, ft_channel, name='image', is_train=self.is_train, regularizer=None)
        with tf.variable_scope('image_METRIC'):
            img_feature = tf.nn.tanh(img_feature_logit)
        img_feature_class = CLASSIFIER(img_feature, class_num, name='image', is_train=self.is_train, regularizer=None)

        # img2view transformation(generator)
        trans_feature_logit = TRANSFORM(img_feature, ft_channel, name='trans', regularizer=self.regularizer)
        with tf.variable_scope('trans'):
            trans_feature = tf.nn.tanh(trans_feature_logit)
        trans_feature_class = CLASSIFIER(trans_feature, class_num, name='trans', is_train=self.is_train, regularizer=None)

        # view features
        view_feature = VIEWS(views, ft_channel, trans_feature=trans_feature, batch_size=self.C, name='view', _type='attention' if (self.testmode==0 or self.testmode==4) else 'mean', is_train=self.is_train, regularizer=None, view_num=self.V)
        view_feature_logit = METRIC(view_feature, ft_channel, name='view', is_train=self.is_train, regularizer=self.regularizer)
        with tf.variable_scope('view_METRIC'):
            view_feature = tf.nn.tanh(view_feature_logit)
        view_feature_class = CLASSIFIER(view_feature, class_num, name='view', is_train=self.is_train, regularizer=None)


        return img_feature, trans_feature, view_feature, img_feature_logit, view_feature_logit, trans_feature_logit, img_feature_class, view_feature_class, trans_feature_class

    def matching(self, features1, features2, same=False):
        """ find the most closest feature of features1 among features2
            feature1: BxC
            feature2: BxC """
        norm_features1 = normalize_feature(features1)
        norm_features2 = normalize_feature(features2)
        dist = calculate_distance(features1, features2, same) # BxB

        matched_indices = tf.argmin(dist, axis=1)

        return matched_indices, norm_features1, norm_features2

    def accuracy(self, img_lab, view_lab, matched_indices):
        gt = np.equal(np.expand_dims(img_lab, 1), np.expand_dims(view_lab, 0)) #BxB

        img_indices = np.asarray([x for x in range(0,gt.shape[0])])
        pred = np.zeros_like(gt)
        pred[img_indices, matched_indices] = 1

        accurate = gt*pred
        accurate = np.sum(accurate, axis=1)

        return np.sum(accurate), accurate.shape[0], accurate

    def test(self):
        self.C = self.len
        self.K = 1

        tf.reset_default_graph()
        BatchProvider = self.DataProvider.make_batch(self.max_epoch, C=self.C, K=self.K, size=(self.H, self.W), is_shuffle=False)

        # define graph
        img = tf.placeholder(tf.float32, [None, self.H, self.W, 3])
        img_lab = tf.placeholder(tf.int32, [None])
        view = tf.placeholder(tf.float32, [None, self.V, self.H, self.W, 3])
        view_lab = tf.placeholder(tf.int32, [None])

        imgf, transf, viewf, imgf_logit, viewf_logit, transf_logit, imgf_class, viewf_class, transf_class = self.build_model(img, view)

        # [B], matched_indices[i] = the most appropriate (to i-th transf feature) feature among viewf features
        test_f1 = imgf if self.testmode==1 else viewf if self.testmode==2 else transf
        test_f2 = imgf if self.testmode==1 else viewf if self.testmode==2 else transf if self.testmode==3 else viewf
        same = False if self.testmode==0 else True
        matched_indices, _, _ = self.matching(test_f1, test_f2, same=same)
        test_f1_lab = view_lab if self.testmode==2 else img_lab
        test_f2_lab = img_lab if (self.testmode==1 or self.testmode==3) else view_lab

        if self.testmode != 0:
            test_f = imgf_class if self.testmode==1 else viewf_class if self.testmode==2 else transf_class
            predict = tf.nn.softmax(test_f, axis=1)
            predict = tf.argmax(predict, axis=1)

            acc = tf.reduce_sum(tf.cast(tf.math.equal(predict, tf.cast(test_f1_lab, tf.int64)), tf.int64))

        # for restoring models
        t_vars = tf.trainable_variables()
        self.trans_vars = [v for v in t_vars if 'trans' in v.name]
        saver_I_CNN_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='image_CNN')
        saver_I_metric_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='image_METRIC')
        saver_I_class_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='image_CLASSIFIER')
        saver_I = tf.train.Saver(var_list=saver_I_CNN_vars + saver_I_metric_vars + saver_I_class_vars, max_to_keep=40)
        saver_V_CNN_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='view_CNN')
        saver_V_metric_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='view_METRIC')
        saver_V_class_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='view_CLASSIFIER')
        saver_V = tf.train.Saver(var_list=saver_V_CNN_vars+saver_V_metric_vars+saver_V_class_vars, max_to_keep=40)
        if self.testmode==0:
            saver_VP = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='VIEWPOOL'), max_to_keep=2)
        saver_T = tf.train.Saver(var_list=self.trans_vars, max_to_keep=40)

        dir_I = self.checkpoint_dir+'/Image'
        dir_V = self.checkpoint_dir+'/View'
        dir_VP = self.checkpoint_dir+'/ViewPool'
        dir_T = self.checkpoint_dir+'/Trans'

        # Session configuration
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            # restore models
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            if tf.train.latest_checkpoint(dir_I) is not None:
                saver_I.restore(sess, tf.train.latest_checkpoint(dir_I))
                print('Load saved model from %s' % tf.train.latest_checkpoint(dir_I))
            if tf.train.latest_checkpoint(dir_V) is not None:
                saver_V.restore(sess, tf.train.latest_checkpoint(dir_V))
                print('Load saved model from %s' % tf.train.latest_checkpoint(dir_V))
            if tf.train.latest_checkpoint(dir_VP) is not None and (self.testmode==0 or self.testmode==4):
                saver_VP.restore(sess, tf.train.latest_checkpoint(dir_VP))
                print('Load saved model from %s' % tf.train.latest_checkpoint(dir_VP))
            if tf.train.latest_checkpoint(dir_T) is not None:
                saver_T.restore(sess, tf.train.latest_checkpoint(dir_T))
                print('Load saved model from %s' % tf.train.latest_checkpoint(dir_T))

            epoch = 0
            global_step = 0
            total_acc = 0
            total_num = 0
            steps = self.max_epoch*self.DataProvider.img_len//self.DataProvider.class_len
            w_true = []
            w_pred = []

            # for visualization
            FIG_DIR = self.checkpoint_dir+'/Test'
            if os.path.exists(FIG_DIR):
                shutil.rmtree(FIG_DIR)
            if not os.path.exists(FIG_DIR):
                os.makedirs(FIG_DIR)
            f=open(os.path.join(FIG_DIR, 'result.txt'), 'w')
            f.write('INDEX          GT               PRED\n')

            while global_step < steps:
                b_imgs, b_views, b_imgs_lab, b_views_lab = next(BatchProvider)
                feed_dict = {img: b_imgs,
                            img_lab: b_imgs_lab,
                            view: b_views,
                            view_lab: b_views_lab}

                b_f1_lab = b_views_lab if self.testmode==2 else b_imgs_lab
                b_f2_lab = b_imgs_lab if (self.testmode==1 or self.testmode==3) else b_views_lab

                # Matching
                if self.testmode == 0: # Trans-View matching
                    b_transf, b_viewf, indices = sess.run([test_f1, test_f2, matched_indices], feed_dict = feed_dict)
                    print('gt: ', b_f1_lab, ' pred: ', indices)
                    ac, num, ac_ind = self.accuracy(b_f1_lab, b_f2_lab, indices)
                    w_true += [x for x in b_f1_lab]
                    w_pred += [x for x in indices]
                else: # Classification
                    b_f, ac, pred = sess.run([test_f, acc, predict], feed_dict=feed_dict)
                    num = b_f.shape[0]
                    print(pred)

                # Calculate accuracy
                total_acc += ac
                total_num += num

                if self.testmode == 0:
                    # Visualize matching results
                    for i in range(b_imgs.shape[0]):
                        plt.clf()
                        edge_color = 'green' if ac_ind[i]==1 else 'red'
                        fig, ax = plt.subplots(1, 2, sharey=True, linewidth=10, edgecolor=edge_color)
                        ax[0].imshow(b_imgs[i]+0.5)
                        img_cls = self.DataProvider.class_numtoidx[b_imgs_lab[i]]
                        ax[0].set_title('Image: %s' % img_cls)
                        ax[0].set_xticks([])
                        ax[0].set_yticks([])
                        ax[1].imshow(b_views[indices[i]][0]+0.5)
                        view_cls = self.DataProvider.class_numtoidx[b_views_lab[indices[i]]]
                        ax[1].set_title('Cad: %s' % view_cls)
                        ax[1].set_xticks([])
                        ax[1].set_yticks([])
                        plt.tight_layout()
                        plt.savefig(FIG_DIR+'/{:03d}_'.format(global_step)+'{:02d}.png'.format(i), dpi=300, edgecolor=fig.get_edgecolor())
                        plt.close()
                        f.write('{:03d}_'.format(global_step)+'{:02d}'.format(i)+'     '+img_cls+'     '+view_cls+'\n')

                global_step += 1

            total_acc /= total_num

            # plot confusion matrix
            label_list = [self.DataProvider.class_numtoidx[x] for x in range(self.DataProvider.class_len)]
            label_list_x = label_list.copy()
            first=0
            prev=''
            for i in range(len(label_list_x)):
                if i==0:
                    first = 1
                    prev=label_list_x[i].split('_')[0]
                    lab = label_list_x[i].split('_')[1]+'\n'+label_list_x[i].split('_')[0]
                else:
                    if prev == label_list_x[i].split('_')[0]:
                        first = 0
                        lab = label_list_x[i].split('_')[1]
                    else:
                        first = 1
                        prev = label_list_x[i].split('_')[0]
                        lab = label_list_x[i].split('_')[1]+'\n'+label_list_x[i].split('_')[0]
                label_list_x[i] = lab
            conf_map = confusion_matrix(w_true, w_pred)
            fig = plt.figure()
            plt.imshow(conf_map, interpolation='nearest', cmap=plt.get_cmap('Blues'))
            plt.title('Confusion matrix')
            plt.colorbar()
            plt.xticks([x for x in range(self.DataProvider.class_len)], label_list_x, fontsize=8)
            plt.yticks([x for x in range(self.DataProvider.class_len)], label_list, fontsize=10)
            plt.ylabel('True Label')
            plt.xlabel('Predicted label\n Accuracy {:.2f}%'.format(total_acc*100))
            plt.tight_layout()
            plt.savefig(FIG_DIR+'/Confusion_matrix.png', dpi=600)
            plt.close()

            print('total accuracy %.2f' % (total_acc*100) + '%, ' + '%d epoch' % (self.max_epoch))
            f.write('total accuracy %.2f' % (total_acc*100)+'%')
            f.close()

