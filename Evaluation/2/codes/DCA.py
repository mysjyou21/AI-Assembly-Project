import tensorflow as tf
import sys
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import shutil
import cv2
from sklearn.metrics import confusion_matrix

from model import CNN, METRIC, VIEWPOOL, TRANSFORM, VIEWS, DISCRIMINATOR, CLASSIFIER
from model import map_ftn
from losses import *
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
        self.DataProvider = Dataset(self.DATA_DIR, view_num=FLAGS.view_num, mode=FLAGS.mode)
        self.len = self.DataProvider.class_len
        self.C = FLAGS.C
        self.K = FLAGS.K
        self.ft_channel = FLAGS.feature_channel
        self.is_train = True if 'train' in FLAGS.mode else False
        # -1: test, 0: train(whole), 1: pre-train image and view CNN/metric models, 2: pre-train trans model
        self.mode = -1 if FLAGS.mode=='test' else 1 if 'feature_img' in FLAGS.mode else 2 if 'feature_view' in FLAGS.mode else 3 if 'trans' in FLAGS.mode else 0
        self.max_epoch = FLAGS.max_epoch

        self.V = FLAGS.view_num
        self.margin = FLAGS.margin

        self.initial_lr = FLAGS.lr
        self.decay_steps = FLAGS.decay_steps
        self.decay_rate = FLAGS.decay_rate
        self.restore = FLAGS.restore
        self.checkpoint_dir = FLAGS.checkpoint_dir
        self.pretrained_checkpoint_path = FLAGS.pretrained_checkpoint_path
        self.init = FLAGS.init
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
        view_feature = VIEWS(views, ft_channel, trans_feature=trans_feature if (self.mode==0 or self.mode==-1) else None, batch_size=self.C, name='view', _type='attention' if (self.mode==0 or (self.mode==-1 and (self.testmode==0 or self.testmode==4))) else 'mean', is_train=self.is_train, regularizer=None, view_num=self.V) #batch_size=self.C
#        view_feature = VIEWS(views, ft_channel, trans_feature=None, batch_size=self.C, name='view', _type='mean', is_train=self.is_train, regularizer=None, view_num=self.V) #batch_size=self.C
        view_feature_logit = METRIC(view_feature, ft_channel, name='view', is_train=self.is_train, regularizer=self.regularizer)
        with tf.variable_scope('view_METRIC'):
            view_feature = tf.nn.tanh(view_feature_logit)
        view_feature_class = CLASSIFIER(view_feature, class_num, name='view', is_train=self.is_train, regularizer=None)

        # discriminator
        real_logit = DISCRIMINATOR(view_feature, class_num, reuse=False, regularizer=self.regularizer)
        fake_logit = DISCRIMINATOR(trans_feature, class_num, reuse=True, regularizer=self.regularizer)

        return img_feature, trans_feature, view_feature, real_logit, fake_logit, img_feature_logit, view_feature_logit, trans_feature_logit, img_feature_class, view_feature_class, trans_feature_class

    def define_optimizer(self, step, img_loss, view_loss, trans_loss, dis_loss):
        t_vars = tf.trainable_variables()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

#        self.lr = self.initial_lr
        self.lr = tf.train.exponential_decay(self.initial_lr, step, self.decay_steps, self.decay_rate, staircase=True)

        self.img_vars = [v for v in t_vars if 'image' in v.name]
        img_update_ops = [v for v in update_ops if 'image' in v.name]
        with tf.control_dependencies(img_update_ops):
            self.img_optimizer = tf.train.AdamOptimizer(self.lr).minimize(img_loss, var_list=self.img_vars)

        self.view_vars = [v for v in t_vars if 'view' in v.name]
        self.viewpool_vars = [v for v in t_vars if 'VIEWPOOL' in v.name]
        view_update_ops = [v for v in update_ops if 'view' in v.name]
        viewpool_update_ops = [v for v in update_ops if 'VIEWPOOL' in v.name]
        with tf.control_dependencies(view_update_ops+viewpool_update_ops):
            self.view_optimizer = tf.train.AdamOptimizer(self.lr).minimize(view_loss, var_list=self.view_vars+self.viewpool_vars)

        self.trans_vars = [v for v in t_vars if 'trans' in v.name]
        self.trans_optimizer = tf.train.AdamOptimizer(self.lr).minimize(trans_loss, var_list=self.trans_vars)

        self.dis_vars = [v for v in t_vars if 'discriminator' in v.name]
        self.dis_optimizer = tf.train.AdamOptimizer(self.lr).minimize(dis_loss, var_list=self.dis_vars)

    def train(self):
        """ train the entire models """
        # Set data provider
        BatchProvider = self.DataProvider.make_batch(self.max_epoch, self.C, self.K, (self.H, self.W))

        # define graph
        img = tf.placeholder(tf.float32, [None, self.H, self.W, 3])
        img_lab = tf.placeholder(tf.int32, [None])
        view = tf.placeholder(tf.float32, [None, self.V, self.H, self.W, 3])
        view_lab = tf.placeholder(tf.int32, [None])

        imgf, transf, viewf, real_logit, fake_logit, imgf_logit, viewf_logit, transf_logit, imgf_class, viewf_class, transf_class = self.build_model(img, view)


        # loss
        img_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(img_lab, depth=self.DataProvider.class_len), logits=imgf_class))
        view_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(view_lab, depth=self.DataProvider.class_len), logits=viewf_class))
        trans_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(img_lab, depth=self.DataProvider.class_len), logits=transf_class))

        gen_loss = generator_loss(fake_logit)
        dis_loss = discriminator_loss(real_logit, fake_logit)

        cmd_loss, trans_f, view_f = cross_mean_discrepancy_loss(transf, viewf, img_lab, view_lab, self.C, self.K, 1)

        trans_loss = trans_class_loss + gen_loss + cmd_loss

        view_loss = view_class_loss + 0.1*cmd_loss*int(self.mode==0)

#        # regluarizer
#        reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#        img_reg_vars = [v for v in reg_vars if 'image' in v.name]
#        view_reg_vars = [v for v in reg_vars if 'view' in v.name]
#        trans_reg_vars = [v for v in reg_vars if 'trans' in v.name]
#        dis_reg_vars = [v for v in reg_vars if 'discriminator' in v.name]

        # optimizer
        step = tf.Variable(0, trainable=False, name='global_step')
        increment_step = tf.assign_add(step, 1)
        reset_step = step.assign(0)

        self.define_optimizer(step, img_loss, view_loss, trans_loss, dis_loss)

        # test
        test_f1 = imgf if self.mode==1 else viewf if self.mode==2 else transf
        test_f2 = imgf if self.mode==1 else viewf if self.mode==2 else transf if self.mode==3 else viewf
        same = False if self.mode==0 else True
        matched_indices = self.matching(test_f1, test_f2, same=same)
        test_f_class = imgf_class if self.mode==1 else viewf_class if self.mode==2 else transf_class
        accurate = tf.argmax(tf.math.softmax(test_f_class, axis=1), axis=1)

        # tensorboard
        l1 = tf.summary.scalar('img_loss', img_loss)
        l2 = tf.summary.scalar('view_loss', view_loss)
        l3 = tf.summary.scalar('view_class_loss', view_class_loss)
        l4 = tf.summary.scalar('trans_loss', trans_loss)
        l5 = tf.summary.scalar('discriminator_loss', dis_loss)
        l6 = tf.summary.scalar('trans_class_loss', trans_class_loss)
        l7 = tf.summary.scalar('trans_generator_loss', gen_loss)
        l8 = tf.summary.scalar('trans_cmd_loss', cmd_loss)

        v1 = tf.summary.scalar('learning_rate', self.lr)

        i1 = tf.summary.image('image', img)
        view_summary = view[:,0,:,:,:]
        i2 = tf.summary.image('view', view_summary)

        var_list = [l1,l2,l3,l4,l5,l6,l7,l8,i1,i2]
        summary_op = tf.summary.merge(var_list)

        saver_I_CNN_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='image_CNN')
        saver_I_metric_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='image_METRIC')
        saver_I_class_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='image_CLASSIFIER')
        saver_I = tf.train.Saver(var_list=saver_I_CNN_vars + saver_I_metric_vars + saver_I_class_vars, max_to_keep=40)
        saver_V_CNN_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='view_CNN')
        saver_V_metric_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='view_METRIC')
        saver_V_class_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='view_CLASSIFIER')
        saver_V = tf.train.Saver(var_list=saver_V_CNN_vars+saver_V_metric_vars+saver_V_class_vars, max_to_keep=40)
        if self.mode==0:
            saver_VP = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='VIEWPOOL'), max_to_keep=40)
        saver_T = tf.train.Saver(var_list=self.trans_vars, max_to_keep=40)
        saver_D = tf.train.Saver(var_list=self.dis_vars+tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global_step'), max_to_keep=40)

        model_name = 'entire_model.ckpt' if self.mode==0 else 'model.ckpt'

        dir_I = self.checkpoint_dir+'/Image'
        dir_V = self.checkpoint_dir+'/View'
        dir_VP = self.checkpoint_dir+'/ViewPool'
        dir_T = self.checkpoint_dir+'/Trans'
        dir_D = self.checkpoint_dir+'/Dis'

        # Session configuration
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
        config.gpu_options.allow_growth = True

        # Open Session
        with tf.Session(config=config) as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            if tf.train.latest_checkpoint(dir_I) is not None:
                saver_I.restore(sess, tf.train.latest_checkpoint(dir_I))
                print('Load saved model from %s' % tf.train.latest_checkpoint(dir_I))
            elif self.mode!=2 and self.restore:
                map_ftn(self.pretrained_checkpoint_path, saver_I_CNN_vars)
                print('Load Imagenet pretrained network from %s' % self.pretrained_checkpoint_path)
            if tf.train.latest_checkpoint(dir_V) is not None:
                saver_V.restore(sess, tf.train.latest_checkpoint(dir_V))
                print('Load saved model from %s' % tf.train.latest_checkpoint(dir_V))
            elif self.mode!=1 and self.restore:
                map_ftn(self.pretrained_checkpoint_path, saver_V_CNN_vars)
                print('Load Imagenet pretrained network from %s' % self.pretrained_checkpoint_path)
            if tf.train.latest_checkpoint(dir_VP) is not None:
                saver_VP.restore(sess, tf.train.latest_checkpoint(dir_VP))
                print('Load saved model from %s' % tf.train.latest_checkpoint(dir_VP))
            if tf.train.latest_checkpoint(dir_T) is not None:
                saver_T.restore(sess, tf.train.latest_checkpoint(dir_T))
                print('Load saved model from %s' % tf.train.latest_checkpoint(dir_T))
            if tf.train.latest_checkpoint(dir_D) is not None:
                saver_D.restore(sess, tf.train.latest_checkpoint(dir_D))
                print('Load saved model from %s' % tf.train.latest_checkpoint(dir_D))
                if self.init:
                    sess.run(reset_step)
                print(sess.run(step))

            writer = tf.summary.FileWriter(self.checkpoint_dir+'/'+str(self.mode), graph=tf.get_default_graph())

            global_step = sess.run(step)
            print('step: %d' % global_step)
            steps_for_epoch = int((self.DataProvider.whole_batch_size)/self.C/self.K)
            epoch = global_step//steps_for_epoch

            # just for print function
            t_img_loss, t_view_loss, t_trans_loss, t_dis_loss = -1,-1,-1,-1

            try:
                while epoch <= self.max_epoch:
                    b_imgs, b_views, b_imgs_lab, b_views_lab = next(BatchProvider)
                    test_lab = b_views_lab if self.mode==2 else b_imgs_lab
                    feed_dict = {img: b_imgs,
                                 img_lab: b_imgs_lab,
                                 view: b_views,
                                 view_lab: b_views_lab}

                    if self.mode==1 or self.mode==0:
                        t_img_loss, _ = sess.run([img_loss, self.img_optimizer], feed_dict={img: b_imgs, img_lab: b_imgs_lab})
#                        t_map, t_man, t_hpd, t_hnd, t_l, pd = sess.run([mask_ap, mask_an, hardest_p, hardest_n, triplet_loss_, pairwise_distances], feed_dict=feed_dict)
                    if self.mode == 0 or self.mode==2:
                        t_view_loss,_ = sess.run([view_loss, self.view_optimizer], feed_dict=feed_dict)

                    if self.mode == 0 or self.mode==3:
                        t_dis_loss, _ = sess.run([dis_loss, self.dis_optimizer], feed_dict=feed_dict)

                    if self.mode == 0 or self.mode==3:
                        t_trans_loss, _ = sess.run([trans_loss, self.trans_optimizer], feed_dict=feed_dict)
#                        t_trans_f, t_view_f = sess.run([trans_f, view_f], feed_dict=feed_dict)
                    sess.run(increment_step)
                    global_step = sess.run(step)

                    summary = sess.run(summary_op, feed_dict=feed_dict)
                    writer.add_summary(summary, global_step)

                    if global_step%steps_for_epoch == 0:
                        epoch += 1

                    if global_step%5 == 0:
                        indices = sess.run(matched_indices[0], feed_dict=feed_dict)
                        sample_testf1, sample_testf2 = sess.run([test_f1, test_f2], feed_dict=feed_dict)
                        print('epoch %d step %d::img loss %.5f, view loss %.5f, trans loss %.5f dis loss %.5f' % (epoch, global_step, t_img_loss, t_view_loss, t_trans_loss, t_dis_loss))
                        print('gt: ', test_lab)
#                        print('f1: ', sample_testf1[[0,self.K,2*self.K][0:8]])
#                        print('f2: ', sample_testf2[[0,self.K,2*self.K][0:8]])
                        ac = sess.run(accurate, feed_dict=feed_dict)
                        print('pred: ', ac)

                    if epoch%1 == 0 and global_step%steps_for_epoch == 0:
                        if self.mode == 1 or self.mode==0:
                            savepath_I = saver_I.save(sess, dir_I+'/'+model_name, global_step=step)
                            print('save image model in %s' % savepath_I)
                        if self.mode == 0 or self.mode == 2:
                            savepath_V = saver_V.save(sess, dir_V+'/'+model_name, global_step=step)
                            print('save view model in %s' % savepath_V)
                        if self.mode == 0 or self.mode == 3:
                            savepath_T = saver_T.save(sess, dir_T+'/'+model_name, global_step=step)
                            print('save trans model in %s' % savepath_T)
                        if self.mode == 0 or self.mode==3:
                            savepath_D = saver_D.save(sess, dir_D+'/'+model_name, global_step=step)
                            print('save dis model in %s' % savepath_D)
                        if self.mode == 0:
                            savepath_VP = saver_VP.save(sess, dir_VP+'/'+model_name, global_step=step)
                            print('save vp model in %s' % savepath_VP)


            except KeyboardInterrupt:
                print('KeyboardInterrupt')
#                if self.mode == 1 or self.mode==0:
#                    savepath_I = saver_I.save(sess, dir_I+'/'+model_name, global_step=step)
#                    print('save image model in %s' % savepath_I)
#                if self.mode == 0 or self.mode == 2:
#                    savepath_V = saver_V.save(sess, dir_V+'/'+model_name, global_step=step)
#                    print('save view model in %s' % savepath_V)
#                if self.mode == 0 or self.mode == 3:
#                    savepath_T = saver_T.save(sess, dir_T+'/'+model_name, global_step=step)
#                    print('save trans model in %s' % savepath_T)
#                if self.mode == 0 or self.mode==3:
#                    savepath_D = saver_D.save(sess, dir_D+'/'+model_name, global_step=step)
#                    print('save dis model in %s' % savepath_D)
#               if self.mode == 0:
#                    savepath_VP = saver_VP.save(sess, dir_VP+'/'+model_name, global_step=step)
#                    print('save vp model in %s' % savepath_VP)
            finally:
                print('end of training')

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
#        self.max_epoch = 1
        self.C = self.len
        self.K = 1

        tf.reset_default_graph()
        BatchProvider = self.DataProvider.make_batch(self.max_epoch, C=self.C, K=self.K, size=(self.H, self.W), is_shuffle=False)

        # define graph
        img = tf.placeholder(tf.float32, [None, self.H, self.W, 3])
        img_lab = tf.placeholder(tf.int32, [None])
        view = tf.placeholder(tf.float32, [None, self.V, self.H, self.W, 3])
        view_lab = tf.placeholder(tf.int32, [None])

        imgf, transf, viewf, _, _, imgf_logit, viewf_logit, transf_logit, imgf_class, viewf_class, transf_class = self.build_model(img, view)

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
            predict = tf.math.argmax(predict, axis=1)

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
        if self.testmode==0 or self.testmode==4:
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

            while global_step < steps: #epoch <= self.max_epoch:
                b_imgs, b_views, b_imgs_lab, b_views_lab = next(BatchProvider)
                feed_dict = {img: b_imgs,
                            img_lab: b_imgs_lab,
                            view: b_views,
                            view_lab: b_views_lab}

                # Matching
                if self.testmode == 0: # Trans-View matching
                    b_transf, b_viewf, indices = sess.run([test_f1, test_f2, matched_indices], feed_dict = feed_dict)
                    print('gt: ', b_imgs_lab, ' pred: ', indices)
                    ac, num, ac_ind = self.accuracy(b_imgs_lab, b_views_lab, indices)
                    w_true += [x for x in b_imgs_lab]
                    w_pred += [x for x in indices]
                else: # Classification
                    b_f, ac, pred = sess.run([test_f1, acc, predict], feed_dict=feed_dict)
                    num = b_f.shape[0]
                    print(pred)

                if self.testmode != 4:
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

