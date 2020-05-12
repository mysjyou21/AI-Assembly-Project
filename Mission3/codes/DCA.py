import tensorflow as tf
import sys
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
import shutil
from sklearn.manifold import TSNE

from model import CNN, METRIC, VIEWPOOL, TRANSFORM, VIEWS, DISCRIMINATOR
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

    def build_model(self, img, views):
        """ img: BxHxWx3
            views: BxVxHxWx3 """
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        # img feature
        img_feature_logit = CNN(img, name='image', is_train=self.is_train, regularizer=self.regularizer)
        img_feature_logit = METRIC(img_feature_logit, name='image', is_train=self.is_train, regularizer=self.regularizer)
        with tf.variable_scope('image_METRIC'):
            img_feature = tf.nn.tanh(img_feature_logit)

        # img2view transformation(generator)
        trans_feature_logit = TRANSFORM(img_feature_logit, name='trans', regularizer=self.regularizer)
        with tf.variable_scope('trans'):
            trans_feature = tf.nn.tanh(trans_feature_logit)

        # view features
        view_feature = VIEWS(views, name='view', _type='mean', is_train=self.is_train, regularizer=self.regularizer)
        view_feature_logit = METRIC(view_feature, name='view', is_train=self.is_train, regularizer=self.regularizer)
        with tf.variable_scope('view_METRIC'):
            view_feature = tf.nn.tanh(view_feature_logit)

        # discriminator  
        real_logit = DISCRIMINATOR(view_feature_logit, reuse=False, regularizer=self.regularizer)
        fake_logit = DISCRIMINATOR(trans_feature_logit, reuse=True, regularizer=self.regularizer)

        return img_feature, trans_feature, view_feature, real_logit, fake_logit, img_feature_logit, view_feature_logit, trans_feature_logit

    def define_optimizer(self, step, img_loss, view_loss, trans_loss, dis_loss):
        t_vars = tf.trainable_variables()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.lr = self.initial_lr
#        self.lr = tf.train.exponential_decay(self.initial_lr, step, self.decay_steps, self.decay_rate, staircase=True)

        self.img_vars = [v for v in t_vars if 'image' in v.name]
        img_update_ops = [v for v in update_ops if 'image' in v.name]
        with tf.control_dependencies(img_update_ops):
            self.img_optimizer = tf.train.AdamOptimizer(self.lr).minimize(img_loss)

        self.view_vars = [v for v in t_vars if 'view' in v.name]
        view_update_ops = [v for v in update_ops if 'view' in v.name]
        with tf.control_dependencies(view_update_ops):
            self.view_optimizer = tf.train.AdamOptimizer(self.lr).minimize(view_loss)

        self.trans_vars = [v for v in t_vars if 'trans' in v.name]
        self.trans_optimizer = tf.train.AdamOptimizer(self.lr).minimize(trans_loss)

        self.dis_vars = [v for v in t_vars if 'discriminator' in v.name]
        self.dis_optimizer = tf.train.AdamOptimizer(self.lr).minimize(dis_loss)

    def train(self):
        """ train the entire models """
        # Set data provider
        BatchProvider = self.DataProvider.make_batch(self.max_epoch, self.C, self.K, (self.H, self.W))

        # define graph
        img = tf.placeholder(tf.float32, [None, self.H, self.W, 3])
        img_lab = tf.placeholder(tf.int32, [None])
        view = tf.placeholder(tf.float32, [None, self.V, self.H, self.W, 3])
        view_lab = tf.placeholder(tf.int32, [None])

        imgf, transf, viewf, real_logit, fake_logit, imgf_logit, viewf_logit, transf_logit = self.build_model(img, view)

        # loss
#        img_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(img_lab, depth=10), logits=imgf_logit))
        img_triplet_loss = triplet_loss(imgf_logit, img_lab, self.margin)
        view_triplet_loss = triplet_loss(viewf_logit, view_lab, self.margin)
        trans_triplet_loss = triplet_loss(transf_logit, img_lab, self.margin)

        gen_loss = generator_loss(fake_logit)
        dis_loss = discriminator_loss(real_logit, fake_logit)

        cmd_loss, trans_f, view_f = cross_mean_discrepancy_loss(transf, viewf, img_lab, view_lab, self.C, self.K)

        trans_loss = 0.5*trans_triplet_loss + gen_loss*int(self.mode==0) + cmd_loss

        # regluarizer
        reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        img_reg_vars = [v for v in reg_vars if 'image' in v.name]
        view_reg_vars = [v for v in reg_vars if 'view' in v.name]
        trans_reg_vars = [v for v in reg_vars if 'trans' in v.name]
        dis_reg_vars = [v for v in reg_vars if 'discriminator' in v.name]

        img_loss = img_triplet_loss #+ tf.contrib.layers.apply_regularization(self.regularizer, img_reg_vars)
        view_loss = view_triplet_loss #+ tf.contrib.layers.apply_regularization(self.regularizer, view_reg_vars)
#        trans_loss += tf.contrib.layers.apply_regularization(self.regularizer, trans_reg_vars)
#        dis_loss += tf.contrib.layers.apply_regularization(self.regularizer, dis_reg_vars)

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

        # tensorboard
        l1 = tf.summary.scalar('img_loss', img_loss)
        l2 = tf.summary.scalar('view_loss', view_loss)
        l3 = tf.summary.scalar('trans_loss', trans_loss)
        l4 = tf.summary.scalar('discriminator_loss', dis_loss)
        l5 = tf.summary.scalar('trans_triplet_loss', trans_triplet_loss)
        l6 = tf.summary.scalar('trans_generator_loss', gen_loss)
        l7 = tf.summary.scalar('trans_cmd_loss', cmd_loss)

        v1 = tf.summary.scalar('learning_rate', self.lr)

        i1 = tf.summary.image('image', img)
        view_summary = view[:,0,:,:,:]
        i2 = tf.summary.image('view', view_summary)

        var_list = [l1,l2,l3,l4,l5,l6,l7,i1,i2]
        summary_op = tf.summary.merge(var_list)

        saver_I = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='image'), max_to_keep=2)
        saver_V = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='view'), max_to_keep=2)
        saver_T = tf.train.Saver(var_list=self.trans_vars, max_to_keep=2)
        saver_D = tf.train.Saver(var_list=self.dis_vars+tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global_step'), max_to_keep=2)
        
        model_name = 'entire_model.ckpt' if self.mode==0 else 'model.ckpt'

        dir_I = self.checkpoint_dir+'/Image'
        dir_V = self.checkpoint_dir+'/View'
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
            if tf.train.latest_checkpoint(dir_V) is not None:
                saver_V.restore(sess, tf.train.latest_checkpoint(dir_V))
                print('Load saved model from %s' % tf.train.latest_checkpoint(dir_V))
            if tf.train.latest_checkpoint(dir_T) is not None:
                saver_T.restore(sess, tf.train.latest_checkpoint(dir_T))
                print('Load saved model from %s' % tf.train.latest_checkpoint(dir_T))
            if tf.train.latest_checkpoint(dir_D) is not None:
                saver_D.restore(sess, tf.train.latest_checkpoint(dir_D))
                print('Load saved model from %s' % tf.train.latest_checkpoint(dir_D))
                print(sess.run(step))

            writer = tf.summary.FileWriter(self.checkpoint_dir, graph=tf.get_default_graph())

            global_step = sess.run(step)
            print('step: %d' % global_step)
            steps_for_epoch = int((self.DataProvider.whole_batch_size-self.C*self.K)/self.C/self.K)
            epoch = global_step//steps_for_epoch

            try:
                while epoch <= self.max_epoch:
                    b_imgs, b_views, b_imgs_lab, b_views_lab = next(BatchProvider)
                    feed_dict = {img: b_imgs,
                                 img_lab: b_imgs_lab,
                                 view: b_views,
                                 view_lab: b_views_lab}

                    # just for print function
                    t_img_loss, t_view_loss, t_trans_loss, t_dis_loss = 0,0,0,0


                    if self.mode == 0 or self.mode==1:
                        t_img_loss, _ = sess.run([img_loss, self.img_optimizer], feed_dict={img: b_imgs, img_lab: b_imgs_lab})
#                        t_map, t_man, t_hpd, t_hnd, t_l, pd = sess.run([mask_ap, mask_an, hardest_p, hardest_n, triplet_loss_, pairwise_distances], feed_dict=feed_dict)
                    if self.mode == 0 or self.mode==2:
                        t_view_loss,_ = sess.run([view_loss, self.view_optimizer], feed_dict={view: b_views, view_lab: b_views_lab})

                    if self.mode == 0:
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

                    if global_step%10 == 0:
                        indices = sess.run(matched_indices[0], feed_dict=feed_dict)
                        sample_testf1, sample_testf2 = sess.run([test_f1, test_f2], feed_dict=feed_dict)
                        print('epoch %d step %d::img loss %.5f, view loss %.5f, trans loss %.5f dis loss %.5f' % (epoch, global_step, t_img_loss, t_view_loss, t_trans_loss, t_dis_loss))
                        print('pred: ', indices)
                        print('f1: ', sample_testf1[0:2*self.K])
                        print('f2: ', sample_testf2[0:2*self.K])

                    if epoch%5 == 0 and global_step%steps_for_epoch == 0:
                        if self.mode == 0 or self.mode == 1:
                            savepath_I = saver_I.save(sess, dir_I+'/'+model_name, global_step=step)
                            print('save image model in %s' % savepath_I)
                        if self.mode == 0 or self.mode == 2:
                            savepath_V = saver_V.save(sess, dir_V+'/'+model_name, global_step=step)
                            print('save view model in %s' % savepath_V)
                        if self.mode == 0 or self.mode == 3:
                            savepath_T = saver_T.save(sess, dir_T+'/'+model_name, global_step=step)
                            print('save trans model in %s' % savepath_T)
                        if self.mode == 0:
                            savepath_D = saver_D.save(sess, dir_D+'/'+model_name, global_step=step)
                            print('save dis model in %s' % savepath_D)


            except KeyboardInterrupt:
                print('KeyboardInterrupt')
            finally:
                if self.mode == 0 or self.mode == 1:
                    savepath_I = saver_I.save(sess, dir_I+'/'+model_name, global_step=step)
                    print('save image model in %s' % savepath_I)
                if self.mode == 0 or self.mode == 2:
                    savepath_V = saver_V.save(sess, dir_V+'/'+model_name, global_step=step)
                    print('save view model in %s' % savepath_V)
                if self.mode == 0 or self.mode == 3:
                    savepath_T = saver_T.save(sess, dir_T+'/'+model_name, global_step=step)
                    print('save trans model in %s' % savepath_T)
                if self.mode == 0:
                    savepath_D = saver_D.save(sess, dir_D+'/'+model_name, global_step=step)
                    print('save dis model in %s' % savepath_D)

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

        return np.sum(accurate), accurate.shape[0]

    def test(self):
        self.max_epoch = 1
        BatchProvider = self.DataProvider.make_batch(self.max_epoch, self.C, self.K, (self.H, self.W), is_shuffle=False)

        # define graph
        img = tf.placeholder(tf.float32, [None, self.H, self.W, 3])
        img_lab = tf.placeholder(tf.int32, [None])
        view = tf.placeholder(tf.float32, [None, self.V, self.H, self.W, 3])
        view_lab = tf.placeholder(tf.int32, [None])

        imgf, transf, viewf, _, _, imgf_logit, viewf_logit, transf_logit = self.build_model(img, view)

#        predict = tf.nn.softmax(imgf_logit, axis=1)
#        predict = tf.math.argmax(predict, axis=1)

#       acc = tf.reduce_sum(tf.cast(tf.math.equal(predict, tf.cast(img_lab, tf.int64)), tf.int64))

        # [B], matched_indices[i] = the most appropriate (to i-th transf feature) feature among viewf features
        matched_indices, _,_ = self.matching(transf, viewf, same=False)

        # for restoring models
        t_vars = tf.trainable_variables()
        self.trans_vars = [v for v in t_vars if 'trans' in v.name]
        saver_I = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='image'), max_to_keep=2)
        saver_V = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='view'), max_to_keep=2)
        saver_T = tf.train.Saver(var_list=self.trans_vars, max_to_keep=2)
        
        dir_I = self.checkpoint_dir+'/Image'
        dir_V = self.checkpoint_dir+'/View'
        dir_T = self.checkpoint_dir+'/Trans'

        # Session configuration
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            # restore models
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            saver_I.restore(sess, tf.train.latest_checkpoint(dir_I))
            print('Load saved model from %s' % tf.train.latest_checkpoint(dir_I))
            saver_V.restore(sess, tf.train.latest_checkpoint(dir_V))
            print('Load saved model from %s' % tf.train.latest_checkpoint(dir_V))
            saver_T.restore(sess, tf.train.latest_checkpoint(dir_T))
            print('Load saved model from %s' % tf.train.latest_checkpoint(dir_T))

            epoch = 0
            global_step = 0
            total_acc = 0
            total_num = 0
            steps_for_epoch = int((self.DataProvider.whole_batch_size-self.C*self.K)/self.C/self.K)

            # for visualization
            FIG_DIR = self.checkpoint_dir+'/Test'
            if os.path.exists(FIG_DIR):
                shutil.rmtree(FIG_DIR)
            if not os.path.exists(FIG_DIR):
                os.makedirs(FIG_DIR)

            while epoch <= self.max_epoch:
                b_imgs, b_views, b_imgs_lab, b_views_lab = next(BatchProvider)
                feed_dict = {img: b_imgs,
                            img_lab: b_imgs_lab,
                             view: b_views}

                # Matching
#                b_imgf, ac = sess.run([imgf_logit, acc], feed_dict=feed_dict)
#                import ipdb
#                ipdb.set_trace()
                b_transf, b_viewf, indices = sess.run([transf, viewf, matched_indices], feed_dict = feed_dict)
                if global_step%5==0:
                    print(indices)

                # Calculate accuracy
                acc, num = self.accuracy(b_imgs_lab, b_views_lab, indices)
                total_acc += acc
                total_num += num #b_imgs_lab.shape[0]

                # Visualize matching results
                for i in range(b_imgs.shape[0]):
                    plt.clf()
                    fig, ax = plt.subplots(1, 2, sharey=True)
                    ax[0].imshow(b_imgs[i])
                    img_cls = self.DataProvider.class_numtoidx[b_imgs_lab[i]]
                    ax[0].set_title('Image: %s' % img_cls)
                    ax[1].imshow(b_views[indices[i]][0])
                    view_cls = self.DataProvider.class_numtoidx[b_views_lab[indices[i]]]
                    ax[1].set_title('View: %s' % view_cls)
                    plt.savefig(FIG_DIR+'/{:03d}_'.format(global_step)+'{:02d}.png'.format(i))
                    plt.close()

                # Visualize features
                shapes = ["o", "v", "<", "s", "p", "P", "*", "h", "D", "d", "^", ">"]
                colors = ["red", "green", "blue", "yellow", "pink", "brown"]
                b_ind = {}
                for i in range(self.C):
                    ind = b_imgs_lab[i*self.K]
                    b_ind[ind] = i
                tsne = TSNE(perplexity=30, n_components=2, init="pca")
                b_f = np.concatenate([b_transf, b_viewf], 0)
                embedded = tsne.fit_transform(b_f)
                plt.figure()
                for i in range(b_imgs.shape[0]):
                    plt.plot(embedded[i][1], embedded[i][0], color=colors[b_ind[b_imgs_lab[i]]], marker=shapes[0], markersize=13)
                    plt.plot(embedded[i+b_imgs.shape[0]][1], embedded[i+b_imgs.shape[0]][0], color=colors[b_ind[b_views_lab[i]]], marker=shapes[1], markersize=13)
                plt.savefig(FIG_DIR+'/{:03d}_feautre.png'.format(global_step))
                plt.close()

                global_step += 1
                if global_step%steps_for_epoch == 0:
                    epoch += 1

            total_acc /= total_num
            print('total accuracy %.4f, %d times' % (total_acc, self.max_epoch))
