import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import sys
import os
import cv2
import time
import glob
from pose_model import MODEL
from pose_dataset import Dataset
from pose_ops import *
from pose_gt import *
from tensorflow.python import pywrap_tensorflow


class POSE_NET(object):

    def __init__(self, args):        
        self.checkpoint_dir = args.opt.pose_model_path

        with args.graph_pose.as_default():
            sess = args.sess_pose
            self.is_training = tf.placeholder(tf.bool)
            self.X = tf.placeholder(tf.float32, [2, 224, 224, 3])
            self.Y = tf.placeholder(tf.float32, [None])        
            self.logit = self.model(self.X, self.is_training)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='img_n_views')
            saver = tf.train.Saver(var_list=var_list)
            if tf.train.latest_checkpoint(self.checkpoint_dir) is not None:             
                print('POSE MODEL : Loading weights from %s' % tf.train.latest_checkpoint(self.checkpoint_dir))
                saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_dir))
            else:
                print('No pose checkpoint found at {}'.format(self.checkpoint_dir))

    def model(self, x, is_training):
        self.regularizer = layers.l2_regularizer(scale=0.1)
        logit = MODEL(x, is_train=is_training, regularizer=self.regularizer)        
        return logit

    def test(self, args, step_num):
        SAVE_PART_ID_POSE = args.opt.save_part_id_pose
        if not os.path.exists(args.opt.part_id_pose_path):
            os.makedirs(args.opt.part_id_pose_path)
        input_images = args.parts[step_num]
        retrieved_classes = args.cad_models[step_num]
        matched_poses = []
        with args.graph_pose.as_default():
            sess = args.sess_pose
            batch_provider = Dataset(args, input_images, retrieved_classes)
            for img_idx, input_image in enumerate(input_images):
                pred_list = []
                img_n_views = batch_provider.return_testbatch(img_idx)                
                for view_idx in range(48):
                    b_img_n_views = img_n_views[2*view_idx:2*view_idx + 2, :]
                    pred = sess.run([self.logit], feed_dict={self.X:b_img_n_views, self.is_training:False})
                    # print('    view {} pred {}'.format(view_idx, pred[0][0]))
                    pred_list.append(pred[0][0][0])
                pred_argmax = np.argmax(pred_list)
                # print('class index : {},  class name : {}, pred_argmax : {}'.format(img_idx, self.retrieved_classes[img_idx], pred_argmax))
                matched_poses.append(pred_argmax)
            if SAVE_PART_ID_POSE:
                for i in range(len(input_images)):
                    plt.clf()
                    fig, ax = plt.subplots(1, 2, sharey=True)
                    ax[0].imshow(batch_provider.resize_and_pad(input_images[i]))
                    ax[0].set_title('detection result')
                    pose_imgs_filenames = sorted(glob.glob(args.opt.pose_views_path + '/' + retrieved_classes[i] + '/*.png'))
                    LABEL_TO_POSE = {v: k for k, v in POSE_TO_LABEL.items()}
                    pose_label = LABEL_TO_POSE[matched_poses[i]]
                    pose_img_filename = ''
                    for v in pose_imgs_filenames:
                        pose_img_filename = v if v.endswith(pose_label + '.png') else pose_img_filename
                    pose_img = cv2.imread(pose_img_filename)
                    ax[1].imshow(batch_provider.resize_and_pad(pose_img))
                    ax[1].set_title('pred cad : {}\npred pose : {}'.format(retrieved_classes[i], pose_label))
                    plt.savefig(args.opt.part_id_pose_path + '/STEP{}_part{}'.format(step_num, i))
                    plt.close()
        return matched_poses

                

