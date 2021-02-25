import tensorflow as tf
import numpy as np
np.set_printoptions(suppress=True, precision=4)
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import os
import glob
import cv2
from scipy.spatial.transform import Rotation as Rot # quat (x, y, z, w)
from .pose_model import MODEL
from .pose_dataset import Dataset
from .pose_gt import *
from .pose_utils import *


class POSE_NET(object):
    """
    input : 
        manual image
        default pose CAD point cloud
    output :
        quaternion, t
    """
    def __init__(self, args):
        self.args = args
        self.graph_pose = args.graph_pose
        self.sess_pose = args.sess_pose 
        self.ds = Dataset(args)
        self.RTS = self.ds.RTS
        self.checkpoint_path = args.args.pose_model_path
        
        with self.graph_pose.as_default():
            sess = self.sess_pose
            # feed values
            self.is_training = tf.placeholder(tf.bool)
            self.X = tf.placeholder(tf.float32, [None, 224, 224, 3])  # cropped image (resized, padded, normalized)
            self.C = tf.placeholder(tf.float32, [None, 50, 3])  # corner points (scaled down)
            # feedforward
            self.logit = self.model(self.X, self.C, self.is_training)
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='view_n_ptcld')
            saver = tf.train.Saver(var_list=var_list)
            if tf.train.latest_checkpoint(self.checkpoint_path) is not None:             
                print('POSE MODEL : Loading weights from %s' % tf.train.latest_checkpoint(self.checkpoint_path))
                saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_path))
            else:
                print('No pose checkpoint found at {}'.format(self.checkpoint_path))


    def model(self, x, y, is_training):
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        logit = MODEL(x, y, is_train=is_training, regularizer=self.regularizer)
        return logit


    def test(self):
        args = self.args

        # parse retrieval results
        stepNcad_indices = [] # {list} (step number, cad index)
        retrieval_results = [] # {list} cad name
        num_cads_per_step = [] # {list} number of retrieval results for each step
        for key, val in sorted(args.retrieval_results.items()):
            for i in range(len(val)):
                stepNcad_indices.append((key, i))
            retrieval_results.extend(val)
            num_cads_per_step.append(len(val))
        num_steps = len(args.retrieval_results)

        
        # UNIT TEST :assuming GT detection results & GT retrieval results
        if args.mode == 'pose_unit_test':
            # load GT pose labels for calculating scores         
            poses_gt = []
            for key, val in sorted(args.pose_gt.items()):
                poses_gt.extend(val)
            poses_gt = [manage_duplicate_pose(retrieval_result, pose_gt) for retrieval_result, pose_gt in zip(retrieval_results, poses_gt)]
        

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with self.graph_pose.as_default():
            sess = self.sess_pose
            # load data
            imgs, cornerpoints = self.ds.return_testbatch()
            # feed forward
            fetches = self.logit
            feed_dict = {self.X: imgs, self.C: cornerpoints, self.is_training: False}
            logits = sess.run(fetches, feed_dict=feed_dict)
            # closest pose
            RT_preds = [logit_to_RT(logit) for logit in logits]
            poses = [self.closest_gt_RT_idx(RT_pred) for RT_pred in RT_preds]


            # UNIT TEST :assuming GT detection results & GT retrieval results
            if args.mode == 'pose_unit_test':
                # calculate scores
                pose_score_list = []
                for pose, pose_gt in zip(poses, poses_gt):
                    score = 1 if pose == pose_gt else 0
                    pose_score_list.append(score)
                total_score = sum(pose_score_list)
                total_N = len(pose_score_list)
                print('-----------------------------')
                print('Pose unit test accuracy : {:.41}% {}/{}'.format(100 * total_score / total_N, total_score, total_N))
                if args.args.pose_visualization:
                    # load views for visualization
                    with open(args.args.view_imgs_adr, 'rb') as f:
                        VIEW_IMGS = np.load(f)
                    # save images
                    count = 0
                    for input_img, pose, pose_gt, cad_name in zip(imgs, poses, poses_gt, retrieval_results):
                        plt.clf()
                        fig, ax = plt.subplots(1, 2, sharey=True)
                        input_img = (input_img * 255).astype(np.uint8)
                        
                        cad_idx = args.cad_names.index(cad_name)
                        pred_pose_img = VIEW_IMGS[cad_idx, pose, :]
                        
                        ax[0].imshow(input_img)
                        ax[0].set_title('input image')
                        ax[1].imshow(pred_pose_img)
                        ax[1].set_title('retreival results : {}\npred pose : {}\ngt pose : {}'.format(cad_name, LABEL_TO_POSE[pose], LABEL_TO_POSE[pose_gt]))
                        if pose == pose_gt:
                            plt.savefig(args.args.pose_intermediate_results_path + '/correct_' + str(count).zfill(3) + '.png')
                        else:
                            plt.savefig(args.args.pose_intermediate_results_path + '/wrong_' + str(count).zfill(3) + '.png')
                        plt.close()
                        count += 1

            
            if args.mode == 'pose':
                if args.args.pose_visualization:
                    # load views for visualization
                    with open(args.args.view_imgs_adr, 'rb') as f:
                        VIEW_IMGS = np.load(f)

                    # save images
                    for stepNcad_idx, input_img, pose, cad_name in zip(stepNcad_indices, imgs, poses, retrieval_results):
                        plt.clf()
                        fig, ax = plt.subplots(1, 2, sharey=True)
                        plt.xticks([])
                        plt.yticks([])
                        input_img = (input_img * 255).astype(np.uint8)
                        cad_idx = args.cad_names.index(cad_name)
                        pred_pose_img = VIEW_IMGS[cad_idx, pose, :]
                        ax[0].axes.xaxis.set_visible(False)
                        ax[0].axes.yaxis.set_visible(False)
                        ax[1].axes.xaxis.set_visible(False)
                        ax[1].axes.yaxis.set_visible(False)
                        ax[0].imshow(input_img)
                        ax[0].set_title('detection result')
                        ax[1].imshow(pred_pose_img)
                        ax[1].set_title('retrieval result : {}\npred pose : {}'.format(cad_name, pose))
                        plt.savefig(args.args.pose_intermediate_results_path + '/output_'\
                            + args.img_names[stepNcad_idx[0]] + '_' + str(stepNcad_idx[1]).zfill(5) + '.png')
                        plt.close()


            # return pose results 
            pose_results = {}
            idx = 0
            for i in range(num_steps):
                pose_results[i] = []
                for j in range(num_cads_per_step[i]):
                    pop = poses.pop(0)
                    pose_results[i].append(pop)
            assert len(poses) == 0
            return pose_results


    def closest_gt_RT_idx(self, RT_pred):
        args = self.args
        return np.argmin([np.linalg.norm(RT - RT_pred) for RT in self.RTS])
