import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import sys
from pose_gt import *

class Dataset(object):
    def __init__(self, args, input_images, retrieved_class):
        self.args = args
        self.imgs = input_images
        self.retrieved_class = retrieved_class
        self.view_dir = args.opt.pose_views_path

    def resize_and_pad(self, img, a=150):
        # resize to 150, 150
        long_side = np.max(img.shape)
        ratio = a / long_side
        img = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio, interpolation = cv2.INTER_AREA)
        # pad to 224, 224
        pad_left = int(np.ceil((224 - img.shape[1]) / 2))
        pad_right = int(np.floor((224 - img.shape[1]) / 2))
        pad_top = int(np.ceil((224 - img.shape[0]) / 2))
        pad_bottom = int(np.floor((224 - img.shape[0]) / 2))
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, None, [255, 255, 255])
        return img

    def return_testbatch(self, img_idx):
        # image
        img = self.imgs[img_idx]

        # resize and pad
        non_zero = np.nonzero(255 - img)
        y_min = np.min(non_zero[0])
        y_max = np.max(non_zero[0])
        x_min = np.min(non_zero[1])
        x_max = np.max(non_zero[1])
        img = img[y_min:y_max + 1, x_min:x_max + 1]
        img = self.resize_and_pad(img)
        
        # views concat
        img = np.expand_dims(img, 0)
        img_n_views = img
        
        view_filenames = sorted(glob.glob(self.view_dir + '/' + self.retrieved_class[img_idx] + '/*'))
        assert len(view_filenames) == 48, 'pose matching : retrieved_class = {},  len(view_filenames) = {}'.format(self.retrieved_class[img_idx], len(view_filenames))   
        for view_idx, view_filename in enumerate(view_filenames):
            view = cv2.imread(view_filename)
            view = np.expand_dims(view, 0)
            img_n_views = np.concatenate((img_n_views, img), 0) if view_idx != 0 else img 
            img_n_views = np.concatenate((img_n_views, view), 0)
        
        img_n_views = img_n_views / 255.0
        return img_n_views

