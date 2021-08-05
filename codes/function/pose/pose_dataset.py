import tensorflow as tf
import numpy as np
np.set_printoptions(suppress=True, precision=6)
import os
import glob
import cv2
import sys
import time
import pickle
from random import shuffle
from .pose_gt import *
from matplotlib import pyplot as plt
from time import time as t


class Dataset(object):
    def __init__(self, args):
        self.args = args
        with open(args.args.cornerpoints_adr, 'rb') as f:
            self.CORNERPOINTS = np.load(f)
        with open(args.args.RTs_adr, 'rb') as f:
            self.RTS = np.load(f)
        with open(args.args.view_imgs_adr, 'rb') as f:
            self.VIEW_IMGS = np.load(f)


    def return_testbatch(self):
        """make test batch"""
        args = self.args

        # crop images with detection_results & batch
        imgs = []
        for i in range(args.num_test_imgs):
            whole_img = args.imgs[i]
            for x, y, w, h in args.detection_results[i]:
                cropped_img = whole_img[y : y + h, x : x + w, :]
                img = self.resize_and_pad(cropped_img)
                img = np.expand_dims(img, 0)
                imgs = np.concatenate((imgs, img), 0) if len(imgs) else img
        imgs = imgs.astype(np.float32)
        imgs /= 255
        

        # choose cornerpoints according to retrieval_resluts & batch
        cornerpoints = []
        for i in range(args.num_test_imgs):
            for cad_name in args.retrieval_results[i]:
                cornerpoint = self.CORNERPOINTS[args.cad_names.index(cad_name)]
                cornerpoint = np.expand_dims(cornerpoint, 0)
                cornerpoints = np.concatenate((cornerpoints, cornerpoint), 0) if len(cornerpoints) else cornerpoint

        return imgs, cornerpoints


    def resize_and_pad(self, img, a=150):
        """Make image similar to detection result
        
        Resize non-zero part of image to size (a, a).
        Then pad image to size (224, 224).
        Used for train batch (when adding data augmentation) and test batch

        Args:
            img: (unit8) image to be resized and padded
            a: (int) resizing size before padding (default=150)
        
        Returns:
            (uint8) resized and padded image
        """
        # find object region
        non_zero = np.nonzero(255 - img)
        y_min = np.min(non_zero[0])
        y_max = np.max(non_zero[0])
        x_min = np.min(non_zero[1])
        x_max = np.max(non_zero[1])
        img = img[y_min:y_max + 1, x_min:x_max + 1]
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
    






        
