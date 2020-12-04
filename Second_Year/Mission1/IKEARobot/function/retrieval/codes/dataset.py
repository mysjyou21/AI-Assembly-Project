import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import sys

from miscc.utils import transform
from random import shuffle, randrange, sample

class Dataset(object):
    def __init__(self, args, step_num, candidate_classes, DATA_DIR='./input/stefan', size=(224,224), view_num=12):
        self.args = args
        self.step_num = step_num
        self.candidate_classes = candidate_classes
        self.DATA_DIR = DATA_DIR
        self.size = size
        self.view_num = view_num
        self.part_imgs = self.args.parts[self.step_num]

    """ make view retrieval testset from given CAD model (a set of view files are already made) ###
    def load_image_step(self):
        IMG_DIR = os.path.join(self.DATA_DIR, 'step_part_images')
        IMG_DIR = os.path.join(IMG_DIR, str(self.step_num))
        step_imgs = sorted(glob.glob(IMG_DIR+'/*'))

        return step_imgs"""

    def load_view_all(self):
        VIEW_DIR = self.args.opt.retrieval_views_path #'./input/stefan/views/VIEWS_GRAY_BLACK'
        view_dirs = [VIEW_DIR + '/' + c for c in sorted(self.candidate_classes)]
        view_dict = {}

        ### read 48 view images for each class(view_dict={class_name:[48 view imgs]}), and sample view_nums images (view_test_dict={class_name:[12 view imgs]}) ###
        for view_dir in view_dirs:
            class_name = view_dir.split('/')[-1]
            class_files = glob.glob(view_dir+'/*')
            class_files = sorted(class_files)
            view_dict[class_name] = class_files
        
        return view_dict

    def _test_batch(self,img_index):
        imgs = []
        views = []
        views_cls = []

        ### step_image dataset (numpy array type) ###
        """
        imgs_filepath = self.load_image_step()

        for filepath in imgs_filepath:
            img = cv2.imread(filepath)
            img = (cv2.resize(img, self.size)).astype(np.float32)/255.
            imgs.append(img)
        """
        # for part_img in self.part_imgs:
        #     img = (cv2.resize(part_img, self.size)).astype(np.float32)/255.
        #     imgs.append(img)
        img = (cv2.resize(self.part_imgs[img_index], self.size)).astype(np.float32)/255.
        imgs.append(img)
        # import ipdb; ipdb.set_trace()
        ### view dataset (class_index dictionary, numpy array view image) ###
        view_dict = self.load_view_all()
        class_id = {} # class_index dictionary
        idx = 0
        for key in view_dict:
            class_id[idx] = key
            views_filepath = view_dict[key]
            view_batch_step = len(views_filepath)//self.view_num
            num_view_batch_steps = self.view_num
            views_model=[]
            for _step in range(num_view_batch_steps):
                view_filename = views_filepath[_step*view_batch_step]
                img = cv2.imread(view_filename)
                img = (cv2.resize(img, self.size)).astype(np.float32)/255.
                views_model.append(img)
            views.append(views_model)
            views_cls.append(idx)
            idx += 1

        return imgs, views, views_cls, class_id

    def test_batch(self,img_index):
        imgs, batch_views, batch_views_cls, step_class_id = self._test_batch(img_index)
        batchs = [imgs, batch_views, batch_views_cls, step_class_id]
        yield batchs    


            
