#import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import sys

sys.path.append('./miscc')
from utils import load_classfile_str, transform
from random import shuffle, randrange

class Dataset(object):
    def __init__(self, DATA_DIR, view_num=4, mode='train'):
        self.mode = mode.split('_')[0]
#        self.mode = 'train'
        self.train_mode = mode.split('_')[-1]
        self.shuffle = True if self.mode == 'train' else False

        self.view_num = view_num
        self.load_data(DATA_DIR)

    def read_class_file(self, cla_path, mode='model'):
        """ read .cla file
            class file Structure:
                PSB Version_num
                numClasses numModels

                className parentClassName numModelsInClass
                model identifier ...
        """
        class_num, instance_num, class_idxtonum, class_numtoidx, class_instances, instance_clsnum = load_classfile_str(cla_path)
        self.class_len = class_num
        if self.class_idxtonum == {}:
            self.class_idxtonum = class_idxtonum
        if self.class_numtoidx == {}:
            self.class_numtoidx = class_numtoidx
        if mode == 'model':
            self.class_models = class_instances
            self.model_clsnum = instance_clsnum
            self.model_len = instance_num
        elif mode == 'img':
            self.class_imgs = class_instances
            self.img_clsnum = instance_clsnum
            self.img_len = instance_num
        else:
            print('Type Error')

    def load_data(self, DATA_DIR):
        """ set img_filenames, view_filenames, img_class, view_class from directories"""
        # class files
        self.class_idxtonum = {}
        self.class_numtoidx = {}
        self.img_clsnum = {}
        self.model_clsnum = {}
        img_cls_path = DATA_DIR+'/Image_%s.cla' % self.mode.capitalize()
        model_cls_path = DATA_DIR+'/Model.cla'
        self.read_class_file(img_cls_path, 'img')
        self.read_class_file(model_cls_path, 'model')

        # img filenames
        self.IMG_DIR = os.path.join(DATA_DIR, 'IMAGES')
        class_keys = [x for x in self.class_numtoidx.keys()]
        self.img_filenames = []
        for key in class_keys:
            img_dir = os.path.join(self.IMG_DIR, self.class_numtoidx[key], self.mode.lower())
            filenames = [os.path.join(img_dir, x+'.png') for x in self.class_imgs[key]]
            filenames = sorted(filenames[:])
            if self.shuffle:
                shuffle(filenames)
            self.img_filenames.append(filenames)

#        # model filenames
#        self.MODEL_DIR = os.path.join(DATA_DIR, 'TARGET_MODELS/models')
#        self.model_filenames = glob.glob(self.MODEL_DIR+'/*.off')  # or png

        # view filenames
        self.VIEW_DIR = os.path.join(DATA_DIR, 'VIEWS') #_GRAY_BLACK')
        self.view_len = 0
        self.view_filenames = {}
        for key in class_keys:
            view_dir = os.path.join(self.VIEW_DIR, self.class_numtoidx[key])
            filenames = glob.glob(view_dir+'/*.png')
            filenames = sorted(filenames[:])
            if self.shuffle:
                shuffle(filenames)
            model_name = self.class_numtoidx[key]
            self.view_filenames[model_name] = []
            self.view_len += self.view_num
            step = 1 #len(filenames)//self.view_num
            for idx in range(len(filenames)): #self.view_num):#filename in filenames[0:self.view_num]:  # maybe view_num == len(filenames)
                self.view_filenames[model_name].append(filenames[min(idx*step, len(filenames)-1)].split('/')[-1].rstrip('.png'))
#                self.view_len += 1

        # for calculating epoch..
#        self.whole_batch_size = self.img_len if (self.train_mode == 'img' or self.mode == 'test') else self.model_len

        self.whole_batch_size = self.model_len if self.train_mode == 'view' else self.img_len

    def whole_batch(self):
        """ calculate the total number of models(the standard of the epoch) """
        self.whole_batch_size = 0
        if self.train_mode == 'img':
            for cls_img in self.class_imgs:
                self.whole_batch_size += len(self.class_imgs[cls_img])
        else:
            for cls_model in self.class_models:
                self.whole_batch_size += len(self.class_models[cls_model])
        print('whole batch: %d' % self.whole_batch_size)

    def _batch(self, class_id, img_K_id, view_K_id, C, K, size=None):
        """  """
        imgs = []
        imgs_cls = []
        views = []
        views_cls = []
        for idx in class_id:
            # Select images
            img_folder = self.IMG_DIR+'/'+self.class_numtoidx[idx]+'/'+self.mode.lower()
            img_idx = self.class_imgs[idx]
            indices = img_K_id[idx] #[i for i in range(len(img_idx))]
            for index in indices:
                img = cv2.imread(img_folder+'/'+index+'.png') #, cv2.IMREAD_GRAYSCALE)
                img = transform(img, size=size) 
                imgs.append(img)
                imgs_cls.append(idx)

            # Select views
            indices = view_K_id[idx] #self.class_models[idx]
            for index in indices:
                model_name = index
                view_folder = self.VIEW_DIR+'/'+model_name
                view_filenames = self.view_filenames[model_name].copy()
                if self.shuffle:
                    shuffle(view_filenames)
                views_model = []
#                for view_filename in view_filenames:
                step = len(view_filenames)//self.view_num
                for index in range(self.view_num):
                    view_filename = view_filenames[min(index*step, len(view_filenames)-1)]
                    img = cv2.imread(view_folder+'/'+view_filename+'.png') #, cv2.IMREAD_GRAYSCALE)
                    img = transform(img, size=size)
                    views_model.append(img)
                views.append(views_model)
                views_cls.append(idx)

        imgs = np.asarray(imgs)
        imgs_cls = np.asarray(imgs_cls)
        views = np.asarray(views)
        views_cls = np.asarray(views_cls)

        return imgs, imgs_cls, views, views_cls

    def _Ks(self):
        imgs_K_id = {}
        views_K_id = {}
        for c in range(self.class_len):
            img_indices = self.class_imgs[c]
            if self.shuffle:
                shuffle(img_indices)
            view_indices = self.class_models[c]
            if self.shuffle:
                shuffle(view_indices)
            imgs_K_id[c]=img_indices
            views_K_id[c] = view_indices

        return imgs_K_id, views_K_id

    def _batch_Ks(self, imgs_K_id, views_K_id, class_id, start_K, K=4):
        b_imgs_K_id = {}
        b_views_K_id = {}
        for c in class_id:
            indices = imgs_K_id[c]
            img_start_K = start_K%(len(indices)-K+1)
            b_imgs_K_id[c] = imgs_K_id[c][img_start_K:img_start_K+K]
            indices = views_K_id[c]
            if len(indices)>2*K:
                view_start_K = start_K%(len(indices)-K+1)
            else:
                view_start_K = start_K%len(indices)
                if view_start_K+K > len(indices):
                    view_start_K = max(view_start_K-K, 0)
            b_views_K_id[c] = views_K_id[c][view_start_K:view_start_K+K]

        return b_imgs_K_id, b_views_K_id

    def make_batch(self, max_epoch, C=4, K=4, size=(224,224), is_shuffle=True):
        """ return C*K batches, until epoch reaches max_epoch """
        steps = int((self.whole_batch_size)/C/K) #if self.mode=='train' else int(self.img_len/C/K) # the standard of "1 epoch" -- view (smaller than img)
        img_per_class = int(self.whole_batch_size/C)
        class_list = [i for i in range(0, self.class_len)]

        epoch = -1
        while epoch < max_epoch: # 1 batch after epoch reaches max_epoch
            epoch += 1
            class_ind = -1
            for step in range(steps):
                step_class_ind = (step*C) // self.class_len
                if step_class_ind > class_ind:
                    if self.shuffle:
                        shuffle(class_list)
                    imgs_K_id, views_K_id = self._Ks()
                    class_ind += 1

                # select C classes
                start_C_idx = step*C%(self.class_len-C+1)
                class_id = class_list[start_C_idx:start_C_idx+C]
#                print('class id', class_id)
                start_K_idx = step//(self.class_len//C)*K

                b_imgs_K_id, b_views_K_id = self._batch_Ks(imgs_K_id, views_K_id, class_id, start_K_idx, K)
#                for k,v in b_imgs_K_id.items():
#                    print('img K ', k, v)
#                for k,v in b_views_K_id.items():
#                    print('view K ',k, v)

                batch_imgs, batch_imgs_cls, batch_views, batch_views_cls = self._batch(class_id, b_imgs_K_id, b_views_K_id, C, K, size)

                yield [batch_imgs, batch_views, batch_imgs_cls, batch_views_cls]
