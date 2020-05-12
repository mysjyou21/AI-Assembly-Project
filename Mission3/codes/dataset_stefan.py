import tensorflow as tf
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
        # img filenames
        self.IMG_DIR = os.path.join(DATA_DIR, 'IMAGES')
        img_dirs = glob.glob(self.IMG_DIR+'/*')
        img_dirs = sorted(img_dirs[:])
        self.img_filenames = []
        for img_di in img_dirs:
            filenames = glob.glob(img_di+'/'+'*.png')
            shuffle(filenames)
            self.img_filenames.append(filenames)

#        # model filenames
#        self.MODEL_DIR = os.path.join(DATA_DIR, 'TARGET_MODELS/models')
#        self.model_filenames = glob.glob(self.MODEL_DIR+'/*.off')  # or png

        # view filenames
        self.VIEW_DIR = os.path.join(DATA_DIR, 'VIEWS_GRAY_BLACK')
        view_dirs = glob.glob(self.VIEW_DIR+'/*')
        self.view_len = 0
        self.view_filenames = {}
        for view_di in view_dirs:
            filenames = glob.glob(view_di+'/*.png')
            filenames = sorted(filenames)
            self.view_len += len(filenames)
            model_name = view_di.split('/')[-1]
            self.view_filenames[model_name] = []
            step = len(filenames)//self.view_num
            for filename in filenames:  # maybe view_num == len(filenames)  #idx in range(self.view_num): #test
                self.view_filenames[model_name].append(filename.split('/')[-1].rstrip('.png')) #filenames[min(idx*step, len(filenames)-1)].split('/')[-1].rstrip('.png'))

        # import ipdb <<< isaac
        # ipdb.set_trace() <<< isaac

        # class files
        self.class_idxtonum = {}
        self.class_numtoidx = {}
        self.img_clsnum = {}
        self.model_clsnum = {}
        img_cls_path = DATA_DIR+'/Image_%s.cla' % self.mode.capitalize()
        model_cls_path = DATA_DIR+'/Model.cla'
        self.read_class_file(img_cls_path, 'img')
        self.read_class_file(model_cls_path, 'model')

        # for calculating epoch..
        self.whole_batch()

    def whole_batch(self):
        """ calculate the total number of models(the standard of the epoch) """
        self.whole_batch_size = 0
        for cls_model in self.class_imgs: #models: 
            self.whole_batch_size += len(self.class_imgs[cls_model])#self.class_models[cls_model])
        print('whole batch: %d' % self.whole_batch_size)

    def _batch(self, class_id, C, K, size=None, mode='train'):
        """  """
        imgs = []
        imgs_cls = []
        views = []
        views_cls = []
        views_pos = []
        num_pos = int(len(self.view_filenames[self.class_numtoidx[class_id[0]]])/self.view_num)
        assert len(self.view_filenames[self.class_numtoidx[class_id[0]]])%self.view_num == 0
        for idx in class_id:
            # Select images
            img_folder = self.IMG_DIR+'/'+self.class_numtoidx[idx]#+'/train' #+self.mode.lower()
            img_idx = self.class_imgs[idx]
            indices = [i for i in range(len(img_idx))]
            shuffle(indices)
            indices = indices[0:K]
            for index in indices:
                img = cv2.imread(img_folder+'/'+img_idx[index]+'.png') #, cv2.IMREAD_GRAYSCALE)
                img = transform(img, size=size) # HAVE-TO-DO!!
#                img = (cv2.resize(img, size)).astype(np.float32)/255.
                imgs.append(img)
                imgs_cls.append(idx)

            # Select views
            indices = self.class_models[idx]
#            indices = indices[randrange(len(indices))] # selected K models
            if mode=='train':
                shuffle(indices)
            indices = indices[0:K]
            for index in indices:
                model_name = index
                view_folder = self.VIEW_DIR+'/'+model_name
                view_filenames = self.view_filenames[model_name]
                step = len(view_filenames)//self.view_num if mode=='train' else 1
                num_steps = self.view_num if mode=='train' else len(view_filenames)

                views_model = []
                for _step in range(num_steps):
                    view_filename = view_filenames[_step*step]
                    img = cv2.imread(view_folder+'/'+view_filename+'.png') #, cv2.IMREAD_GRAYSCALE)
                    img = transform(img, size=size)
#                    img = (cv2.resize(img, size)).astype(np.float32)/255.
                    views_model.append(img)
                    pos = int(view_filename.split('_')[-2])
                    views_pos.append(pos)
                    if mode=='test':
                        views.append(views_model)
                        views_cls.append(idx)
                        views_model = []
                if mode=='train':
                    views.append(views_model)
                    views_cls.append(index)

        imgs = np.asarray(imgs)
        imgs_cls = np.asarray(imgs_cls)
        views = np.asarray(views)
        views_cls = np.asarray(views_cls)
        views_pos = np.asarray(views_pos)
        views_pos = np.reshape(views_pos, [-1, self.view_num])

        return imgs, imgs_cls, views, views_cls, views_pos

    def make_batch(self, max_epoch, C=4, K=4, size=(224,224), is_shuffle=True):
        """ return C*K batches, until epoch reaches max_epoch """
        steps = int(self.whole_batch_size/C/K)  # the standard of "1 epoch" -- view (smaller than img)
        class_list = [i for i in range(0, self.class_len)]

        epoch = -1
        while epoch < max_epoch: # 1 batch after epoch reaches max_epoch
            epoch += 1
            for step in range(steps):
                if is_shuffle:
                    if (step*C)%self.class_len == 0:
                        shuffle(class_list)

                # select C classes
                start_idx = (step*C)%(self.class_len-C+1)
                class_id = class_list[start_idx:start_idx+C]
                batch_imgs, batch_imgs_cls, batch_views, batch_views_cls, _ = self._batch(class_id, C, K, size)

                yield [batch_imgs, batch_views, batch_imgs_cls, batch_views_cls]

    def _test_batch(self, step, size=None):
        imgs = []
        imgs_cls = []
        views = []
        views_cls = []
        class_id = [i for i in range(0, self.class_len)]
        for idx in class_id:
            # Select images
            img_folder = self.IMG_DIR+'/'+self.class_numtoidx[idx]#+'/train' #+self.mode.lower()
            img_idx = self.class_imgs[idx]
            img = cv2.imread(img_folder+'/'+img_idx[step]+'.png') #, cv2.IMREAD_GRAYSCALE)
#             img = transform(img, size=size) # HAVE-TO-DO!!
            img = (cv2.resize(img, size)).astype(np.float32)/255.
            imgs.append(img)
            imgs_cls.append(idx)

            # Select views
            indices = self.class_models[idx]
            for index in indices:
                model_name = index
                view_folder = self.VIEW_DIR+'/'+model_name
                view_filenames = self.view_filenames[model_name]
                views_model = []
                for view_filename in view_filenames:
                    img = cv2.imread(view_folder+'/'+view_filename+'.png') #, cv2.IMREAD_GRAYSCALE)
#                    img = transform(img, size=size)
                    img = (cv2.resize(img, size)).astype(np.float32)/255.
                    views_model.append(img)
                views.append(views_model)
                views_cls.append(idx)

        imgs = np.asarray(imgs)
        imgs_cls = np.asarray(imgs_cls)
        views = np.asarray(views)
        views_cls = np.asarray(views_cls)

        return imgs, imgs_cls, views, views_cls


    def test_batch(self, K=1, size=(224,224), mode='pos'):
        """ test batch for retrieval, STEFAN """
        if mode=='pos':
            C = 1
            num_step = self.class_len
        else:
            C = self.class_len
            num_step = self.img_len//C
            assert self.img_len%C == 0

        step = -1
        while step < num_step:
            step += 1
            if mode == 'pos':
                class_id = [step]
                batch_imgs, batch_imgs_cls, batch_views, batch_views_cls, batch_views_pos = self._batch(class_id,C=C,K=K, size=size, mode='test') #step, size)
                batchs = [batch_imgs, batch_views, batch_imgs_cls, batch_views_cls, batch_views_pos]
            else:
                batch_imgs, batch_imgs_cls, batch_views, batch_views_cls = self._test_batch(step, size)
                batchs = [batch_imgs, batch_views, batch_imgs_cls, batch_views_cls, None]
            yield batchs
