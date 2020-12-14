import os
import glob
import cv2
import pickle
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import DataLoader

from ..models.fastener_block_model import FastenerBlockModel

import sys
sys.path.append('./function/utilities')
from function.utilities.utils import *

class Color():
    def __init__(self):
        """color palette (BGR) """
        self.colors = {
            0 : (0, 0, 0),
            1 : (0, 0, 255),
            2 : (0, 165, 255),
            3 : (0, 255, 255),
            4 : (0, 128, 0),
            5 : (255, 0, 0),
            6 : (130, 0, 75),
            7 : (255, 0, 255),
            8 : (255, 255, 0),
        }
    def __call__(self, x):
        return self.colors[x]

class FastenerDetection():
    def __init__(self, args):
        self.args = args.opt
        self.checkpoint_path = self.args.fastener_model_path
        self.fastener_function_path = './function/Fastener'

        self.fastener_names_list = ['bracket', 'long_screw', 'short_screw', 'wood_pin']
        self.OCR_to_fastener = {
            '122620' : 'bracket',
            '104322' : 'long_screw',
            '122925' : 'short_screw',
            '101350' : 'wood_pin'
        }
        self.fastener_to_OCR = {v:k for k, v in self.OCR_to_fastener.items()}

        # load weights
        with args.graph_fastener.as_default():
            sess = args.sess_fastener
            self.model = FastenerBlockModel(3, 5) # in channels, id channels
            print('FASTENER MODEL : Loading saved model from', self.checkpoint_path + '/fastener.pt')
            checkpoint = torch.load(self.checkpoint_path + '/fastener.pt', map_location='cuda:0')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model.cuda(0)

        refresh_folder(args.opt.fastener_detection_prediction_maps_path)
        refresh_folder(args.opt.fastener_detection_visualization_path)

    def test(self, args, step_num):
        """detect fasteners

        update self.fastener_loc

        Args:
            args: self of Assembly class
            step_num: step number
        """
        color = Color()
        image = args.steps[step_num].copy()
        H, W, _ = image.shape
        with torch.no_grad():
            # load image on GPU
            image[2300:, ...] = 255 # remove bottom region
            image = cv2.resize(image.astype(np.uint8), (480, 640), interpolation=cv2.INTER_AREA)
            image_tensor = transforms.ToTensor()(image)
            image_tensor = torch.unsqueeze(image_tensor, 0)
            image_tensor = image_tensor.cuda(0)

            # feed forward
            idmask_pred = self.model(image_tensor)
            idmask_pred = torch.argmax(idmask_pred, dim=1).squeeze().cpu().detach().numpy()  # leave maximum value
            ID = idmask_pred.astype(np.uint8)
            if args.opt.save_fastener_prediction_maps:
                _H, _W = idmask_pred.shape
                ID_color = np.zeros((_H, _W, 3)).astype(np.uint8)
                for h in range(_H):
                    for w in range(_W):
                        ID_color[h, w, :] = color(idmask_pred[h, w])
                save_path = args.opt.fastener_detection_prediction_maps_path
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_path + '/STEP{}_ID.png'.format(step_num), np.concatenate((image, ID_color), axis=1))

            # post-process ID-map with bubble(circle, rectangle) locations
            _H, _W = ID.shape
            group_circles_locs = args.circles_loc[step_num]
            for group_circles_loc in group_circles_locs:
                for circle_loc in group_circles_locs:
                    x, y, w, h = circle_loc[:4]
                    x, y, w, h = int(x*_W/W), int(y*_H/H), int(w*_W/W), int(h*_H/H)
                    ID[y:y+h,x:x+w] = 0
            group_rectangles_locs = args.rectangles_loc[step_num]
            for group_rectangles_loc in group_rectangles_locs:
                for rectangle_loc in group_rectangles_locs:
                    x, y, w, h = rectangle_loc[:4]
                    x, y, w, h = int(x*_W/W), int(y*_H/H), int(w*_W/W), int(h*_H/H)
                    ID[y:y+h,x:x+w] = 0
            if args.opt.save_fastener_prediction_maps:
                _H, _W = ID.shape
                ID_color = np.zeros((_H, _W, 3)).astype(np.uint8)
                for h in range(_H):
                    for w in range(_W):
                        ID_color[h, w, :] = color(ID[h, w])
                cv2.imwrite(save_path + '/STEP{}_ID_1.png'.format(step_num), np.concatenate((image, ID_color), axis=1))

            # post-process ID-map with connector serial OCRs
            group_OCRs = args.connectors_serial_OCR[step_num]
            fastener_names = []
            for group_OCR in group_OCRs:
                for OCR in group_OCR:
                    try:
                        fastener_name = self.OCR_to_fastener[OCR]
                        fastener_names.append(fastener_name)
                    except:
                        pass
            unused_fastener_names = list(set(self.fastener_names_list) - set(fastener_names))
            unused_fastener_ids = [self.fastener_names_list.index(x) + 1 for x in unused_fastener_names]
            for unused_fastener_id in unused_fastener_ids:
                ID = np.where(ID == unused_fastener_id, 0, ID)
            if args.opt.save_fastener_prediction_maps:
                _H, _W = ID.shape
                ID_color = np.zeros((_H, _W, 3)).astype(np.uint8)
                for h in range(_H):
                    for w in range(_W):
                        ID_color[h, w, :] = color(ID[h, w])
                cv2.imwrite(save_path + '/STEP{}_ID_2.png'.format(step_num), np.concatenate((image, ID_color), axis=1))

            # update self.fasteners_loc
            
            _H, _W = ID.shape
            fastener_dict = dict()
            for idx, fastener_name in enumerate(self.fastener_names_list):
                fastener_dict[fastener_name] = list()
                fastener_id = idx + 1
                mask = np.where(ID == fastener_id, 255, 0).astype(np.uint8)
                retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                if retval == 1:
                    continue
                stats = stats[1:, :]
                centroids = centroids[1:, :]
                for stat, centroid in zip(stats, centroids):
                    indiv_fastener_loc = list()
                    x, y, w, h = stat[:4]
                    x, y, w, h = int(x*W/_W), int(y*H/_H), int(w*W/_W), int(h*H/_H)
                    indiv_fastener_loc.append(tuple([x, y, w, h]))
                    cx, cy = centroid
                    cx, cy = int(cx*W/_W), int(cy*H/_H)
                    indiv_fastener_loc.append(tuple([cx, cy]))
                    fastener_dict[fastener_name].append(indiv_fastener_loc)
            args.fasteners_loc[step_num] = fastener_dict
    
    
    def visualize(self, args, step_num):
        """visualize fasteners

        save fastener detection visualization of step

        Args:
            args: self of Assembly class
            step_num: step number

        Uses:
            self.steps
            self.fastener_loc
        """
        if args.opt.save_fastener_visualization:
            color = Color()
            image = args.steps[step_num].copy()
            H, W, _ = image.shape
            fastener_dict = args.fasteners_loc[step_num]
            for idx, fastener_name in enumerate(self.fastener_names_list):
                fastener_locs = fastener_dict[fastener_name]
                fastener_id = idx + 1
                for fastener_loc in fastener_locs:
                    x, y, w, h = fastener_loc[0]
                    px, py = x, y - 20
                    cv2.rectangle(image, (x, y), (x+w, y+h), color=color(fastener_id), thickness=2)
                    cv2.putText(image, fastener_name, (px, py), cv.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=color(fastener_id), thickness=3)
            save_path = args.opt.fastener_detection_visualization_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path + '/STEP{}.png'.format(step_num), image)
                    

