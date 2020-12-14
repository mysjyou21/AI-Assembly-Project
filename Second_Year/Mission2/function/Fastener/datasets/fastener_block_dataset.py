import os
import cv2
import json
import pickle
import torch
import numpy as np
from scipy.spatial.distance import cdist

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from config import *
from datasets.add_non_part_components import Synthesizer
from datasets.add_fasteners import Fastener_Synthesizer


class FastenerBlockDataset(Dataset):

    def __init__(self, args, mode='train', transform=None):
        self.args = args
        self.imgs = input_images


    def __len__(self):
        return len(self.img_adrs)


    def __getitem__(self, i):
        if self.mode == 'train':
            img_adr, image, IDmask = self.get_masks(i)
            return img_adr, image, IDmask
        elif self.mode == 'test':
            img_adr, image = self.get_masks(i)
            return img_adr, image


    def get_masks(self, i):
        args = self.args

        if self.mode == 'train':
            
            # load image, parse names
            img_adr = self.img_adrs[i]
            scene_path = os.path.dirname(img_adr).replace('/rgb', '')
            scene_name = img_adr.split('/')[-3]
            rgb_name = os.path.basename(img_adr).split('.')[0]
            image = cv2.imread(img_adr)

            # add fasteners
            image, fastener_mask = self.fastener_synth.add_fasteners(image)

            # add non part components (bubbles)
            if args.add_non_part_components:
                image, non_part_components_mask = self.synth.add_non_part_components(image)
            else:
                non_part_components_mask = np.zeros((image.shape[0], image.shape[1])).astype(np.uint8)
            
            def erase_non_part_component_region(mask, non_part_components_mask):
                temp_mask = np.where(non_part_components_mask > 0, 0, mask)
                return temp_mask
            fastener_mask = erase_non_part_component_region(fastener_mask, non_part_components_mask)

            # resize
            image = cv2.resize(image, dsize=(args.w, args.h), interpolation=cv2.INTER_AREA)
            IDmask = cv2.resize(fastener_mask, dsize=(args.w, args.h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

            image = self.transform(image)
            IDmask = (torch.from_numpy(IDmask)).type(torch.int64)
            
            return img_adr, image, IDmask
        
        elif self.mode == 'test':
            args = self.args

            # load image, parse names
            img_adr = self.img_adrs[i]
            image = cv2.imread(img_adr)

            # remove carpet
            image = self.prep_carpet(image)
            
            # resize
            image = cv2.resize(image, dsize=(args.w, args.h), interpolation=cv2.INTER_AREA)
            image = self.transform(image)
            
            return img_adr, image


    def prep_carpet(self, img):
        """
        Make gray-color region (like carpet in 'stefan-03.png') to white.

        Input
            img: 3-channel or 2-channel image

        Output
            img_masked: processed image
        """
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = np.copy(img)

        img_gray = 255 - img_gray

        # carpet pixel value in stefan-03.png is 44 (211 before inverting)
        # Make carpet-shape mask
        margin = 20
        carpet_pixel_value = 44
        threshold_min = carpet_pixel_value - margin
        threshold_max = carpet_pixel_value + margin
        mask = (img_gray >= threshold_min) * (img_gray < threshold_max)
        mask = 1 - np.uint8(mask)
        if len(img.shape) == 3:
            mask = np.stack((mask, mask, mask), axis=2)

        # Get masked image
        img_masked = mask * (255 - img)
        img_masked = 255 - img_masked

        return img_masked

