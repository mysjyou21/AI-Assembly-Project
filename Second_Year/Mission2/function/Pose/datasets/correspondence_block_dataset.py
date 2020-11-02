import os
import cv2
import json
import pickle
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class CorrespondenceBlockDataset(Dataset):

    def __init__(self, args, input_images, retrieved_class):
        self.args = args
        self.imgs = input_images
        self.retrieved_class = retrieved_class



    def __len__(self):
        return len(self.img_indices)


    def __getitem__(self, i):
        args = self.args
        img_adr = self.img_adrs[self.img_indices[i]]
        image = cv2.imread(img_adr)
        image = cv2.resize(image.astype(np.uint8), (240, 320), interpolation=cv2.INTER_NEAREST)
        scene_path = os.path.dirname(img_adr).replace('/rgb', '')
        scene_name = img_adr.split('/')[-3]
        rgb_name = os.path.basename(img_adr).split('.')[0]
        mask_visib_adrs = sorted(glob.glob(scene_path + '/mask_visib/*'))
        mask_visib_adrs = [x for x in mask_visib_adrs if os.path.basename(x).startswith(rgb_name)]
        mask_visibs = [cv2.imread(mask_visib_adr, cv2.IMREAD_GRAYSCALE) for mask_visib_adr in mask_visib_adrs]


        image = self.transform(image)

        return img_adr, image
