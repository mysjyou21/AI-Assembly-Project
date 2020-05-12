import glob
import cv2
import numpy as np
from random import randint
import os
# for one image in a folder

DATA_DIR = '../../data/stefan_8pages/IMAGES'
FOLDERS = glob.glob(DATA_DIR + '/*')

for FOLDER in FOLDERS:
    FILENAMES = glob.glob(FOLDER + '/*.png')
    print(FILENAMES)
    for FILENAME in FILENAMES:
        img = cv2.imread(FILENAME)
        H, W, _ = img.shape
        filename = os.path.basename(FILENAME).rstrip('.png')
        # filename = FILENAME.split('/')[-1].rstrip('.png')
        index = filename.rfind('_')
        name = filename[0:index]
        idx = int(filename[index + 1:])
        for i in range(9):
            pts1 = np.float32([[0, 0], [W // 2, H // 2], [W // 2, 0]])
            pts2 = np.float32([[randint(0, W // 16), randint(0, H // 16)], [randint(W // 2 - W // 16, W // 2 + W // 16), randint(H // 2 - H // 16, H // 2 + H // 16)], [randint(W // 2 - W // 16, W // 2 + W // 16), randint(0, H // 16)]])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(img, M, (W, H), borderValue=(255, 255, 255))
            idx += 1
            cv2.imwrite(FOLDER + '/' + name + '_' + str(idx) + '.png', dst)
