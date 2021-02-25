import glob
import cv2
import numpy as np
from random import randint
import os
# for one image in a folder

#DATA_DIRS = ['../../data/new_view/ivar/IMAGES', '../../data/new_view/kaustby/IMAGES', '../../data/new_view/norrayd/IMAGES', '../../data/new_view/ingolf/IMAGES']
DATA_DIRS = ['../../data/new_view/kaustby/IMAGES']
for DATA_DIR in DATA_DIRS:
    FOLDERS = glob.glob(DATA_DIR + '/*')
    print(FOLDERS)
    
    for FOLDER in FOLDERS:
        FILENAMES = glob.glob(FOLDER + '/*_0.png')
        print(FILENAMES)
        for FILENAME in FILENAMES:
            img = cv2.imread(FILENAME)
            H, W, _ = img.shape
            filename = os.path.basename(FILENAME).rstrip('.png')
    #        if filename=='chair_part2_0' or filename=='chair_part3_0':
    #            bordervalue=(211, 212, 209)
    #        else:
            bordervalue=(255,255,255)
            # filename = FILENAME.split('/')[-1].rstrip('.png')
            index = filename.rfind('_')
            name = filename[0:index]
            idx = int(filename[index + 1:])
            temp = np.concatenate([img[0], img[-1], img[:][0], img[:][-1]])
    #        bordervalue = (np.median(temp[:][0]), np.median(temp[:][1]), np.median(temp[:][2]))
    #        print(bordervalue)
            for i in range(9):
                pts1 = np.float32([[0, 0], [W // 2, H // 2], [W // 2, 0]])
                pts2 = np.float32([[randint(0, W // 18), randint(0, H // 18)], [randint(W // 2 - W // 18, W // 2 + W // 18), randint(H // 2 - H // 18, H // 2 + H // 18)], [randint(W // 2 - W // 18, W // 2 + W // 18), randint(0, H // 18)]]) #//16
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(img, M, (W, H), borderValue=bordervalue)
                idx += 1
                cv2.imwrite(FOLDER + '/' + name + '_' + str(idx) + '.png', dst)
