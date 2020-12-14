import os, glob
import sys
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import ndimage

def fastener_loader(self, cut_image, component_list, fasteners_loc={}):
    if len(cut_image.shape) == 3:
        gray = cv2.cvtColor(cut_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = cut_image.copy()
    inv_ = 255 - gray
    inv = np.where(inv_>0,255,0)/255 #binarization
    for component in component_list:
        x = component[0]
        y = component[1]
        w = component[2]
        h = component[3]
        inv[y:y+h,x:x+w] = 0

    thick_vertical_eraser = np.ones((1,3))
    horizon_eraser = np.ones((3,1))

    thick_vertical_erase_sub = cv2.erode(inv,thick_vertical_eraser,iterations=2)
    thick_vertical_erase = cv2.dilate(thick_vertical_erase_sub,thick_vertical_eraser,iterations=2)
    thin_vertical_lines_sub = inv - thick_vertical_erase
    thin_vertical_lines = np.clip(thin_vertical_lines_sub,0,1)

    horizon_erase_sub = cv2.erode(thin_vertical_lines,horizon_eraser,iterations=10)
    horizon_erase = cv2.dilate(horizon_erase_sub,horizon_eraser,iterations=10)
    vertical_lines = horizon_erase.copy()
    vertical_lines_img = 255*vertical_lines.astype(np.uint8)

    sub_comb = vertical_lines.copy()
    # kernel_cc = [[1,1,1],[1,1,1],[1,1,1]]
    labeled_fastener, num = ndimage.label(sub_comb)
    obj_fastener = ndimage.find_objects(labeled_fastener)

    candidates_fastener = list(range(1,num+1))
    h_list = []
    w_list = []

    for i in range(1, num+1):
        shape = obj_fastener[i-1]
        w_obj = shape[1].stop - shape[1].start
        h_obj = shape[0].stop - shape[0].start
        h_list.append(h_obj)
        w_list.append(w_obj)

        # if h_obj <= 30 and i in candidates_fastener:
        #     candidates_fastener.remove(i)
        if h_obj >= 300 and i in candidates_fastener:
            candidates_fastener.remove(i)

    fastener_img = np.zeros(sub_comb.shape)
    fastenerInfo_list = list()
    used_fasteners_loc = self.fasteners_loc[self.step_num][self.OCR_to_fastener[self.connector]]
    
    filterd_fastener_id = 0
    for i in range(len(candidates_fastener)):
        fastener_id = i
        shape = obj_fastener[candidates_fastener[i]-1]
        x_start = shape[1].start
        x_end = shape[1].stop
        x_mean = int((x_start+x_end)/2)

        y_start = shape[0].start
        y_end = shape[0].stop

        if len(used_fasteners_loc) >= self.mult:
            count = 0
            for loc in used_fasteners_loc:
                if count == 0:
                    if self.connector != "104322" and self.connector != "122925":
                        x1 = loc[0][0] - 50
                        x2 = loc[0][0] + loc[0][2] + 50
                        y1 = loc[0][1] - 50
                        y2 = loc[0][1] + loc[0][3] + 50
                        if x_mean <= x1 or x_mean >= x2 or y_end <= y1 or y_start >= y2:
                            continue
                        else:
                            fastenerInfo = [filterd_fastener_id,(x_mean,y_start),(x_mean,y_end)]
                            fastenerInfo_list.append(fastenerInfo.copy())
                            filterd_fastener_id += 1
                            count += 1

                            ##### visualization #####
                            region = labeled_fastener[y_start:y_end, x_start:x_end]
                            region = ((region==candidates_fastener[i])*255).astype(np.uint8)
                            fastener_img[y_start:y_end, x_start:x_end] += region
                    elif self.connector =="104322":
                        if self.opt.mission1:
                            x1 = loc[0][0] - 50
                            x2 = loc[0][0] + loc[0][2] + 50
                            y1 = loc[0][1] - 50
                            if x_mean <= x1 or x_mean >= x2 or y_end <= y1:
                                continue
                            else:
                                fastenerInfo = [filterd_fastener_id,(x_mean,y_start),(x_mean,y_end)]
                                fastenerInfo_list.append(fastenerInfo.copy())
                                filterd_fastener_id += 1
                                count += 1

                                ##### visualization #####
                                region = labeled_fastener[y_start:y_end, x_start:x_end]
                                region = ((region==candidates_fastener[i])*255).astype(np.uint8)
                                fastener_img[y_start:y_end, x_start:x_end] += region
                        else:
                            y_length = abs(y_end - y_start)
                            x1 = loc[0][0]
                            x2 = loc[0][0] + loc[0][2]
                            y1 = loc[0][1]
                            y2 = loc[0][1] + loc[0][3]
                            if self.opt.temp and (self.step_num == 4 or self.step_num==5):
                                if x2 > 1200:
                                    continue
                            if x_mean <= x1 or x_mean >= x2 or y_end <= y1:
                                continue
                            else:
                                if y_start <= y2 + 300 or y_length > 70:
                                    fastenerInfo = [filterd_fastener_id,(x_mean,y_start),(x_mean,y_end)]
                                    fastenerInfo_list.append(fastenerInfo.copy())
                                    filterd_fastener_id += 1
                                    count += 1

                                    ##### visualization #####
                                    region = labeled_fastener[y_start:y_end, x_start:x_end]
                                    region = ((region==candidates_fastener[i])*255).astype(np.uint8)
                                    fastener_img[y_start:y_end, x_start:x_end] += region
                    else:
                        x1 = loc[0][0] - 50
                        x2 = loc[0][0] + loc[0][2] + 50
                        y1 = loc[0][1] - 50
                        y2 = loc[0][1] + loc[0][3] + 50
                        if x_mean <= x1 or x_mean >= x2 or y_end <= y1:
                            continue
                        else:
                            fastenerInfo = [filterd_fastener_id,(x_mean,y_start),(x_mean,y_end)]
                            fastenerInfo_list.append(fastenerInfo.copy())
                            filterd_fastener_id += 1
                            count += 1

                            ##### visualization #####
                            region = labeled_fastener[y_start:y_end, x_start:x_end]
                            region = ((region==candidates_fastener[i])*255).astype(np.uint8)
                            fastener_img[y_start:y_end, x_start:x_end] += region

        else:
            fastenerInfo = [filterd_fastener_id,(x_mean,y_start),(x_mean,y_end)]
            fastenerInfo_list.append(fastenerInfo.copy())
            filterd_fastener_id += 1

            ##### visualization #####
            region = labeled_fastener[y_start:y_end, x_start:x_end]
            region = ((region==candidates_fastener[i])*255).astype(np.uint8)
            fastener_img[y_start:y_end, x_start:x_end] += region
    cv2.imwrite(os.path.join(self.v_dir, str(self.step_num) + '_fasteners.png'), fastener_img)
    ###############################

    return fastenerInfo_list

def fastener_loader_v2(self, cut_image, holes, component_list, fasteners_loc={}):
    if len(cut_image.shape) == 3:
        gray = cv2.cvtColor(cut_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = cut_image.copy()
    inv_ = 255 - gray
    inv = np.where(inv_>0,255,0)/255 #binarization

    ###### component 내부 삭제 #########
    for component in component_list:
        x = component[0]
        y = component[1]
        w = component[2]
        h = component[3]
        inv[y:y+h,x:x+w] = 0

    ###### hole 근처 아니면 삭제 #######
    near_hole = inv.copy()
    check_W, check_H = 100, 150
    for part in list(holes.keys()):
        for hole in holes[part]:
            _, x, y, hole_id = hole
            if hole_id == self.connector:
                near_hole[y-check_H:y+check_H, x-check_W:x+check_W] = 0
    inv = inv - near_hole

    thick_vertical_eraser = np.ones((1,3))

    thick_vertical_erase_sub = cv2.erode(inv,thick_vertical_eraser,iterations=2)
    thick_vertical_erase = cv2.dilate(thick_vertical_erase_sub,thick_vertical_eraser,iterations=2)
    thin_vertical_lines_sub = inv - thick_vertical_erase
    thin_vertical_lines = np.clip(thin_vertical_lines_sub,0,1)
    ##### 두꺼운선 삭제 ######

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 15))
    vertical = cv2.morphologyEx(thin_vertical_lines, cv2.MORPH_OPEN, verticalStructure)

    #longStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 60))
    #long = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, longStructure)

    #thickStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    #thick = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, thickStructure)

    #vertical = vertical - long - thick
    #vertical = np.clip(vertical, 0, 1)

    vertical_lines = vertical.copy()
    self.vertical_lines_img = 255*vertical_lines.astype(np.uint8)
    fastener_img = np.zeros(inv.shape)
    labeled_fastener, num = ndimage.label(vertical_lines.copy())
    obj_fastener = ndimage.find_objects(labeled_fastener)

    candidates_fastener = list(range(1,num+1))

    candidate = {}
    for i in candidates_fastener:
        bbox = obj_fastener[i-1]
        x, w, y, h = bbox[1].start, bbox[1].stop - bbox[1].start, bbox[0].start, bbox[0].stop - bbox[0].start
        candidate[i] = [x, y, w, h]


    key_comb = combinations(list(candidate.keys()), 2)
    for key1, key2 in key_comb:
        x1, y1, w1, h1 = candidate[key1]
        x2, y2, w2, h2 = candidate[key2]
        #condition1 = (np.abs(x1 - x2) < 35) and (np.abs((y1 + h1/2) - (y2 + h2/2)) < 5)
        condition1 = False
        #condition2 = (np.abs(x1 - x2) < 2) and (np.abs((y1 + h1/2) - (y2 + h2/2)) < 120)
        if condition1:# or condition2:
            if key1 in candidates_fastener:
                candidates_fastener.remove(key1)
                labeled_fastener[y1:y1+h1, x1:x1+w1] = 0

            if key2 in candidates_fastener:
                candidates_fastener.remove(key2)
                labeled_fastener[y2:y2+h2, x2:x2+w2] = 0

    fastenerInfo_list = list()
    used_fasteners_loc = self.fasteners_loc[self.step_num][self.OCR_to_fastener[self.connector]]
    filterd_fastener_id = 0
    for i in range(len(candidates_fastener)):
        fastener_id = i
        shape = obj_fastener[candidates_fastener[i]-1]
        x_start = shape[1].start
        x_end = shape[1].stop
        x_mean = int((x_start+x_end)/2)

        y_start = shape[0].start
        y_end = shape[0].stop

        if len(used_fasteners_loc) >= self.mult:
            count = 0
            for loc in used_fasteners_loc:
                x1 = loc[0][0] - 50
                x2 = loc[0][0] + loc[0][2] + 50
                y1 = loc[0][1] - 50
                y2 = loc[0][1] + loc[0][3] + 50
                if count == 0:
                    if self.connector != "104322" and self.connector != "122925":
                        if x_mean <= x1 or x_mean >= x2 or y_end <= y1 or y_start >= y2:
                            pass
                        else:
                            fastenerInfo = [filterd_fastener_id,(x_mean,y_start),(x_mean,y_end)]
                            fastenerInfo_list.append(fastenerInfo.copy())
                            filterd_fastener_id += 1
                            count += 1

                            ##### visualization #####
                            region = labeled_fastener[y_start:y_end, x_start:x_end]
                            region = ((region==candidates_fastener[i])*255).astype(np.uint8)
                            fastener_img[y_start:y_end, x_start:x_end] += region
                    else:
                        if x_mean <= x1 or x_mean >= x2:
                            pass
                        else:
                            fastenerInfo = [filterd_fastener_id,(x_mean,y_start),(x_mean,y_end)]
                            fastenerInfo_list.append(fastenerInfo.copy())
                            filterd_fastener_id += 1
                            count += 1

                            ##### visualization #####
                            region = labeled_fastener[y_start:y_end, x_start:x_end]
                            region = ((region==candidates_fastener[i])*255).astype(np.uint8)
                            fastener_img[y_start:y_end, x_start:x_end] += region
        else:
            fastenerInfo = [filterd_fastener_id,(x_mean,y_start),(x_mean,y_end)]
            fastenerInfo_list.append(fastenerInfo.copy())
            filterd_fastener_id += 1

            ##### visualization #####
            region = labeled_fastener[y_start:y_end, x_start:x_end]
            region = ((region==candidates_fastener[i])*255).astype(np.uint8)
            fastener_img[y_start:y_end, x_start:x_end] += region
    cv2.imwrite(os.path.join(self.v_dir, str(self.step_num) + '_fasteners.png'), fastener_img)
    ###############################

    return fastenerInfo_list