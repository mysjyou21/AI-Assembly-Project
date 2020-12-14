import os, glob
import sys
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
sys.path.append(os.path.join((os.path.dirname(os.path.abspath(os.path.dirname(__file__)))),'Grouping_mid'))
from hole_loader import *

def part_hole_connector_loader(self, step_num, part_id_list, part_RT_list, K, H, W, debug_mode=False, debug_img=None, used_parts=[]):
    part_holeInfo_dict = {}
    part_holename_dict = {}
    step_name = 'step' + str(step_num-1)
    for part_idx, part_id in enumerate(part_id_list):
        if part_id in self.mid_id_list:
            hole_XYZ, holename = mid_loader('step%i'%(step_num-1), self.opt.hole_path_2, self.opt.cad_path, used_parts=used_parts)
            # if step_num == 5:
            #     import ipdb; ipdb.set_trace()
            try:
                hole_XYZ = hole_XYZ[part_id]
                holename = holename[part_id]
                if part_id == 'part7' or part_id == 'part8':
                    real_used_idx = [holename.index(x) for x in holename if ('C122620' not in x) or (('C122620' in x) and ('hole_1' in x))]
                    hole_XYZ = hole_XYZ[real_used_idx,:]
                    holename = [holename[i] for i in real_used_idx]
                    hole_id1 = [int(x.split('_')[-1]) for x in holename if 'C122620' not in x]
                    hole_id2 = list(range(7,7+len([x for x in holename if 'C122620' in x])))
                    hole_id = hole_id1 + hole_id2
                else:
                    hole_id = [int(x.split("_")[-1]) for x in holename]
            except KeyError:
                holename = []
                hole_id = [-1]
                hole_XYZ = np.array([[0.0,0.0,0.0]])
            # if step_num == 5:
            #     import ipdb; ipdb.set_trace()

        else:
            hole_XYZ, holename = base_loader(part_id, self.opt.hole_path_2, self.opt.cad_path)
            if part_id == 'part7' or part_id == 'part8':
                real_used_idx = [holename.index(x) for x in holename if ('C122620' not in x) or (('C122620' in x) and ('hole_1' in x))]
                hole_XYZ = hole_XYZ[real_used_idx,:]
                holename = [holename[i] for i in real_used_idx]
                hole_id1 = [int(x.split('_')[-1]) for x in holename if 'C122620' not in x]
                hole_id2 = list(range(7,7+len([x for x in holename if 'C122620' in x])))
                hole_id = hole_id1 + hole_id2
            else:
                hole_id = [int(x.split("_")[-1]) for x in holename]
        RT = part_RT_list[part_idx]
        hole_x, hole_y = project_points(
            hole_XYZ, K, RT, H, W)

        idx = 0
        holeInfo_list = list()
        for xh, yh in zip(hole_x, hole_y):
            holeInfo = [hole_id[idx], xh, yh]
            holeInfo_list.append(holeInfo.copy())
            idx += 1

        holeInfo_list =add_connector(int(part_id[-1]), holeInfo_list)

        part_holeInfo_dict[part_id] = holeInfo_list.copy()
        part_holename_dict[part_id] = holename.copy()

    ##### visualization #####
        for hole_info in holeInfo_list:
            x_coord = hole_info[1]
            y_coord = hole_info[2]
            if debug_mode:
                debug_img = cv2.circle(debug_img, (x_coord,y_coord), 4, (0,0,255), -1)

        cv2.imwrite(os.path.join(self.v_dir, str(self.step_num) + '_hole_check.png'), debug_img)
    #########################

    return part_holename_dict, part_holeInfo_dict

def project_points(points, K, RT, H, W):
    ones = np.ones((points.shape[0], 1))
    xyz = np.append(points[:, :3], ones, axis=1)  # Nx4 : (x, y, z, 1)
    xy = K @ RT @ xyz.T
    coord = xy[:2, :] / xy[2, :]

    coord = (np.floor(coord)).T.astype(int)
    x = np.clip(coord[:, 0], 0, W - 1)
    y = np.clip(coord[:, 1], 0, H - 1)
    return x, y

def add_connector(part_id, holeInfo_list):

    hole_id_list = [x[0] for x in holeInfo_list]
    if part_id == 1:
        new_holeInfo_list = list()
        for hole_id in hole_id_list:
            if hole_id == -1:
                holeInfo = holeInfo_list[0]
                holeInfo.append('-1')
                new_holeInfo_list.append(holeInfo.copy())
            else:
                holeInfo = [x for x in holeInfo_list if hole_id == x[0]][0]
                # holeInfo.append('122925')
                holeInfo.append('-1')
                new_holeInfo_list.append(holeInfo.copy())
        holeInfo_list = sorted(new_holeInfo_list, key=lambda x:x[0])

    if part_id == 2 or part_id == 3:
        new_holeInfo_list = list()
        for hole_id in hole_id_list:
            if hole_id == -1:
                holeInfo = holeInfo_list[0]
                holeInfo.append('-1')
                new_holeInfo_list.append(holeInfo.copy())
            else:
                holeInfo = [x for x in holeInfo_list if hole_id == x[0]][0]
                if holeInfo[0] in [1,2,5,6]:
                    holeInfo.append('101350')
                elif holeInfo[0] in [3,4]:
                    holeInfo.append('104322')
                else:
                    holeInfo.append('122620')
                new_holeInfo_list.append(holeInfo.copy())
        holeInfo_list = sorted(new_holeInfo_list, key=lambda x:x[0])

    if part_id == 4:
        new_holeInfo_list = list()
        for hole_id in hole_id_list:
            if hole_id == -1:
                holeInfo = holeInfo_list[0]
                holeInfo.append('-1')
                new_holeInfo_list.append(holeInfo.copy())
            else:
                holeInfo = [x for x in holeInfo_list if hole_id == x[0]][0]
                if holeInfo[0] in [1,2,3,4,7,8]:
                    holeInfo.append('101350')
                else:
                    holeInfo.append('104322')
                new_holeInfo_list.append(holeInfo.copy())
        holeInfo_list = sorted(new_holeInfo_list, key=lambda x:x[0])

    if part_id == 5 or part_id == 6:
        new_holeInfo_list = list()
        for hole_id in hole_id_list:
            if hole_id == -1:
                holeInfo = holeInfo_list[0]
                holeInfo.append('-1')
                new_holeInfo_list.append(holeInfo.copy())
            else:
                holeInfo = [x for x in holeInfo_list if hole_id == x[0]][0]
                if holeInfo[0] in [1,3,4,6,7,8,10]:
                    holeInfo.append('101350')
                else:
                    holeInfo.append('104322')
                new_holeInfo_list.append(holeInfo.copy())
        holeInfo_list = sorted(new_holeInfo_list, key=lambda x:x[0])

    if part_id == 7 or part_id == 8:
        new_holeInfo_list = list()
        for hole_id in hole_id_list:
            if hole_id == -1:
                holeInfo = holeInfo_list[0]
                holeInfo.append('-1')
                new_holeInfo_list.append(holeInfo.copy())
            else:
                holeInfo = [x for x in holeInfo_list if hole_id == x[0]][0]
                if holeInfo[0] in [1,2,5,6]:
                    holeInfo.append('101350')
                elif holeInfo[0] in [3,4]:
                    holeInfo.append('104322')
                else:
                    holeInfo.append('122925')
                new_holeInfo_list.append(holeInfo.copy())
        holeInfo_list = sorted(new_holeInfo_list, key=lambda x:x[0])
    return holeInfo_list
