################### General: 임의의 2개 + 101350 ######################
import os, glob
import sys
import cv2
import shutil
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
sys.path.append(os.path.join((os.path.dirname(os.path.abspath(os.path.dirname(__file__)))),'Grouping_mid'))
from hole_loader import *
from find_matching import *
from fastener_loader import *
from part_hole_connector_loader import *
from calculate_dist import *

class MSN2_hole_detector():
    def __init__(self, opt):
        self.opt = opt
        directory = os.path.join(self.opt.intermediate_results_path, 'hole_detection_visualization')
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            os.makedirs(directory)
        

    def main_hole_detector(self, step_num, step_images, parts_info, connectors, mults, \
        mid_id_list, K, mid_RT, RTs_dict, hole_pairs, component_list, mid_base, find_mid=False, used_parts=[], fasteners_loc={}):
        self.step_num = step_num
        self.parts_info = parts_info
        self.connectors = connectors
        self.mults = mults
        self.mid_id_list = mid_id_list
        self.mid_RT = mid_RT
        self.RTs_dict = RTs_dict
        self.hole_pairs = hole_pairs
        self.component_list = component_list
        self.mid_base = mid_base
        self.fasteners_loc = fasteners_loc
        self.v_dir = os.path.join(self.opt.intermediate_results_path, 'hole_detection_visualization')
        if not os.path.exists(self.v_dir):
            os.mkdir(self.v_dir)

        self.OCR_to_fastener = {
            '122620' : 'bracket',
            '104322' : 'long_screw',
            '122925' : 'short_screw',
            '101350' : 'wood_pin'
        }

        self.cut_image = step_images[step_num]
        H = self.cut_image.shape[0]
        W = self.cut_image.shape[1]
        self.mid_base[step_num] = self.mid_id_list
        step_connector = self.connectors[step_num][0]
        # 연결자가 있으면 사용하고, 없으면 hole 정보 없이 self.parts_info(group_as_action으로 넘어가는 정보) 생성
        if len(step_connector) == 0 or ((step_num > 2) and int(mid_RT.sum())==0):
            part_id_list = [x[0] for x in self.parts_info[step_num]]
            new_id_list = [x for x in part_id_list if x not in self.mid_id_list]
            step_name = 'step' + str(step_num-1)
            matched_pose_mid = self.closest_gt_RT_index(mid_RT)
            used_part_hole = [[step_name, matched_pose_mid, []]]
            if len(new_id_list) != 0:
                for new_id in new_id_list:
                    new_RT = [x[1] for x in self.parts_info[step_num] if new_id in x][0]
                    matched_pose_new = self.closest_gt_RT_index(new_RT)
                    used_part_hole.append([new_id, matched_pose_new, []])
            else:
                pass
            step_info = used_part_hole+[['']]
            self.parts_info[step_num] = step_info
            self.hole_pairs[step_num] = []

            print(step_info)

            return self.parts_info, self.hole_pairs, self.mid_base
        elif len(step_connector) != 0:
            connector = step_connector[0]
            self.connector = connector

        step_mult = self.mults[step_num][0]
        if len(step_mult) != 0 and int(step_mult[0]) != -1:
            connector_num = int(step_mult[0])
        else:
            connector_num = int(self.mults[1][0][0])
        self.mult = connector_num
        
        # 체결선의 수가 mult number보다 작아서 indexError나는 경우에 대한 처리
        fastenerInfo_list = fastener_loader(self, self.cut_image.copy(), component_list, self.fasteners_loc)
        if len(fastenerInfo_list) < self.mult:
            part_id_list = [x[0] for x in self.parts_info[step_num]]
            new_id_list = [x for x in part_id_list if x not in self.mid_id_list]
            step_name = 'step' + str(step_num-1)
            matched_pose_mid = self.closest_gt_RT_index(mid_RT)
            used_part_hole = [[step_name, matched_pose_mid, []]]
            if len(new_id_list) != 0:
                for new_id in new_id_list:
                    new_RT = [x[1] for x in self.parts_info[step_num] if new_id in x][0]
                    matched_pose_new = self.closest_gt_RT_index(new_RT)
                    used_part_hole.append([new_id, matched_pose_new, []])
            else:
                pass
            step_info = used_part_hole+[['']]
            self.parts_info[step_num] = step_info
            self.hole_pairs[step_num] = []

            print(step_info)

            return self.parts_info, self.hole_pairs, self.mid_base
        # part ID, Pose, KRT
        step_parts_info = self.parts_info[step_num]
        self.K = K
        part_id_list = [x[0] for x in step_parts_info]
        part_RT_list = [x[1] for x in step_parts_info]

        self.new_id_list = [x for x in part_id_list if x not in self.mid_id_list]
        part_holename_dict, part_holeInfo_dict, hole_id_dict = part_hole_connector_loader(self, step_num, part_id_list, part_RT_list, K, H, W, True, self.cut_image.copy(), used_parts=used_parts)
        fastenerInfo_list = fastener_loader(self, self.cut_image.copy(), component_list, self.fasteners_loc)
        part_holeInfo_dict_initial = part_holeInfo_dict.copy()
        
        if self.opt.temp and (self.step_num == 4 or self.step_num==5):
            connector_num = 2
            self.mult = 2

        # 입력된 연결자 정보로, hole 활성화
        for part_id, part_holeInfo in part_holeInfo_dict.items():
            part_holeInfo_temp_ = part_holeInfo_dict[part_id].copy()
            part_holeInfo_temp = [x for x in part_holeInfo_temp_ if connector in x]
            part_holeInfo_dict[part_id] = part_holeInfo_temp.copy()

        part_holeInfo_dict_up = {}
        part_holeInfo_dict_down = {}
        new_part_holeInfo_dict_up = {}
        for key, val in part_holeInfo_dict.items():
            val_temp = sorted(val,key=lambda x:x[2])
            cut_idx = int(len(val_temp)/2)
            part_holeInfo_dict_up[key] = val_temp[:cut_idx].copy()
            part_holeInfo_dict_down[key] = val_temp[cut_idx:].copy()
            if key in self.new_id_list:
                new_part_holeInfo_dict_up[key] = val_temp[:cut_idx].copy()
        # 중간 산출물 단위로 묶어줌 --> 중간산출물 + 새로운 부품으로 나뉘어짐
        matched_pose_temp = {}
        for part_info in step_parts_info:
            part_RT = self.closest_gt_RT_index(part_info[1])
            if part_RT not in matched_pose_temp.keys():
                matched_pose_temp[part_RT] = [part_info[0]]
            else:
                id_list = matched_pose_temp[part_RT].copy()
                id_list.append(part_info[0])
                matched_pose_temp[part_RT] = id_list
        total_count = len(matched_pose_temp.keys()) # --> 큰 단위로, 2개 이상이면 결합 진행

        # 위에 있는 연결자 파악
        connectors_loc = self.fasteners_loc[self.step_num][self.OCR_to_fastener[self.connector]]
        self.use_connectors_loc = False
        if len(connectors_loc) >= connector_num:
            self.use_connectors_loc = True
            connectors_center = [x[1] for x in connectors_loc]
            part_up_connectors = {}
            if self.connector == '101350':
                for new_id in self.new_id_list:
                    if new_id != 'part5' and new_id != 'part6':
                        up_connectors = list()
                        new_id_holes = part_holeInfo_dict_up[new_id]
                        max_holes_x = max([x[1] for x in new_id_holes])
                        min_holes_x = min([x[1] for x in new_id_holes])
                        max_holes_y = max([x[2] for x in new_id_holes])
                        for connector_center in connectors_center:
                            if connector_center[0] < max_holes_x + 100 and connector_center[0] > min_holes_x - 100 and connector_center[1] < max_holes_y+50:
                                up_connectors.append(connector_center)
                                part_up_connectors[new_id] = up_connectors.copy()
            up_connectors_num = sum([len(val) for key, val in part_up_connectors.items()])

        if (total_count>1 or (len(self.mid_id_list)!=0 and len(self.new_id_list)!=0) or len(self.new_id_list)>1) \
            and self.connector != '122620' and self.connector != '122925' and self.connector != '101350':
            step_info, hole_pairs = hole_pair_matching(self, step_num, connector_num, fastenerInfo_list, part_holeInfo_dict, part_holename_dict, step_parts_info, self.cut_image.copy())
            # hole이 아예 없으면 임의로 넣어줌
            
            if len(step_info[-1][0]) != 0:
                for i,partInfo in enumerate(step_info[:-1]):
                    partname = partInfo[0]
                    holeInfo = partInfo[2].copy()
                    if len(holeInfo) == 0:
                        hole_id_temp = part_holeInfo_dict_down[partname][-1][0]
                        hole_id_dict_temp = hole_id_dict[partname]
                        hole_id_position = hole_id_dict_temp.index(hole_id_temp)
                        holename_temp = part_holename_dict[partname][hole_id_position]
                        holename_temp = partname + '_1-' + holename_temp
                        partInfo[-1] = [holename_temp]
                        step_info[i] = partInfo.copy()
            self.parts_info[step_num] = step_info
            self.hole_pairs[step_num] = hole_pairs
            print(step_info)

        elif (total_count>1 or (len(self.mid_id_list)!=0 and len(self.new_id_list)!=0) or len(self.new_id_list)>1) and self.connector == "101350":
            if self.use_connectors_loc:
                step_info_sub1 = point_matching(self,step_num, up_connectors_num, fastenerInfo_list.copy(), new_part_holeInfo_dict_up.copy(), part_holename_dict.copy(), step_parts_info.copy(), self.cut_image.copy())
                connector_num -= up_connectors_num
            else:
                connector_num_temp = int(connector_num/2)
                step_info_sub1 = point_matching(self,step_num, connector_num_temp, fastenerInfo_list.copy(), new_part_holeInfo_dict_up.copy(), part_holename_dict.copy(), step_parts_info.copy(), self.cut_image.copy())
                matched_count = 0
                for partInfo in step_info_sub1:
                    if partInfo != ['']:
                        matched_count += len(partInfo[2])
                connector_num -= matched_count
            step_info_sub2, hole_pairs = hole_pair_matching(self, step_num, connector_num, fastenerInfo_list.copy(), part_holeInfo_dict.copy(), part_holename_dict.copy(), step_parts_info.copy(), self.cut_image.copy())
            step_info = list()
            for part1_info in step_info_sub1:
                for part2_info in step_info_sub2:
                    if part1_info == ['']:
                        continue
                    else:
                        part1_id = part1_info[0]
                        part2_id = part2_info[0]
                        if part1_id == part2_id:
                            part1_RT = part1_info[1]
                            new_info_holes = part1_info[2] + part2_info[2]
                            step_info.append([part1_id, part1_RT,new_info_holes])
                        else:
                            step_info.append(part2_info)
            if len(step_info) == 0:
                step_info = step_info_sub2

            # 체결 관계는 있는데 hole이 아예 없으면 넣어줌
            if len(step_info[-1][0]) != 0:
                for i,partInfo in enumerate(step_info[:-1]):
                    partname = partInfo[0]
                    holeInfo = partInfo[2].copy()
                    if len(holeInfo) == 0:
                        hole_id_temp = part_holeInfo_dict_down[partname][-1][0]
                        hole_id_dict_temp = hole_id_dict[partname]
                        hole_id_position = hole_id_dict_temp.index(hole_id_temp)
                        holename_temp = part_holename_dict[partname][hole_id_position]
                        partInfo[-1] = [holename_temp]
                        step_info[i] = partInfo.copy()
            self.parts_info[step_num] = step_info
            self.hole_pairs[step_num] = hole_pairs
            print(step_info)
        else:
            step_info = point_matching(self,step_num, connector_num, fastenerInfo_list, part_holeInfo_dict, part_holename_dict, step_parts_info, self.cut_image.copy())
            # hole이 아예 없으면 임의로 넣어줌
            if len(step_info[-1][0]) != 0:
                for i,partInfo in enumerate(step_info[:-1]):
                    partname = partInfo[0]
                    holeInfo = partInfo[2].copy()
                    if len(holeInfo) == 0:
                        hole_id_temp = part_holeInfo_dict_down[partname][-1][0]
                        hole_id_dict_temp = hole_id_dict[partname]
                        hole_id_position = hole_id_dict_temp.index(hole_id_temp)
                        holename_temp = part_holename_dict[partname][hole_id_position]
                        partInfo[-1] = [holename_temp]
                        step_info[i] = partInfo.copy()
            self.parts_info[step_num] = step_info
            self.hole_pairs[step_num] = []
            print(step_info)
        return self.parts_info, self.hole_pairs, self.mid_base

    def closest_gt_RT_index(self, RT_pred):
        def R_distance(RT1, RT2):
            R1 = RT1[:, :3]
            R2 = RT2[:, :3]
            R = R1.T @ R2
            theta = np.rad2deg(np.arccos((np.trace(R) - 1)/2))
            return theta
        return np.argmin([R_distance(RT, RT_pred) for RT in self.RTs_dict])
