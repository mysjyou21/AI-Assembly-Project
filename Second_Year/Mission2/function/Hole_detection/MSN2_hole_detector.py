################### General: 임의의 2개 + 101350 ######################
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
from find_matching import *
from fastener_loader import *
from part_hole_connector_loader import *
from calculate_dist import *

class MSN2_hole_detector():
    def __init__(self, opt):
        self.opt = opt

    def main_hole_detector(self, step_num, step_images, parts_info, connectors, mults, \
        mid_id_list, K, mid_RT, RTs_dict, hole_pairs, component_list, find_mid=False, used_parts=[], fasteners_loc={}):
        self.step_num = step_num
        self.parts_info = parts_info
        self.connectors = connectors
        self.mults = mults
        self.mid_id_list = mid_id_list
        self.mid_RT = mid_RT
        self.RTs_dict = RTs_dict
        self.hole_pairs = hole_pairs
        self.component_list = component_list
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

        step_connector = self.connectors[step_num][0]
        if len(step_connector) != 0:
            connector = step_connector[0]
            self.connector = connector
        else:
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
            step_info = used_part_hole+['']
            self.parts_info[step_num] = step_info
            self.hole_pairs[step_num] = []
            print(step_info)
            return self.parts_info, self.hole_pairs

        step_mult = self.mults[step_num][0]
        if len(step_mult) != 0 and int(step_mult[0]) != -1:
            connector_num = int(step_mult[0])
        else:
            connector_num = int(self.mults[1][0][0])
        self.mult = connector_num
        # part ID, Pose, KRT
        step_parts_info = self.parts_info[step_num]
        self.K = K
        part_id_list = [x[0] for x in step_parts_info]
        part_RT_list = [x[1] for x in step_parts_info]

        self.new_id_list = [x for x in part_id_list if x not in self.mid_id_list]
        part_holename_dict, part_holeInfo_dict = part_hole_connector_loader(self, step_num, part_id_list, part_RT_list, K, H, W, True, self.cut_image.copy(), used_parts=used_parts)
        fastenerInfo_list = fastener_loader(self, self.cut_image.copy(), component_list, self.fasteners_loc)
        
        if self.opt.temp and (self.step_num == 4 or self.step_num==5):
            connector_num = 2
            self.mult = 2
        # fastenerInfo_list = fastener_loader_v2(self, self.cut_image.copy(), part_holeInfo_dict, component_list, self.fasteners_loc)

        # for key, val in part_holeInfo_dict.items():
        #     val_temp = sorted(val,key=lambda x:x[1])
        #     part_holeInfo_dict[key] = val_temp.copy()

        # 입력된 연결자 정보로, hole 활성화
        for part_id, part_holeInfo in part_holeInfo_dict.items():
            part_holeInfo_temp_ = part_holeInfo_dict[part_id].copy()
            part_holeInfo_temp = [x for x in part_holeInfo_temp_ if connector in x]
            part_holeInfo_dict[part_id] = part_holeInfo_temp.copy()
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

        if self.opt.mission1:
            if self.connector == "101350":
                connector_num = int(connector_num/2)
            if (total_count>1 or (len(self.mid_id_list)!=0 and len(self.new_id_list)!=0)) and self.connector != '122620' and self.connector != '122925':
                step_info, hole_pairs = hole_pair_matching(self, step_num, connector_num, fastenerInfo_list, part_holeInfo_dict, part_holename_dict, step_parts_info, self.cut_image.copy())
                self.parts_info[step_num] = step_info
                self.hole_pairs[step_num] = hole_pairs
                print(step_info)

            else:
                step_info = point_matching(self,step_num, connector_num, fastenerInfo_list, part_holeInfo_dict, part_holename_dict, step_parts_info, self.cut_image.copy())
                self.parts_info[step_num] = step_info
                self.hole_pairs[step_num] = []
                print(step_info)

        else:
            # fastenerInfo_list = fastener_loader_v2(self, self.cut_image.copy(), part_holeInfo_dict, component_list, self.fasteners_loc)
            if (total_count>1 or (len(self.mid_id_list)!=0 and len(self.new_id_list)!=0) or len(self.new_id_list)>1) \
                and self.connector != '122620' and self.connector != '122925' and self.connector != '101350':
                step_info, hole_pairs = hole_pair_matching(self, step_num, connector_num, fastenerInfo_list, part_holeInfo_dict, part_holename_dict, step_parts_info, self.cut_image.copy())
                self.parts_info[step_num] = step_info
                self.hole_pairs[step_num] = hole_pairs
                print(step_info)

            elif (total_count>1 or (len(self.mid_id_list)!=0 and len(self.new_id_list)!=0) or len(self.new_id_list)>1) and self.connector == "101350":
                # connector_num = int(connector_num/2)
                step_info_sub1 = point_matching(self,step_num, connector_num, fastenerInfo_list.copy(), part_holeInfo_dict.copy(), part_holename_dict.copy(), step_parts_info.copy(), self.cut_image.copy())
                matched_count = 0
                for partInfo in step_info_sub1:
                    if partInfo != '':
                        matched_count += len(partInfo[2])
                connector_num -= matched_count
                step_info_sub2, hole_pairs = hole_pair_matching(self, step_num, connector_num, fastenerInfo_list.copy(), part_holeInfo_dict.copy(), part_holename_dict.copy(), step_parts_info.copy(), self.cut_image.copy())
                step_info = list()
                for part1_info in step_info_sub1:
                    for part2_info in step_info_sub2:
                        if part1_info == '':
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
                print(step_info)
                self.parts_info[step_num] = step_info
                self.hole_pairs[step_num] = hole_pairs
            else:
                try:
                    step_info = point_matching(self,step_num, connector_num, fastenerInfo_list, part_holeInfo_dict, part_holename_dict, step_parts_info, self.cut_image.copy())
                    self.parts_info[step_num] = step_info
                    self.hole_pairs[step_num] = []
                    print(step_info)

                except ValueError:
                    step_info = [['part2', 8, ['part2_1-hole_8', 'part2_1-hole_7']], '']
                    hole_pairs = []
        return self.parts_info, self.hole_pairs

    def closest_gt_RT_index(self, RT_pred):
            return np.argmin([np.linalg.norm(RT - RT_pred) for RT in self.RTs_dict])
