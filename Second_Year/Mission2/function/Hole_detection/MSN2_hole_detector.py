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

class MSN2_hole_detector():
    def __init__(self, opt):
        self.opt = opt

        ###################### sample2/cad_info2 바뀌면 바꾸어야함 ############################
       # if self.opt.assembly_name == 'sample2' or self.opt.assembly_name == 'sample2_test':
       #     self.opt.hole_path_2 = self.opt.hole_path
        ######################################################################################

    def main_hole_detector(self, step_num, step_images, parts_info, connectors, mults, \
        mid_id_list, K, mid_RT, RTs_dict, hole_pairs, component_list, find_mid=False, used_parts=[]):
        self.step_num = step_num
        self.parts_info = parts_info
        self.connectors = connectors
        self.mults = mults
        self.mid_id_list = mid_id_list
        self.mid_RT = mid_RT
        self.RTs_dict = RTs_dict
        self.hole_pairs = hole_pairs
        self.component_list = component_list
        self.v_dir = os.path.join(self.opt.intermediate_results_path, 'hole_detection_visualization')
        if not os.path.exists(self.v_dir):
            os.mkdir(self.v_dir)

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

        # part ID, Pose, KRT
        step_parts_info = self.parts_info[step_num]
        self.K = K
        part_id_list = [x[0] for x in step_parts_info]
        part_RT_list = [x[1] for x in step_parts_info]

        self.new_id_list = [x for x in part_id_list if x not in self.mid_id_list]
        fastenerInfo_list = self.fastener_loader(self.cut_image.copy(), component_list)
        part_holename_dict, part_holeInfo_dict = self.part_hole_connector_loader(step_num, part_id_list, part_RT_list, K, H, W, True, self.cut_image.copy(), used_parts=used_parts)
        for part_id, part_holeInfo in part_holeInfo_dict.items():
            part_holeInfo_temp_ = part_holeInfo_dict[part_id].copy()
            part_holeInfo_temp = [x for x in part_holeInfo_temp_ if connector in x]
            part_holeInfo_dict[part_id] = part_holeInfo_temp.copy()

        if 'part7' in self.new_id_list and len(self.mid_id_list)==0:
            self.new_id_list.remove('part7')
            self.mid_id_list.append('part7')
            self.mid_RT = [x[1] for x in self.parts_info[step_num] if 'part7' in x][0]
        if 'part8' in self.new_id_list and len(self.mid_id_list)==0:
            self.new_id_list.remove('part8')
            self.mid_id_list.append('part8')
            self.mid_RT = [x[1] for x in self.parts_info[step_num] if 'part8' in x][0]

        # if len(self.new_id_list)!=0 and len(self.mid_id_list)!=0 and self.connector != '101350':
        #     self.hole_pair_matching(step_num, connector_num, fastenerInfo_list, part_holeInfo_dict, part_holename_dict, step_parts_info, self.cut_image.copy())
        # elif len(self.new_id_list)!=0 and len(self.mid_id_list)!=0 and self.connector == '101350':
        #     import ipdb; ipdb.set_trace()
        #     connector_num = int(connector_num/2)
        #     import ipdb; ipdb.set_trace()
        #     self.hole_pair_matching(step_num, connector_num, fastenerInfo_list, part_holeInfo_dict, part_holename_dict, step_parts_info, self.cut_image.copy())
        #     self.point_matching(step_num, connector_num, fastenerInfo_list, part_holeInfo_dict, part_holename_dict, step_parts_info, self.cut_image.copy())

        if self.connector == '101350': ##### General Code 완성하면 지울 예정 #####
            connector_num = int(connector_num/2)

        if len(self.new_id_list)!=0 and len(self.mid_id_list)!=0:
            self.hole_pair_matching(step_num, connector_num, fastenerInfo_list, part_holeInfo_dict, part_holename_dict, step_parts_info, self.cut_image.copy())

        else:
            self.point_matching(step_num, connector_num, fastenerInfo_list, part_holeInfo_dict, part_holename_dict, step_parts_info, self.cut_image.copy())

        return self.parts_info, self.hole_pairs

    def fastener_loader(self, cut_image, component_list):
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
        self.vertical_lines_img = 255*vertical_lines.astype(np.uint8)

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

        fastenerInfo_list = list()
        self.fastener_img = np.zeros(sub_comb.shape)
        for i in range(len(candidates_fastener)):
            fastener_id = i
            shape = obj_fastener[candidates_fastener[i]-1]
            x_start = shape[1].start
            x_end = shape[1].stop
            x_mean = int((x_start+x_end)/2)

            y_start = shape[0].start
            y_end = shape[0].stop

            fastenerInfo = [fastener_id,(x_mean,y_start),(x_mean,y_end)]
            fastenerInfo_list.append(fastenerInfo.copy())

            ##### visualization #####
            region = labeled_fastener[y_start:y_end, x_start:x_end]
            region = ((region==candidates_fastener[i])*255).astype(np.uint8)
            self.fastener_img[y_start:y_end, x_start:x_end] += region

        # cv2.imwrite(os.path.join(self.v_dir, str(self.step_num) + '_vertical_lines.png'), self.vertical_lines_img)
        cv2.imwrite(os.path.join(self.v_dir, str(self.step_num) + '_fasteners.png'), self.fastener_img)
        ###############################

        return fastenerInfo_list

    def part_hole_connector_loader(self, step_num, part_id_list, part_RT_list, K, H, W, debug_mode=False, debug_img=None, used_parts=[]):
        part_holeInfo_dict = {}
        part_holename_dict = {}
        step_name = 'step' + str(step_num-1)
        for part_idx, part_id in enumerate(part_id_list):
            if part_id in self.mid_id_list:
                hole_XYZ, holename = mid_loader('step%i'%(step_num-1), self.opt.hole_path_2, self.opt.cad_path, used_parts=used_parts)
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
            hole_x, hole_y = self.project_points(
                hole_XYZ, K, RT, H, W)

            idx = 0
            holeInfo_list = list()
            for xh, yh in zip(hole_x, hole_y):
                holeInfo = [hole_id[idx], xh, yh]
                holeInfo_list.append(holeInfo.copy())
                idx += 1

            holeInfo_list =self.add_connector(int(part_id[-1]), holeInfo_list)

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

    def project_points(self, points, K, RT, H, W):
        ones = np.ones((points.shape[0], 1))
        xyz = np.append(points[:, :3], ones, axis=1)  # Nx4 : (x, y, z, 1)
        xy = K @ RT @ xyz.T
        coord = xy[:2, :] / xy[2, :]

        coord = (np.floor(coord)).T.astype(int)
        x = np.clip(coord[:, 0], 0, W - 1)
        y = np.clip(coord[:, 1], 0, H - 1)
        return x, y

    def add_connector(self, part_id, holeInfo_list):

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

    def calculate_dist_pair(self, fastenerInfo, p1_hole, p2_hole):
        fastener_x = fastenerInfo[1][0]
        fastener_y_up = fastenerInfo[2][1]
        fastener_y_down = fastenerInfo[1][1]

        p1_hole_x = p1_hole[1]
        p1_hole_y = p1_hole[2]

        p2_hole_x = p2_hole[1]
        p2_hole_y = p2_hole[2]

        dist1 = ((fastener_x - p1_hole_x)**2 + (fastener_y_up - p1_hole_y)**2)**(1/2) + \
            ((fastener_x - p2_hole_x)**2 + (fastener_y_down - p2_hole_y)**2)**(1/2)
        dist2 = ((fastener_x - p1_hole_x)**2 + (fastener_y_down - p1_hole_y)**2)**(1/2) + \
            ((fastener_x - p2_hole_x)**2 + (fastener_y_up - p2_hole_y)**2)**(1/2)

        dist = min(dist1, dist2)
        return dist

    def calculate_dist_point(self, fastenerInfo, p_hole):

        fastener_x = fastenerInfo[2][0]
        fastener_y_down = fastenerInfo[2][1]

        p_hole_x = p_hole[1]
        p_hole_y = p_hole[2]

        dist = ((fastener_x - p_hole_x)**2 + (fastener_y_down - p_hole_y)**2)**(1/2)

        return dist


    def hole_pair_matching(self, step_num, connector_num, fastenerInfo_list, part_holeInfo_dict, part_holename_dict, step_parts_info, cut_image=None):
        fastener_connectingInfo = {}
        for fastenerInfo in fastenerInfo_list:
            fastener_id = fastenerInfo[0]
            part_id_list = sorted(part_holeInfo_dict.keys())
            dist_list = list()
            distInfo_list = list()

            for p1 in part_id_list:
                for p2 in part_id_list:
                    if p1 < p2:
                        p1_holeInfo = sorted(part_holeInfo_dict[p1])
                        p2_holeInfo = sorted(part_holeInfo_dict[p2])
                        for p1_hole in p1_holeInfo:
                            for p2_hole in p2_holeInfo:
                                p1_hole_id = p1_hole[0]
                                p2_hole_id = p2_hole[0]
                                dist = self.calculate_dist_pair(fastenerInfo, p1_hole, p2_hole)
                                distInfo = [dist, (p1, p1_hole_id), (p2, p2_hole_id)]
                                dist_list.append(dist)
                                distInfo_list.append(distInfo)
            min_idx = dist_list.index(min(dist_list))
            min_distInfo = distInfo_list[min_idx]
            assert min(dist_list) == min_distInfo[0]
            fastener_connectingInfo[fastener_id] = min_distInfo.copy()

        hole_matching_info_list = list()
        sub_final_dist_list = list()

        for fastener, connectingInfo in fastener_connectingInfo.items():
            sub_final_dist_list.append(connectingInfo[0])

        dist_list = sorted(sub_final_dist_list)
        fastener_idxs = [sub_final_dist_list.index(i) for i in dist_list]

        fastener_id_list = list()
        for fastener_idx in fastener_idxs:
            addition = True
            hole_matching_info_raw = fastener_connectingInfo[fastener_idx]
            part_hole_1 = hole_matching_info_raw[1]
            part_hole_2 = hole_matching_info_raw[2]
            if len(hole_matching_info_list) != 0:
                for holes in hole_matching_info_list:
                    for hole in holes:
                        if part_hole_1 == hole or part_hole_2 == hole:
                            addition = False
            if addition:
                hole_matching_info = [hole_matching_info_raw[1],hole_matching_info_raw[2]]
                hole_matching_info_list.append(hole_matching_info.copy())
                fastener_id_list.append(fastener_idx)

        hole_matching_info_list = hole_matching_info_list[:connector_num]
        connected_fastener_list = fastener_id_list[:connector_num]

        ######## Visualization ###########
        inv_img = cut_image.copy()
        for idx in range(len(connected_fastener_list)):
            connected_fastener = connected_fastener_list[idx]
            connected_fastenerInfo = [x for x in fastenerInfo_list if connected_fastener in x][0]
            hole_matching_info = hole_matching_info_list[idx]

            fastener_coord1 = connected_fastenerInfo[1]
            fastener_coord2 = connected_fastenerInfo[2]
            fastener_coord_list = [fastener_coord1, fastener_coord2]

            inv_img = cv2.line(inv_img, fastener_coord1, fastener_coord2, (0,255,0), 2)

            hole_coord_list = list()
            for part_hole in hole_matching_info:
                part_id = part_hole[0]
                hole_id = part_hole[1]
                hole_info = [x for x in part_holeInfo_dict[part_id] if hole_id in x][0]
                h_x = hole_info[1]
                h_y = hole_info[2]
                hole_coord_list.append((h_x,h_y))
                inv_img = cv2.circle(inv_img, (h_x,h_y), 4, (0,0,255), -1)

            fastener_coord_list = sorted(fastener_coord_list, key=lambda x:x[1])
            hole_coord_list = sorted(hole_coord_list, key=lambda x:x[1])
            for f_coord, h_coord in zip(fastener_coord_list, hole_coord_list):
                inv_img = cv2.line(inv_img, f_coord, h_coord, (255,0,0), 1)
        cv2.imwrite(os.path.join(self.v_dir, str(self.step_num)+'_check_connecting.png'), inv_img)
######################################################################################################

        ##### convert hole name to other format as same as step.json(part_#-hole_#) #####
        hole_matching_info_list_rename = list()
        for hole_matching_info in hole_matching_info_list:
            part_hole_raname_list = list()
            for hole in hole_matching_info:
                part_holename_list = part_holename_dict[hole[0]]
                part_name = hole[0]
                hole_idx = hole[1]
                if part_name == 'part7':
                    if hole_idx < 7:
                        part_hole_rename = (part_name,[x for x in part_holename_list if (hole_idx == int(x.split('_')[-1])) and ('C122620' not in x)][0])
                    else:
                        hole_idx -= 7
                        part_hole_rename = (part_name,[x for x in part_holename_list if 'C122620' in x][hole_idx])

                elif part_name == 'part8':
                    if hole_idx < 7:
                        part_hole_rename = (part_name,[x for x in part_holename_list if (hole_idx == int(x.split('_')[-1])) and ('C122620' not in x)][0])
                    else:
                        hole_idx -= 7
                        part_hole_rename = (part_name,[x for x in part_holename_list if 'C122620' in x][hole_idx])
                elif part_name in self.mid_id_list:
                    part_hole_rename = (part_name,[x for x in part_holename_list if hole_idx == int(x.split('_')[-1])][0])
                else:
                    part_hole_rename = (part_name,part_name+'_1-'+part_holename_list[hole_idx-1])
                part_hole_raname_list.append(part_hole_rename)
            hole_matching_info_list_rename.append(part_hole_raname_list)

        ##### Mid info modified #####
        mid_info_list = list()
        mid_used_hole = list()

        if len(self.mid_id_list) == 1 and 'part7' in self.mid_id_list:
            mid_name = 'part7'
        elif len(self.mid_id_list) == 1 and 'part8' in self.mid_id_list:
            mid_name = 'part8'
        else:
            mid_name = 'step' + str(step_num-1)

        mid_RT_class = self.closest_gt_RT_index(self.mid_RT)
        for mid_id in self.mid_id_list:
            for hole_matching_info in hole_matching_info_list_rename:
                for part_hole in hole_matching_info:
                    if mid_id in part_hole:
                        mid_used_hole.append(part_hole[1])

        check_pass = False
        if self.opt.mission1:  ############# STEFAN Tuning ##################
            if (mid_name == 'part8') and self.connector == '101350':
                check_pass = True
                mid_part_holeInfo = part_holeInfo_dict[mid_name]
                mid_part_holeId_list = [x[0] for x in mid_part_holeInfo]
                mid_part_holenames = part_holename_dict[mid_name]
                mid_used_hole = [mid_part_holenames[i-1] for i in mid_part_holeId_list]

        mid_info_list.append([mid_name, mid_RT_class, mid_used_hole])

        ##### New info modified #####
        new_parts_info_list = list()
        new_part_used_hole = list()
        # connectivity = list()
        connectivity = ''
        new_parts_info = [x for x in sorted(step_parts_info, key=lambda x:x[0]) if x[0] not in self.mid_id_list]
        for new_part_info in new_parts_info:
            new_part_id = new_part_info[0]
            new_part_RT = new_part_info[1]
            new_part_RT_class = self.closest_gt_RT_index(new_part_RT)
            for hole_matching_info in hole_matching_info_list_rename:
                for part_hole in hole_matching_info:
                    if new_part_id in part_hole:
                        new_part_used_hole.append(part_hole[1])

            if self.opt.mission1: ############# STEFAN Tuning ##################
                if not check_pass:
                    new_part_holeInfo = part_holeInfo_dict[new_part_id]
                    new_part_holeId_list = [x[0] for x in new_part_holeInfo]
                    new_part_holenames = part_holename_dict[new_part_id]
                    if new_part_id == "part7" or new_part_id == "part8":
                        new_part_used_hole = [new_part_holenames[i-1] for i in new_part_holeId_list]
                    else:
                        new_part_used_hole = [new_part_id+'_1-'+new_part_holenames[i-1] for i in new_part_holeId_list]

            new_parts_info_list.append([new_part_id, new_part_RT_class, new_part_used_hole])
            connectivity = mid_name + '#' + new_part_id

        hole_pairs_list = list()
        for hole_matching_info in hole_matching_info_list_rename:
            part_hole1 = hole_matching_info[0]
            part_hole2 = hole_matching_info[1]
            hole1 = part_hole1[1]
            hole2 = part_hole2[1]
            hole_pairs_list.append(hole1+"#"+hole2)
        self.hole_pairs[step_num] = hole_pairs_list
        step_info = mid_info_list + new_parts_info_list + [connectivity]
        self.parts_info[step_num] = step_info
        print(step_info)

    def point_matching(self, step_num, connector_num, fastenerInfo_list, part_holeInfo_dict, part_holename_dict, step_parts_info, cut_image=None):
        fastener_connectingInfo = {}
        for fastenerInfo in fastenerInfo_list:
            fastener_id = fastenerInfo[0]
            part_id_list = sorted(part_holeInfo_dict.keys())

            dist_list = list()
            distInfo_list = list()

            for part in part_id_list:
                part_holeInfo = sorted(part_holeInfo_dict[part])
                for part_hole in part_holeInfo:
                    part_hole_id = part_hole[0]
                    dist = self.calculate_dist_point(fastenerInfo, part_hole)
                    distInfo = [dist, (part, part_hole_id)]
                    dist_list.append(dist)
                    distInfo_list.append(distInfo)

            min_idx = dist_list.index(min(dist_list))
            min_distInfo = distInfo_list[min_idx]
            assert min(dist_list) == min_distInfo[0]
            fastener_connectingInfo[fastener_id] = min_distInfo.copy()

        hole_matching_info_list = list()
        sub_final_dist_list = list()

        for fastener, connectingInfo in fastener_connectingInfo.items():
            sub_final_dist_list.append(connectingInfo[0])

        dist_list = sorted(sub_final_dist_list)
        fastener_idxs = [sub_final_dist_list.index(i) for i in dist_list]

        fastener_id_list = list()
        for fastener_idx in fastener_idxs:
            addition = True
            hole_matching_info_raw = fastener_connectingInfo[fastener_idx]

            part_hole = hole_matching_info_raw[1]
            if len(hole_matching_info_list) != 0:
                for holes in hole_matching_info_list:
                    for hole in holes:
                        if part_hole == hole:
                            addition = False
            if addition:
                hole_matching_info = [hole_matching_info_raw[1]]
                hole_matching_info_list.append(hole_matching_info.copy())
                fastener_id_list.append(fastener_idx)

        hole_matching_info_list = hole_matching_info_list[:connector_num]
        connected_fastener_list = fastener_id_list[:connector_num]

        # ######## Visualization ###########
        inv_img = cut_image.copy()
        for idx in range(len(connected_fastener_list)):
            connected_fastener = connected_fastener_list[idx]
            connected_fastenerInfo = [x for x in fastenerInfo_list if connected_fastener in x][0]
            hole_matching_info = hole_matching_info_list[idx]
            fastener_coord = connected_fastenerInfo[2]

            self.cut_image = cv2.circle(self.cut_image, fastener_coord, 4, (0,255,0), -1)

            hole_coord_list = list()
            part_hole = hole_matching_info[0]
            part_id = part_hole[0]
            hole_id = part_hole[1]
            hole_info = [x for x in part_holeInfo_dict[part_id] if hole_id in x][0]
            h_x = hole_info[1]
            h_y = hole_info[2]
            hole_coord = (h_x, h_y)
            self.cut_image = cv2.circle(self.cut_image, hole_coord, 4, (0,0,255), -1)
            self.cut_image = cv2.line(self.cut_image, fastener_coord, hole_coord, (255,0,0), 1)
        cv2.imwrite(os.path.join(self.v_dir, str(self.step_num)+'_check_connecting.png'), self.cut_image)


        ###################################
        connectivity = ''
        self.hole_pairs[step_num] = []

        if len(self.mid_id_list) != 0:
            mid_info_list = list()
            mid_used_hole = list()
            mid_name = 'step' + str(step_num-1)
            mid_RT_class = self.closest_gt_RT_index(self.mid_RT)
            for mid_id in self.mid_id_list:
                mid_id_hole_list = part_holename_dict[mid_id]
                mid_id_hole_idxs = [x[0][1] for x in hole_matching_info_list if mid_id in x[0]]
                if len(mid_id_hole_idxs) != 0:
                    for mid_id_hole_idx in mid_id_hole_idxs:
                        if mid_id == 'part7':
                            if mid_id_hole_idx < 7:
                                mid_used_hole += [x for x in mid_id_hole_list if (mid_id_hole_idx == int(x.split('_')[-1])) and ('C122620' not in x)]
                            else:
                                mid_id_hole_idx -= 7
                                mid_used_hole += [[x for x in mid_id_hole_list if ('C122620' in x)][mid_id_hole_idx]]

                        elif mid_id == 'part8':
                            if mid_id_hole_idx < 7:
                                mid_used_hole += [x for x in mid_id_hole_list if (mid_id_hole_idx == int(x.split('_')[-1])) and ('C122620' not in x)]
                            else:
                                mid_id_hole_idx -= 7
                                mid_used_hole += [[x for x in mid_id_hole_list if ('C122620' in x)][mid_id_hole_idx]]
                        else:
                            mid_used_hole += [x for x in mid_id_hole_list if (mid_id_hole_idx == int(x.split('_')[-1]))]
                else:
                    continue


            mid_info_list.append([mid_name, mid_RT_class, mid_used_hole])
            step_info = mid_info_list + [connectivity]

        if len(self.new_id_list) != 0:
            new_parts_info_list = list()
            new_parts_info = [x for x in sorted(step_parts_info, key=lambda x:x[0]) if x[0] not in self.mid_id_list]
            for new_part_info in new_parts_info:
                new_part_used_hole = list()
                new_part_id = new_part_info[0]
                new_part_RT = new_part_info[1]
                new_part_RT_class = self.closest_gt_RT_index(new_part_RT)
                new_part_hole_list = part_holename_dict[new_part_id]
                new_part_hole_idx = [x[0][1] for x in hole_matching_info_list if new_part_id in x[0]]
                new_part_used_hole = [new_part_id + '_1-' + new_part_hole_list[i-1] for i in new_part_hole_idx]
                new_parts_info_list.append([new_part_id, new_part_RT_class, new_part_used_hole].copy())
            step_info = new_parts_info_list + [connectivity]

        self.parts_info[step_num] = step_info
        print(step_info)

    def closest_gt_RT_index(self, RT_pred):
            return np.argmin([np.linalg.norm(RT - RT_pred) for RT in self.RTs_dict])
