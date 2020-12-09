import os
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage

class MSN2_hole_detector():
    def __init__(self, opt):
        self.opt = opt

    def main_hole_detector(self, step_num, step_images, parts_info, connectors, mults, \
        mid_id_list, K, mid_RT, RTs_dict, hole_pairs, component_list, find_mid=False):
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

        cut_image = step_images[step_num]
        H = cut_image.shape[0]
        W = cut_image.shape[1]

        def closest_gt_RT_index(RT_pred):
            return np.argmin([np.linalg.norm(RT - RT_pred) for RT in self.RTs_dict])
        
        # if self.connectors[step_num]!=0:
        #     step_connector = self.connectors[step_num][0]
        #     if len(step_connector) != 0:
        #         connector = step_connector[0]

        step_connector = self.connectors[step_num][0]
        if len(step_connector) != 0:
            connector = step_connector[0]
        else:
            step_name = 'step' + str(step_num-1)
            matched_pose = closest_gt_RT_index(mid_RT)
            parts_info[step_num] = [[step_name, matched_pose, []],'']
            hole_pairs[step_num] = []
            
            return parts_info, hole_pairs

        # if self.mults[step_num]!=0:
            # step_mult = self.mults[step_num][0]    
        step_mult = self.mults[step_num][0]
        if len(step_mult) != 0:
            connector_num = int(step_mult[0])
        else:
            connector_num = int(self.mults[1][0][0])

        # part ID, Pose, KRT
        step_parts_info = self.parts_info[step_num]
        self.K = K
        part_id_list = [x[0] for x in step_parts_info]
        part_RT_list = [x[1] for x in step_parts_info]

        self.new_id_list = [x for x in part_id_list if x not in self.mid_id_list]
        fastenerInfo_list = self.fastener_loader(cut_image, component_list)
        part_holename_dict, part_holeInfo_dict = self.part_hole_connector_loader(step_num, part_id_list, part_RT_list, K, H, W, True, cut_image)

        for part_id, part_holeInfo in part_holeInfo_dict.items():
            part_holeInfo_temp_ = part_holeInfo_dict[part_id].copy()
            part_holeInfo_temp = [x for x in part_holeInfo_temp_ if connector in x]
            part_holeInfo_dict[part_id] = part_holeInfo_temp.copy()

        if 'part7' in self.new_id_list and len(self.mid_id_list)==0:
            self.new_id_list.remove('part7')
            self.mid_id_list.append('part7')
        if 'part8' in self.new_id_list and len(self.mid_id_list)==0:
            self.new_id_list.remove('part8')
            self.mid_id_list.append('part8')

        if len(self.new_id_list)!=0 and len(self.mid_id_list)!=0:
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

            for fastener_idx in range(len(fastener_idxs)):
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

            hole_matching_info_list = hole_matching_info_list[:connector_num]

            ##### convert hole name to other format as same as step.json(part_#-hole_#) #####
            hole_matching_info_list_rename = list()
            for hole_matching_info in hole_matching_info_list:
                part_hole_raname_list = list()
                for hole in hole_matching_info:
                    part_holename_list = part_holename_dict[hole[0]]
                    part_holename_idx = hole[1]
                    part_hole_rename = (hole[0],part_holename_list[part_holename_idx-1])
                    part_hole_raname_list.append(part_hole_rename)
                hole_matching_info_list_rename.append(part_hole_raname_list)

            ##### Regardless Connectivity, make list consisting of holes #####
            # used_part_hole = [x[0] for x in hole_matching_info_list] + [x[1] for x in hole_matching_info_list]
            # used_part_hole_rename = [x[0] for x in hole_matching_info_list_rename] + [x[1] for x in hole_matching_info_list_rename]

            ##### Mid info modified #####
            mid_info_list = list()
            mid_used_hole = list()
            if len(self.mid_id_list) == 1 and 'part7' in self.mid_id_list:
                mid_name = 'part7'
            elif len(self.mid_id_list) == 1 and 'part8' in self.mid_id_list:
                mid_name = 'part8'
            else:
                mid_name = 'step' + str(step_num-1)
            mid_RT_class = closest_gt_RT_index(self.mid_RT)
            for mid_id in self.mid_id_list:
                for hole_matching_info in hole_matching_info_list_rename:
                    for part_hole in hole_matching_info:
                        if mid_id in part_hole:
                            mid_used_hole.append(part_hole[1])
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
                new_part_RT_class = closest_gt_RT_index(new_part_RT)
                for hole_matching_info in hole_matching_info_list_rename:
                    for part_hole in hole_matching_info:
                        if new_part_id in part_hole:
                            new_part_used_hole.append(part_hole[1])
                new_parts_info_list.append([new_part_id, new_part_RT_class, new_part_used_hole])
                connectivity = mid_name + '#' + new_part_id
                # connectivity_temp = mid_name + '#' + new_part_id
                # if len(self.new_id_list) > 1:
                #     connectivity.append(connectivity_temp)
                # else:
                #     connectivity = connectivity_temp

            hole_pairs_list = list()
            for hole_matching_info in hole_matching_info_list_rename:
                part_hole1 = hole_matching_info[0]
                part_hole2 = hole_matching_info[1]
                hole1 = part_hole1[0] + '_1-' + part_hole1[1]
                hole2 = part_hole2[0] + '_1-' + part_hole2[1]
                hole_pairs_list.append(hole1+"#"+hole2)
            self.hole_pairs[step_num] = hole_pairs_list
            step_info = mid_info_list + new_parts_info_list + [connectivity]
            self.parts_info[step_num] = step_info
            print("=================modified parts Info==================")
            print(step_info)
            print("======================================================")
            # print(self.parts_info[step_num])
            # print(self.hole_pairs[step_num])

        else:
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

            for fastener_idx in range(len(fastener_idxs)):
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
            hole_matching_info_list = hole_matching_info_list[:connector_num]
            # hole_matching_info_list_rename = list()
            # for hole in hole_matching_info_list:
            #     hole_rename = hole[0][0] + "_1-" + "hole_" + str(hole[0][1])
            #     hole_matching_info_list_rename.append(hole_rename)

            connectivity = ''
            self.hole_pairs[step_num] = []

            if len(self.mid_id_list) != 0:
                mid_info_list = list()
                mid_used_hole = list()
                mid_name = 'step' + str(step_num-1)
                mid_RT_class = closest_gt_RT_index(self.mid_RT)
                for mid_id in self.mid_id_list:
                    mid_id_hole_list = part_holename_dict[mid_id]
                    mid_id_hole_idx = [x[0][1] for x in hole_matching_info_list if mid_id in x[0]]
                    mid_used_hole = [mid_id_hole_list[i-1] for i in mid_id_hole_idx]
                    # mid_used_hole += [x for x in hole_matching_info_list_rename if mid_id in x]
                mid_info_list.append([mid_name, mid_RT_class, mid_used_hole])
                step_info = mid_info_list + [connectivity]

            if len(self.new_id_list) != 0:
                new_parts_info_list = list()
                new_parts_info = [x for x in sorted(step_parts_info, key=lambda x:x[0]) if x[0] not in self.mid_id_list]
                for new_part_info in new_parts_info:
                    new_part_used_hole = list()
                    new_part_id = new_part_info[0]
                    new_part_RT = new_part_info[1]
                    new_part_RT_class = closest_gt_RT_index(new_part_RT)
                    new_part_hole_list = part_holename_dict[new_part_id]
                    new_part_hole_idx = [x[0][1] for x in hole_matching_info_list if new_part_id in x[0]]
                    new_part_used_hole = [new_part_hole_list[i-1] for i in new_part_hole_idx]
                    new_parts_info_list.append([new_part_id, new_part_RT_class, new_part_used_hole].copy())
                step_info = new_parts_info_list + [connectivity]
                    
            self.parts_info[step_num] = step_info
            print("=================modified parts Info==================")
            print(step_info)
            print("======================================================")

        return self.parts_info, self.hole_pairs

    def fastener_loader(self, cut_image, component_list):
        if len(cut_image.shape) == 3:
            gray = cv2.cvtColor(cut_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = cut_image.copy()
        inv_ = 255 - gray
        inv = np.where(inv_>254,255,0)/255 #binarization

        for component in component_list:
            x = component[0]
            y = component[1]
            w = component[2]
            h = component[3]
            inv[y:y+h,x:x+w] = 0

        thick_vertical_eraser = np.ones((1,3))
        horizon_eraser = np.ones((3,1))

        thick_vertial_erase_sub = cv2.erode(inv,thick_vertical_eraser,iterations=2)
        thick_vertial_erase = cv2.dilate(thick_vertial_erase_sub,thick_vertical_eraser,iterations=2)
        thin_vertical_lines_sub = inv - thick_vertial_erase
        thin_vertical_lines = np.clip(thin_vertical_lines_sub,0,1)

        horizon_erase_sub = cv2.erode(thin_vertical_lines,horizon_eraser,iterations=10)
        horizon_erase = cv2.dilate(horizon_erase_sub,horizon_eraser,iterations=10)
        vertical_lines = horizon_erase.copy()

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

            if h_obj <= 30 and i in candidates_fastener:
                candidates_fastener.remove(i)
            if h_obj >= 150 and i in candidates_fastener:
                candidates_fastener.remove(i)

        fastenerInfo_list = list()
        result = np.zeros(sub_comb.shape)
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

            ##### DEBUG #####
            region = labeled_fastener[y_start:y_end, x_start:x_end]
            region = ((region==candidates_fastener[i])*255).astype(np.uint8)
            result[y_start:y_end, x_start:x_end] += region
        
        
        cv2.imwrite(os.path.join(self.v_dir, str(self.step_num) + '_fastener_test.png'), result)
        #################

        return fastenerInfo_list

    def part_hole_connector_loader(self, step_num, part_id_list, part_RT_list, K, H, W, debug_mode=False, debug_img=None):
        part_holeInfo_dict = {}
        part_holename_dict = {}
        step_name = 'step' + str(step_num-1)
        for part_idx, part_id in enumerate(part_id_list):
            if part_id in self.mid_id_list:
                holename, hole_id, hole_XYZ = self.mid_loader_part_hole(step_name, part_id, self.opt.hole_path, self.opt.cad_path, 100)
            else:
                holename, hole_id, hole_XYZ = self.base_loader(part_id, self.opt.hole_path, self.opt.cad_path, 100)

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

        ##### DEBUG #####
            for hole_info in holeInfo_list:
                x_coord = hole_info[1]
                y_coord = hole_info[2]
                if debug_mode:
                    debug_img = cv2.circle(debug_img, (x_coord,y_coord), 4, (0,0,255), -1)

            cv2.imwrite(os.path.join(self.v_dir, str(self.step_num) + '_hole_check.png'), debug_img)
        #################

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
        hole_num = len(holeInfo_list)
        if part_id == 1:
            for i in range(hole_num):
                holeInfo = holeInfo_list[i]
                holeInfo.append('122925')
                holeInfo_list[i] = holeInfo.copy()

        if part_id == 2 or part_id == 3:
            for i in range(hole_num):
                holeInfo = holeInfo_list[i]
                if i in [0,1,4,5]:
                    holeInfo.append('101350')
                elif i in [2,3]:
                    holeInfo.append('104322')
                else:
                    holeInfo.append('122620')
                holeInfo_list[i] = holeInfo.copy()

        if part_id == 4:
            for i in range(hole_num):
                holeInfo = holeInfo_list[i]
                if i in [0,1,2,3,6,7]:
                    holeInfo.append('101350')
                else:
                    holeInfo.append('104322')
                holeInfo_list[i] = holeInfo.copy()

        if part_id == 5 or part_id == 6:
            for i in range(hole_num):
                holeInfo = holeInfo_list[i]
                if i in [0,2,3,5,6,7,9]:
                    holeInfo.append('101350')
                else:
                    holeInfo.append('104322')
                holeInfo_list[i] = holeInfo.copy()

        if part_id == 7 or part_id == 8:
            for i in range(hole_num):
                holeInfo = holeInfo_list[i]
                if i in [0,1,4,5]:
                    holeInfo.append('101350')
                elif i in [2,3]:
                    holeInfo.append('104322')
                else:
                    holeInfo.append('')
                holeInfo_list[i] = holeInfo.copy()

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

    def base_loader(self, part_name, json_path, center_dir, scale=100):
        """ input: part_name = 'part3'
        output: hole_XYZ(np array), #norm_XYZ(np array),
                hole_dic={hole_name: [CenterX, CenterY, CenterZ]} #, NormalX, NormalY, NormalZ]}
                hole_name: 'part2_1-hole_3' """
        
        hole_id = [] ##### DEBUG #####
        if part_name == "part7":
            part_name = "step1_b"
        elif part_name == "part8":
            part_name = "step1_a"

        part_file = '%s/%s.json'%(json_path, part_name)
        part_data = []
        with open(part_file, 'r') as f:
            part_data = json.load(f)
        with open(os.path.join(center_dir, 'center.json'), 'r') as f:
            center_data = json.load(f)
        with open(os.path.join(center_dir, 'obj_min.json'), 'r') as f: # to correct global coordinates of our obj files
            min_data = json.load(f)

        part_dic = part_data["data"]
        holes = part_dic["hole"]
        holename = [x for x in sorted(holes.keys())]
        if part_name == 'step1_a' or part_name == 'step1_b':
            holename1 = sorted([x for x in holename if 'C122620' in x])
            holename2 = sorted([x for x in holename if 'C122620' not in x], key=lambda x: int(x.split('-')[1].split('_')[1]))
            holename = holename2 + holename1
            hole_id = list(range(1,len(holename)+1))
        else:
            holename = sorted(holename, key=lambda x:int(x.split('_')[1]))
            hole_id = list(range(1,len(holename)+1))
        hole_XYZ = [[holes[k]["CenterX"], holes[k]["CenterY"], holes[k]["CenterZ"]] for k in holename]
        if part_name in center_data.keys():
            center_XYZ = center_data[part_name]
        else:
            print('No center data, ', part_name)
            center_XYZ = [0,0,0]
    #    if "CenterPointX" in part_dic.keys():
    #        center_XYZ = [part_dic["CenterPointX"], part_dic["CenterPointY"], part_dic["CenterPointZ"]]
    #    else:
    #        center_XYZ = [0,0,0]
        if "MinPointX" in part_dic.keys() and 'step' not in part_name:
            min_XYZ = [part_dic["MinPointX"], part_dic["MinPointY"], part_dic["MinPointZ"]]
        else:
            min_XYZ = [0,0,0]
        center_XYZ = np.array(list(map(float, center_XYZ)))/scale
        min_XYZ = np.array(list(map(float, min_XYZ)))/scale
        if 'step' not in part_name:
            min_blender_XYZ = np.array(min_data[part_name])/scale
            min_XYZ = min_XYZ - min_blender_XYZ
        hole_XYZ = [np.array(list(map(float, temp)))/scale - center_XYZ - min_XYZ for temp in hole_XYZ]

        hole_XYZ = [hole_XYZ[i] for i in range(len(hole_XYZ))]
        hole_XYZ = np.stack(hole_XYZ)

        return holename, hole_id, hole_XYZ

    def mid_loader_part_hole(self, step_name, part_id, json_path, center_dir, scale=100):
        """ input: step_name = 'step3'
        output: hole_XYZ(np array),
                hole_dic={hole_name: [CenterX, CenterY, CenterZ]}
                hole_name: 'part2_1-hole_3' """
        hole_id = [] #### DEBUG #####
        part_file = '%s/%s.json'%(json_path, step_name)
        part_data = []
        with open(part_file, 'r') as f:
            part_data = json.load(f)
        with open(os.path.join(center_dir, 'center.json'), 'r') as f:
            center_data = json.load(f)

        part_dic = part_data["data"]
        holes = part_dic["hole"]
        if part_id == 'part7':
            holename = [x for x in sorted(holes.keys()) if ('part2' in x) or ('C122620_3' in x) or ('C122620_4' in x)]
            holename1 = sorted([x for x in holename if 'C122620' in x])
            holename2 = sorted([x for x in holename if 'C122620' not in x], key=lambda x: int(x.split('-')[1].split('_')[1]))
            holename = holename2 + holename1
            hole_id = list(range(1,len(holename)+1))
            hole_XYZ = [[holes[k]["CenterX"], holes[k]["CenterY"], holes[k]["CenterZ"]] for k in holename]
        elif part_id == 'part8':
            holename = [x for x in sorted(holes.keys()) if ('part3' in x) or ('C122620_1' in x) or ('C122620_2' in x)]
            holename1 = sorted([x for x in holename if 'C122620' in x])
            holename2 = sorted([x for x in holename if 'C122620' not in x], key=lambda x: int(x.split('-')[1].split('_')[1]))
            holename = holename2 + holename1
            hole_id = list(range(1,len(holename)+1))
            hole_XYZ = [[holes[k]["CenterX"], holes[k]["CenterY"], holes[k]["CenterZ"]] for k in holename]
        else:
            holename = [x for x in sorted(holes.keys()) if part_id in x]
            holename = [part_id + '_1-hole_' + str(i) for i in range(1,len(holename)+1)]
            hole_id = list(range(1,len(holename)+1))
            hole_XYZ = [[holes[k]["CenterX"], holes[k]["CenterY"], holes[k]["CenterZ"]] for k in holename]
        # id_list = list(set([hole.split('-')[0] for hole in holename]))
        # id_list = [x.replace('_1','') if 'C122620' not in x else 'part8' if (x=='C122620_1' or x=='C122620_2') else 'part7' for x in id_list]
        # id_list = list(set(id_list))

        # hole_XYZ = [[holes[k]["CenterX"], holes[k]["CenterY"], holes[k]["CenterZ"]] for k in holename]
        if step_name in center_data.keys():
            center_XYZ = center_data[step_name]
        else:
            print('No center data, ', step_name)
            center_XYZ = [0,0,0]
    #    if "CenterPointX" in part_dic.keys():
    #        center_XYZ = [part_dic["CenterPointX"], part_dic["CenterPointY"], part_dic["CenterPointZ"]]
    #    else:
    #        center_XYZ = [0,0,0]
    #    if "MinPointX" in part_dic.keys():
    #        min_XYZ = [part_dic["MinPointX"], part_dic["MinPointY"], part_dic["MinPointZ"]]
    #    else:
        min_XYZ = [0,0,0]

        center_XYZ = np.array(list(map(float, center_XYZ)))/scale
        min_XYZ = np.array(list(map(float, min_XYZ)))/scale
        hole_XYZ = [np.array(list(map(float, temp)))/scale - center_XYZ - min_XYZ for temp in hole_XYZ]

        hole_XYZ = [hole_XYZ[i] for i in range(len(hole_XYZ))]
        hole_XYZ = np.stack(hole_XYZ)
        # part_hole_idx = np.array([hole.split('-')[0] for hole in holename])
        # part_hole_idx = np.array([x.replace('_1','') if 'C122620' not in x else 'part8' if (x=='C122620_1' or x=='C122620_2') else 'part7' for x in part_hole_idx])

        # part_hole_dic = {}
        # for id in id_list:
        #     idx = np.where(part_hole_idx == id)[0]
        #     part_hole_XYZ = hole_XYZ[idx]
        #     part_hole_dic[id] = part_hole_XYZ

        # if 'part7' in id_list:
        #     assert 'part2' in id_list
        #     part_hole_dic['part7'] = np.concatenate([part_hole_dic['part7'], part_hole_dic['part2']])
        #     del part_hole_dic['part2']
        # if 'part8' in id_list:
        #     assert 'part3' in id_list
        #     part_hole_dic['part8'] = np.concatenate([part_hole_dic['part8'], part_hole_dic['part3']])
        #     del part_hole_dic['part3']

        return holename, hole_id, hole_XYZ