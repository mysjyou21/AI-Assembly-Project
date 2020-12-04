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
        mid_id_list, K, mid_RT, RTs_dict, hole_pairs, component_list):
        cut_image = step_images[step_num]
        H = cut_image.shape[0]
        W = cut_image.shape[1]
        # Component except parts
        # connector = self.connectors_serial_OCR[step_num][0][0]
        # connector_num = int(self.connectors_mult_OCR[step_num][0][0])
        connector = "101350"
        connector_num = 2

        #######
        # component like 'circle, rectangle' should be added in "A List"... Need to be erased from cut image
        #######

        # part ID, Pose, KRT
        step_parts_info = parts_info[step_num]
        K = K
        part_id_list = [x[0] for x in step_parts_info]
        part_RT_list = [x[1] for x in step_parts_info]

        fastenerInfo_list = self.fastener_loader(cut_image, component_list)
        part_holeInfo_dict = self.part_hole_connector_loader(step_num, part_id_list, part_RT_list, K, H, W)

        # Just activate holes which connected by connectors
        for part_id, part_holeInfo in part_holeInfo_dict.items():
            part_holeInfo_temp_ = part_holeInfo_dict[part_id].copy()
            part_holeInfo_temp = [x for x in part_holeInfo_temp_ if connector in x]
            part_holeInfo_dict[part_id] = part_holeInfo_temp.copy()

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
                                dist = self.calculate_dist(fastenerInfo, p1_hole, p2_hole)
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
            hole_matching_info_rename = list()
            for hole in hole_matching_info:
                hole_rename = hole[0] + "_1-" + "hole_" + str(hole[1])
                hole_matching_info_rename.append(hole_rename)
            hole_matching_info_list_rename.append(hole_matching_info_rename)

        ##### Regardless Connectivity, make list consisting of holes #####
        used_part_hole = [x[0] for x in hole_matching_info_list] + [x[1] for x in hole_matching_info_list]
        used_part_hole_rename = [x[0] for x in hole_matching_info_list_rename] + [x[1] for x in hole_matching_info_list_rename]

        def closest_gt_RT_index(RT_pred):
            return np.argmin([np.linalg.norm(RT - RT_pred) for RT in RTs_dict])

        if mid_id_list:
            mid_info_list = list()
            mid_used_hole = list()
            mid_name = 'step' + str(step_num-1)
            mid_RT_class = closest_gt_RT_index(mid_RT)
            for mid_id in mid_id_list:
                mid_used_hole += [x for x in used_part_hole_rename if mid_id in x]
            mid_info_list.append([mid_name, mid_RT_class, mid_used_hole])

            new_parts_info_list = list()
            new_part_used_hole = list()
            connectivity = ''
            new_parts_info = [x for x in sorted(step_parts_info) if x[0] not in mid_id_list]
            if len(new_parts_info)!=0:
                for new_part_info in new_parts_info:
                    new_part_id = new_part_info[0]
                    new_part_RT = new_part_info[1]
                    new_part_RT_class = closest_gt_RT_index(new_part_RT)
                    for hole_matching_info in hole_matching_info_rename:
                        new_part_used_hole += [int(x[-1]) for x in hole_matching_info if new_part_id in x]
                    new_part_used_hole = [x[1] for x in used_part_hole if x[0] == new_part_id]
                    new_parts_info_list.append([new_part_id, new_part_RT_class, new_part_used_hole])

                connectivity = mid_name + '#' + new_part_id
            hole_pairs[step_num] = [[x[0]+"#"+x[1] for x in hole_matching_info_list_rename]]
            step_info = mid_info_list + new_parts_info_list + [connectivity]
            parts_info[step_num] = step_info
            print("=================modified parts Info==================")
            print(step_info)
            print("======================================================")

        return parts_info, hole_pairs

    def fastener_loader(self, cut_image, component_erase_list):
        if len(cut_image.shape) == 3:
            gray = cv2.cvtColor(cut_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = cut_image.copy()
        inv_ = 255 - gray
        inv = np.where(inv_>254,255,0)/255 #binarization

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
        kernel_cc = [[1,1,1],[1,1,1],[1,1,1]]
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

            # Visualize
            region = labeled_fastener[y_start:y_end, x_start:x_end]
            region = ((region==candidates_fastener[i])*255).astype(np.uint8)
            result[y_start:y_end, x_start:x_end] += region

        return fastenerInfo_list

    def part_hole_connector_loader(self, step_num, part_id_list, part_RT_list, K, H, W, debug_mode=False, debug_img=None):
        part_holeInfo_dict = {}
        for part_idx, part_id in enumerate(part_id_list):
            if part_id == "part7":
                part_id = "part2"
                hole_XYZ = self.base_loader(part_id)
                part_id = "part7"
            elif part_id == "part8":
                part_id = "part3"
                hole_XYZ = self.base_loader(part_id)
                part_id = "part8"
            else:
                hole_XYZ = self.base_loader(part_id)

            RT = part_RT_list[part_idx]
            hole_x, hole_y = self.project_points(
                hole_XYZ, K, RT, H, W)

            hole_id = 0
            holeInfo_list = list()
            for xh, yh in zip(hole_x, hole_y):
                holeInfo = [hole_id, xh, yh]
                holeInfo_list.append(holeInfo.copy())
                hole_id += 1

            holeInfo_list =self.add_connector(int(part_id[-1]), holeInfo_list)

            part_holeInfo_dict[part_id] = holeInfo_list.copy()

            for hole_info in holeInfo_list:
                x_coord = hole_info[1]
                y_coord = hole_info[2]
        return part_holeInfo_dict

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

    def calculate_dist(self, fastenerInfo, p1_hole, p2_hole):
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

    def base_loader(self, part_name, scale=1):
        """ input: part_name = 'part3'
        output: hole_XYZ(np array), #norm_XYZ(np array),
                hole_dic={hole_name: [CenterX, CenterY, CenterZ]} #, NormalX, NormalY, NormalZ]}
                hole_name: 'part2_1-hole_3' """

        part_file = '%s/%s.json'%(self.opt.hole_path, part_name)
        part_data = []
        with open(part_file, 'r') as f:
            part_data = json.load(f)

        part_dic = part_data["data"]
        holes = part_dic["hole"]
        holename = [x for x in sorted(holes.keys())]
        holename = sorted(holename, key=lambda x:int(x.split('_')[1]))
        hole_XYZ = [[holes[k]["CenterX"], holes[k]["CenterY"], holes[k]["CenterZ"]] for k in holename]
        center_XYZ = [part_dic["CenterPointX"], part_dic["CenterPointY"], part_dic["CenterPointZ"]]

        center_XYZ = np.array(list(map(float, center_XYZ)))/scale
        hole_XYZ = [np.array(list(map(float, temp)))/scale for temp in hole_XYZ]

        hole_XYZ = [hole_XYZ[i] for i in range(len(hole_XYZ))]
        hole_XYZ = np.stack(hole_XYZ)

        return hole_XYZ


