import os, glob
import sys
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
from calculate_dist import *

def hole_pair_matching(self, step_num, connector_num, fastenerInfo_list, part_holeInfo_dict, part_holename_dict, step_parts_info, cut_image=None):
    # 각 체결선 별로 hole 까지의 거리 계산해서 체결선 별로 정보 저장
    
    fastenerInfo_list_original = fastenerInfo_list.copy()
    part_holeInfo_dict_original = part_holeInfo_dict.copy()
    fastener_connectingInfo_total = {}
    impossible_matching = {}
    for i in range(len(fastenerInfo_list_original)):
        impossible_matching[i] = list()
    intersection = True
    while intersection:
        intersection = False
        fastener_id_list = list()
        hole_matching_info_list = list()
        dist_list_fail_check = list()
        fastenerInfo_list = fastenerInfo_list_original.copy()
        part_holeInfo_dict = part_holeInfo_dict_original.copy()
        count = 0
        while count < connector_num:
            fastener_connectingInfo = {}
            for fastenerInfo in fastenerInfo_list:
                fastener_id = fastenerInfo[0]
                part_id_list = sorted(part_holeInfo_dict.keys())
                dist_list = list()
                distInfo_list = list()
                for p1 in part_id_list:
                    for p2 in part_id_list:
                        if (p1 in self.mid_id_list) and (p2 in self.mid_id_list):
                            continue 
                        if p1 < p2:
                            p1_holeInfo = sorted(part_holeInfo_dict[p1])
                            p2_holeInfo = sorted(part_holeInfo_dict[p2])
                            for p1_hole in p1_holeInfo:
                                for p2_hole in p2_holeInfo:
                                    p1_hole_id = p1_hole[0]
                                    p2_hole_id = p2_hole[0]
                                    dist = calculate_dist_pair(fastenerInfo, p1_hole, p2_hole)
                                    distInfo = [dist, (p1, p1_hole_id), (p2, p2_hole_id)]
                                    matchedInfo = [(p1, p1_hole_id),(p2, p2_hole_id)]
                                    matchedInfo = sorted(matchedInfo,key=lambda x:x[0])
                                    if matchedInfo in impossible_matching[fastener_id]:
                                        continue
                                    dist_list.append(dist)
                                    distInfo_list.append(distInfo)
                min_idx = dist_list.index(min(dist_list))
                min_distInfo = distInfo_list[min_idx]
                assert min(dist_list) == min_distInfo[0]
                fastener_connectingInfo[fastener_id] = min_distInfo.copy()
                if count == 0:
                    fastener_connectingInfo_total[fastener_id] = min_distInfo.copy()
            sub_final_dist_list = list()
            fastener_idxs_sub = list()
            for fastener, connectingInfo in fastener_connectingInfo.items():
                sub_final_dist_list.append(connectingInfo[0])
                fastener_idxs_sub.append(fastener)
            try:
                min_dist = sorted(sub_final_dist_list)[0]
                dist_list_fail_check.append(min_dist)
            except IndexError:
                break
            fastener_idx = fastener_idxs_sub[sub_final_dist_list.index(min_dist)]

            # 체결선을 바탕으로 어떤 hole끼리 매칭되었는지 저장하는데, 한번 사용된 hole은 다시 사용될 수 없음.
            hole_matching_info_raw = fastener_connectingInfo[fastener_idx]
            part_hole_1 = hole_matching_info_raw[1]
            part_hole_2 = hole_matching_info_raw[2]
            hole_matching_info = [hole_matching_info_raw[1],hole_matching_info_raw[2]]
            hole_matching_info_list.append(hole_matching_info.copy())
            fastenerInfo_list = [x for x in fastenerInfo_list if fastener_idx != x[0]]

            part1_hole_list = part_holeInfo_dict[part_hole_1[0]].copy()
            fastener_id_list.append(fastener_idx)
            part1_hole_list = [x for x in part1_hole_list if part_hole_1[1] != x[0]]
            part_holeInfo_dict[part_hole_1[0]] = part1_hole_list.copy()

            part2_hole_list = part_holeInfo_dict[part_hole_2[0]].copy()
            part2_hole_list = [x for x in part2_hole_list if part_hole_2[1] != x[0]]
            part_holeInfo_dict[part_hole_2[0]] = part2_hole_list.copy()

            count += 1

        # 직선 사이에 교점이 있는지 체크: x좌표의 차이의 곱이 음수면 교점 발생
        matched_holes_coord = list()
        for m in hole_matching_info_list:
            coord_sub = list()
            for h in m:
                h_coord = [(x[1],x[2]) for x in part_holeInfo_dict_original[h[0]] if h[1] == x[0]][0]
                coord_sub.append(h_coord)
            matched_holes_coord.append(coord_sub.copy())
            h_coord = sorted(matched_holes_coord, key=lambda x:x[1])
        for i in range(len(matched_holes_coord)):
            for j in range(len(matched_holes_coord)):
                if i < j:
                    matched_holes_1 = matched_holes_coord[i]
                    matched_holes_2 = matched_holes_coord[j]
                    mh1_up_x = matched_holes_1[0][0]
                    mh1_down_x = matched_holes_1[1][0]
                    mh2_up_x = matched_holes_2[0][0]
                    mh2_down_x = matched_holes_2[1][0]
                    if (mh1_up_x - mh2_up_x)*(mh1_down_x-mh2_down_x) < 0:
                        intersection = True
                        fastener_id1 = fastener_id_list[i]
                        fastener_id2 = fastener_id_list[j]
                        impossible_f1_list = impossible_matching[fastener_id1].copy()
                        impossible_f2_list = impossible_matching[fastener_id2].copy()
                        impossible_f1_list.append(sorted(hole_matching_info_list[i].copy(),key=lambda x:x[0]))
                        impossible_f2_list.append(sorted(hole_matching_info_list[j].copy(),key=lambda x:x[0]))
                        impossible_matching[fastener_id1] = impossible_f1_list.copy()
                        impossible_matching[fastener_id2] = impossible_f2_list.copy()

    # point matching again setting
    recomb = list()
    for idx in fastener_id_list:
        matchingInfo = fastener_connectingInfo_total[idx]

        h1 = matchingInfo[1]
        h1_info = part_holeInfo_dict_original[h1[0]]
        h1_info = [x for x in h1_info if h1[1] == x[0]][0]
        h1_coord = (h1_info[1], h1_info[2])

        h2 = matchingInfo[2]
        h2_info = part_holeInfo_dict_original[h2[0]]
        h2_info = [x for x in h2_info if h2[1] == x[0]][0]
        h2_coord = (h2_info[1], h2_info[2])
        h_coord_zip = [h1_coord, h2_coord]

        f_info = fastenerInfo_list_original[idx]
        f_coord_zip = [f_info[1], f_info[2]]
        recomb.append([f_coord_zip, h_coord_zip])
    back_to_pm, idx_list = change_to_point_matching(recomb)
    if back_to_pm > 0:
        self.point_matching_again_num = back_to_pm
        hole_matching_info_list = hole_matching_info_list[:-back_to_pm]
        fastener_id_list = fastener_id_list[:-back_to_pm]
    # is_fail set
    if len(dist_list_fail_check) != 0:
        max_dist = max(dist_list_fail_check)
        if max_dist > 800:
            self.is_fail = True

    ######## Visualization ###########
    inv_img = cut_image.copy()
    for idx in fastener_id_list:
        connected_fastenerInfo = [x for x in fastenerInfo_list_original if idx in x][0]
        hole_matching_info = hole_matching_info_list[fastener_id_list.index(idx)]

        fastener_coord1 = connected_fastenerInfo[1]
        fastener_coord2 = connected_fastenerInfo[2]
        fastener_coord_list = [fastener_coord1, fastener_coord2]

        self.cut_image = cv2.line(self.cut_image, fastener_coord1, fastener_coord2, (0,255,0), 2)

        hole_coord_list = list()
        for part_hole in hole_matching_info:
            part_id = part_hole[0]
            hole_id = part_hole[1]
            hole_info = [x for x in part_holeInfo_dict_original[part_id] if hole_id in x][0]
            h_x = hole_info[1]
            h_y = hole_info[2]
            hole_coord_list.append((h_x,h_y))
            self.cut_image = cv2.circle(self.cut_image, (h_x,h_y), 4, (0,0,255), -1)

        fastener_coord_list = sorted(fastener_coord_list, key=lambda x:x[1])
        hole_coord_list = sorted(hole_coord_list, key=lambda x:x[1])
        for f_coord, h_coord in zip(fastener_coord_list, hole_coord_list):
            self.cut_image = cv2.line(self.cut_image, f_coord, h_coord, (255,0,0), 1)
    cv2.imwrite(os.path.join(self.v_dir, str(self.step_num)+'_check_connecting.png'), self.cut_image)

    ##### json에서 읽은 그대로 상태인 hole 이름을, parts_info로 넘겨주기 위한 형태로 변환 #####
    hole_matching_info_list_rename = list()
    for hole_matching_info in hole_matching_info_list:
        part_hole_raname_list = list()
        for hole in hole_matching_info:
            part_holename_list = part_holename_dict[hole[0]]
            part_name = hole[0]
            hole_idx = hole[1]
            if part_name == 'part7' or part_name == 'part8' or part_name == 'part7_1' or part_name == 'part8_1':
                if hole_idx < 7:
                    part_hole_rename = (part_name,[x for x in part_holename_list if (hole_idx == int(x.split('_')[-1])) and ('C122620' not in x)][0])
                else:
                    hole_idx -= 7
                    part_hole_rename = (part_name,[x for x in part_holename_list if 'C122620' in x][hole_idx])
            elif part_name in self.mid_id_list:
                part_hole_rename = (part_name,[x for x in part_holename_list if hole_idx == int(x.split('_')[-1])][0])
            else:
                if '_' not in part_name:
                    part_hole_rename = (part_name,part_name+'_1-'+part_holename_list[hole_idx-1])
                else:
                    part_hole_rename = (part_name,part_name+'-'+part_holename_list[hole_idx-1])
            part_hole_raname_list.append(part_hole_rename)
        hole_matching_info_list_rename.append(part_hole_raname_list)

    # 위에서는 hole 이름만 변경했다면, 이제는 parts_info에 들어가는 정보 형태로 가공하는 과정
    connectivity = ['']
    mid_info_list = list()
    mid_used_hole = list()
    if len(self.mid_id_list) != 0:
        ##### Mid info modified #####
        mid_name = 'step' + str(step_num-1)
        mid_RT_class = self.closest_gt_RT_index(self.mid_RT)
        for mid_id in self.mid_id_list:
            for hole_matching_info in hole_matching_info_list_rename:
                for part_hole in hole_matching_info:
                    if mid_id in part_hole:
                        mid_used_hole.append(part_hole[1])
        mid_info_list.append([mid_name, mid_RT_class, mid_used_hole])
    if len(self.new_id_list) != 0:
        ##### New info modified #####
        new_parts_info_list = list()
        new_parts_info = [x for x in sorted(step_parts_info, key=lambda x:x[0]) if x[0] not in self.mid_id_list]
        for new_part_info in new_parts_info:
            new_part_used_hole = list()
            new_part_id = new_part_info[0]
            new_part_RT = new_part_info[1]
            new_part_RT_class = self.closest_gt_RT_index(new_part_RT)
            for hole_matching_info in hole_matching_info_list_rename:
                for part_hole in hole_matching_info:
                    if new_part_id in part_hole:
                        new_part_used_hole.append(part_hole[1])

            new_parts_info_list.append([new_part_id, new_part_RT_class, new_part_used_hole])
    step_info_sub = mid_info_list + new_parts_info_list
    # Connectivity
    connectivity_candidate = list()
    for hole_matching in hole_matching_info_list:
        matched_parts = sorted([x[0] for x in hole_matching])
        if matched_parts not in connectivity_candidate:
            connectivity_candidate.append(matched_parts)
    if len(mid_info_list) == 0:
        connectivity = [x[0] +'#'+ x[1] for x in connectivity_candidate]
    else:
        connectivity = list()
        for candidate in connectivity_candidate:
            connectivity_temp = list()
            for part in candidate:
                if part in self.mid_id_list:
                    connectivity_temp.append(mid_name)
                else:
                    connectivity_temp.append(part)
            connectivity_label = connectivity_temp[0]+'#'+connectivity_temp[1]
        connectivity.append(connectivity_label)
    hole_pairs_list = list()
    for hole_matching_info in hole_matching_info_list_rename:
        part_hole1 = hole_matching_info[0]
        part_hole2 = hole_matching_info[1]
        hole1 = part_hole1[1]
        hole2 = part_hole2[1]
        hole_pairs_list.append(hole1+"#"+hole2)
    step_info = step_info_sub + [connectivity]
    return step_info, hole_pairs_list

def point_matching(self, step_num, connector_num, fastenerInfo_list, part_holeInfo_dict, part_holename_dict, step_parts_info, cut_image=None):
    for part,holes in part_holeInfo_dict.items():
        part_holeInfo_dict[part] = sorted(holes,key=lambda x:x[0]).copy()
    fastener_connectingInfo = {}
    for fastenerInfo in fastenerInfo_list:
        fastener_id = fastenerInfo[0]
        part_id_list = sorted(part_holeInfo_dict.keys())
        dist_list = list()
        distInfo_list = list()
        dist_list = list()
        distInfo_list = list()

        for part in part_id_list:
            part_holeInfo = sorted(part_holeInfo_dict[part])
            for part_hole in part_holeInfo:
                part_hole_id = part_hole[0]
                dist = calculate_dist_point(fastenerInfo, part_hole)
                distInfo = [dist, (part, part_hole_id)]
                dist_list.append(dist)
                distInfo_list.append(distInfo)
        min_idx = dist_list.index(min(dist_list))
        min_distInfo = distInfo_list[min_idx]
        assert min(dist_list) == min_distInfo[0]
        fastener_connectingInfo[fastener_id] = min_distInfo.copy()
    hole_matching_info_list = list()
    sub_final_dist_list = list()
    fastener_idxs_sub = list()
    for fastener, connectingInfo in fastener_connectingInfo.items():
        sub_final_dist_list.append(connectingInfo[0])
        fastener_idxs_sub.append(fastener)
    dist_list = sorted(sub_final_dist_list)
    fastener_idxs = list()
    fastener_id_list = list()
    fastener_idxs = [sub_final_dist_list.index(i) for i in dist_list if i not in fastener_idxs]
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
    
    # 연결자 수가 체결선 수보다 같거나 적으면 연결자 수를 사용하고, 그 반대이면 체결선 수를 사용 (IndexError 해결)
    if connector_num <= len(fastener_idxs):
        hole_matching_info_list = hole_matching_info_list[:connector_num]
        connected_fastener_list = fastener_id_list[:connector_num]
    else:
        hole_matching_info_list = hole_matching_info_list
        connected_fastener_list = fastener_id_list
    
    ######### Visualization ###########
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
    
    connectivity = ['']
    # 매칭된 부품이 mid인지 new인지 판별
    matched_parts = [x[0][0] for x in hole_matching_info_list]
    mid_matched = False
    new_matched = False
    for mp in matched_parts:
        if mp in self.mid_id_list:
            mid_matched = True
        elif mp in self.new_id_list:
            new_matched = True
    if len(self.mid_id_list) != 0:
        if mid_matched:
            mid_info_list = list()
            mid_used_hole = list()
            mid_name = 'step' + str(step_num-1)
            mid_RT_class = self.closest_gt_RT_index(self.mid_RT)
            for mid_id in self.mid_id_list:
                mid_id_hole_list = part_holename_dict[mid_id]
                mid_id_hole_idxs = [x[0][1] for x in hole_matching_info_list if mid_id in x[0]]
                if len(mid_id_hole_idxs) != 0:
                    for mid_id_hole_idx in mid_id_hole_idxs:
                        if mid_id == 'part7' or mid_id == 'part8' or mid_id == 'part7_1' or mid_id == 'part8_1':
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
        if new_matched:
            new_parts_info_list = list()
            new_parts_info = [x for x in sorted(step_parts_info, key=lambda x:x[0]) if x[0] not in self.mid_id_list]
            for new_part_info in new_parts_info:
                new_part_used_hole = list()
                new_part_id = new_part_info[0]
                new_part_RT = new_part_info[1]
                new_part_RT_class = self.closest_gt_RT_index(new_part_RT)
                new_part_hole_list = part_holename_dict[new_part_id]
                new_part_hole_idxs = [x[0][1] for x in hole_matching_info_list if new_part_id in x[0]]
                if new_part_id == 'part7' or new_part_id == 'part8' or new_part_id == 'part7_1' or new_part_id == 'part8_1':
                    for new_part_hole_idx in new_part_hole_idxs:
                        if new_part_hole_idx < 7:
                            new_part_used_hole += [x for x in new_part_hole_list if (new_part_hole_idx == int(x.split('_')[-1])) and ('C122620' not in x)]
                        else:
                            new_part_hole_idx -= 7
                            new_part_used_hole += [[x for x in new_part_hole_list if ('C122620' in x)][new_part_hole_idx]]
                else:
                    if "_" not in new_part_id:
                        new_part_used_hole = [new_part_id + '_1-' + new_part_hole_list[i-1] for i in new_part_hole_idxs]
                    else:
                        new_part_used_hole = [new_part_id + '-' + new_part_hole_list[i-1] for i in new_part_hole_idxs]
                if len(new_part_used_hole) == 0:
                    continue
                new_parts_info_list.append([new_part_id, new_part_RT_class, new_part_used_hole].copy())
            step_info = new_parts_info_list + [connectivity]
    return step_info
