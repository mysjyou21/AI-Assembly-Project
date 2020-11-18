import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import color
import glob, os
import sys
import json
from copy import deepcopy
sys.path.append('./funtion/utilities')
from utilities.utils_hole import read_csv
from config import *
opt = init_args()

part_holes = read_csv(opt.cadinfo_path)

#hole_info_path = './function/utilities/hole.json'
#with open(hole_info_path, "r") as f:
#    part_holes_ = json.load(f)
# hole-connectivity
part_connectivity = []
for key in part_holes.keys():
    if '#' in key:
        part_connectivity.append(key)

hole_pose_path = './function/utilities/hole_pose.json'
with open(hole_pose_path, 'r') as f:
    pose_dic = json.load(f)

hole_loc_path = './function/utilities/hole_loc.json'
with open(hole_loc_path, 'r') as f:
    loc_dic = json.load(f)

hole_pose_part_path = './function/utilities/hole_pose_part.json'
with open(hole_pose_part_path, 'r') as f:
    pose_dic_part = json.load(f)

hole_loc_part_path = './function/utilities/hole_loc_part.json'
with open(hole_loc_part_path, 'r') as f:
    loc_dic_part = json.load(f)

with open('./function/utilities/hole_pair.json', 'r') as f:
    hole_pairs = json.load(f)

sort_key = {}
sort_key['F'] = ['y', True]
sort_key['B'] = ['y', False]
sort_key['R'] = ['x', True]
sort_key['L'] = ['x', False]

straight = {}
keys = part_holes.keys()
keys = [x for x in keys if '#' not in x]
true_keys = ['part2', 'part3', 'step1_a', 'step1_b']
for key in keys:
    straight[key] = True if key in true_keys else False

pt_origin = np.array([[882,78], [1053,78], [882,1004], [1053,1004]], np.float32)
pt_left = np.array([[871,93], [1039,114], [883,924], [1028,961]], np.float32)
pt_right = np.array([[897,112], [1064,90], [906,959], [1050,920]], np.float32)

M1 = cv2.getPerspectiveTransform(pt_left, pt_origin)
M2 = cv2.getPerspectiveTransform(pt_right, pt_origin)


def diagonal_se(r, direction='left'):
    kernel = np.zeros((2*r+1, 2*r+1), dtype=np.uint8)

    if direction=='left':
        for i in range(0, 2*r+1):
            kernel[i,i] = 1
    else: #'right'
        for i in range(0, 2*r+1):
            kernel[i,2*r-i] = 1

    return kernel

def segment_part_boundaries(bin, kernel_rec, kernel_horz3, kernel_cc, imgname, h_min_obj_th=100):
    fined5 = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel_rec)
    fined5 = cv2.dilate(fined5, kernel_horz3, iterations=1)
    # to group parts
    labeled_part, num = ndimage.label(fined5, kernel_cc)
#    temp = (color.label2rgb(labeled_part)*255).astype(np.uint8)
#    cv2.imwrite(os.path.join(OUTPUT_DIR, imgname.rstrip('.png')+'_partbound0.png'), temp)
    obj_part = ndimage.find_objects(labeled_part)
    candidates_part = list(range(1,num+1))
    bound_pose = []
#    h_min_obj_th = 50 #100
    for i in range(1, num+1):
        shape = obj_part[i-1]
        w_obj = shape[1].stop - shape[1].start
        h_obj = shape[0].stop - shape[0].start
        if h_obj < h_min_obj_th and i in candidates_part:
            candidates_part.remove(i)
        else:
            y1, y2 = shape[0].start, shape[0].stop
            x1, x2 = shape[1].start, shape[1].stop
            bound_pose.append([(y1,x1), (y2,x2)])

    same_group_index = {}
    same_group_index_by_example = {}

    same_group_th_h = 10
    same_group_th_w = 0
    for i in range(len(candidates_part)):
        idx = candidates_part[i]
        bp = bound_pose[i]
        temp1 = [j for j in range(i) if (bound_pose[j][0][0]-same_group_th_h<=bp[0][0] and bound_pose[j][0][1]-same_group_th_w<=bp[0][1] and bound_pose[j][1][0]+same_group_th_h>=bp[1][0] and bound_pose[j][1][1]+same_group_th_w>=bp[1][1])] # i is inside j
        temp2 = [j for j in range(i) if (bound_pose[j][0][0]+same_group_th_h>=bp[0][0] and bound_pose[j][0][1]+same_group_th_w>=bp[0][1] and bound_pose[j][1][0]-same_group_th_h<=bp[1][0] and bound_pose[j][1][1]-same_group_th_w<=bp[1][1])] # j is inside i

        if len(temp1) > 0:
            key = candidates_part[temp1[0]]
            if key not in same_group_index.keys():
                key = same_group_index_by_example[key]
            same_group_index[key].append(idx)
            same_group_index_by_example[idx] = key

        elif len(temp2) > 0:
            key = candidates_part[temp2[0]]
            if key not in same_group_index.keys():
                key = same_group_index_by_example[key]
            same_group_index[key].append(idx)
            same_group_index_by_example[idx] = key
        else:
            same_group_index[idx] = [idx]
            same_group_index_by_example[idx] = idx

    # save grouped parts image as result_obj, whole part boundary image as obj_region
    obj_region = np.zeros_like(bin)
    part_bound = []
    result_objs = []
    for cand in same_group_index.keys():
        result_obj = np.zeros_like(bin)
        cand_val = same_group_index[cand]
        for idx in cand_val:
            shape = obj_part[idx-1]
            x_start = shape[1].start
            y_start = shape[0].start
            x_end = shape[1].stop
            y_end = shape[0].stop

            region = labeled_part[y_start:y_end, x_start:x_end]
            region = ((region==idx)*255).astype(np.uint8)
            result_obj[y_start:y_end, x_start:x_end] += region
            obj_region[y_start:y_end, x_start:x_end] += region

        temp = np.where(result_obj == 255)
        y1, y2 = np.min(temp[0]), np.max(temp[0])
        x1, x2 = np.min(temp[1]), np.max(temp[1])
        part_bound.append([x1,y1,x2,y2])
##        cv2.imwrite(imgname.rstrip('.png')+'_partbound1_%d.png' % cand, bin[y1:y2, x1:x2])
        result_objs.append(result_obj)
#        cv2.imwrite(os.path.join(OUTPUT_DIR, imgname.rstrip('.png')+'_partbound1_%d.png' % cand), result_obj)

##    cv2.imwrite(imgname.rstrip('.png')+'_partbound1.png', obj_region)
    return obj_region, result_objs, part_bound

def detect_fasteners(img, imgname, parts_loc, parts_id, ud_check=True, in_check=True, h_obj_th=100, \
    h_min_th=15, h_max_th=150, rate_min_th=4, rate_max_th=None):
    """ detect fasteners in the image(img) and identify the relative position of the fasteners and the parts by using the boundaries of the parts """
    # input image preprocessing
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    inv = 255 - gray
    _, bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # structuring elements
    kernel_horz1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    kernel_diagl = diagonal_se(5, 'left')
    kernel_diagr = diagonal_se(5, 'right')
    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3,5))
    kernel_horz2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4,1))
    kernel_vert2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4,8)) #4,8
    kernel_vert3 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,4)) #3,6))
    kernel_vert4 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2)) #2,4 for stefan_06
    kernel_rec = cv2.getStructuringElement(cv2.MORPH_RECT, (5,4))
    kernel_horz3 = cv2.getStructuringElement(cv2.MORPH_RECT, (15,4))
    kernel_horz4 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))

    kernel_cc = [[1,1,1],[1,1,1],[1,1,1]]

    # 1. Detect candidates for thin vertical lines
    #  1) Remove horizontal, cross, left diagonal, right diagonal regions
    fined = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_horz1)
    fined2 = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_cross)
    fined3 = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_diagl)
    fined4 = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_diagr)

    comb = fined | fined3 | fined4
    sub_comb = np.maximum(inv-(comb), np.zeros_like(inv)).astype(np.uint8)
    #  2) Leave only (almost) vertical lines
    sub_comb = cv2.morphologyEx(sub_comb, cv2.MORPH_CLOSE, kernel_vert)
##    visualize = cv2.addWeighted(inv, 0.2, sub_comb, 0.8, 20)
##    cv2.imwrite(imgname.rstrip('.png')+'_subcomb.png', visualize)
    #  3) Binarization
    _, sub_comb = cv2.threshold(sub_comb, 127, 255, cv2.THRESH_BINARY)
    sub_comb = cv2.dilate(sub_comb, kernel_vert4, iterations=1)
    # 4) Additional removing
    fined = cv2.morphologyEx(sub_comb, cv2.MORPH_OPEN, kernel_horz2)
    sub_comb = sub_comb - fined

    # 2. Group the remained region (as connected components), then threshold the results according the the size
    # h_min_th = 15   ### HYPERPARAMETER
    # h_max_th = 150   ### HYPERPARAMETER
    # rate_th = 4     ### HYPERPARAMETER
    labeled_fastener, num = ndimage.label(sub_comb, kernel_cc)
    obj_fastener = ndimage.find_objects(labeled_fastener)
#    temp = (color.label2rgb(labeled_fastener)*255).astype(np.uint8)
#    cv2.imwrite(imgname.rstrip('.png')+'_subcomb_labeled_fastener.png', temp)
    candidates_fastener = list(range(1,num+1))
    for i in range(1, num+1):
        shape = obj_fastener[i-1]
        w_obj = shape[1].stop - shape[1].start
        h_obj = shape[0].stop - shape[0].start

        if h_obj <= h_min_th and i in candidates_fastener:
            candidates_fastener.remove(i)
        if h_obj >= h_max_th and i in candidates_fastener:
            candidates_fastener.remove(i)
        if h_obj/w_obj < rate_min_th and i in candidates_fastener:
            candidates_fastener.remove(i)
        if rate_max_th:
            # print('######################## Rate_max_Thresh Working! ########################')
            if h_obj/w_obj >= rate_max_th and i in candidates_fastener:
                candidates_fastener.remove(i)
    result = np.zeros_like(gray)
#    print('fastener', len(candidates_fastener))
    for i in range(len(candidates_fastener)):
        shape = obj_fastener[candidates_fastener[i]-1]
        x_start = shape[1].start
        y_start = shape[0].start
        x_end = shape[1].stop
        y_end = shape[0].stop

        region = labeled_fastener[y_start:y_end, x_start:x_end]
        region = ((region==candidates_fastener[i])*255).astype(np.uint8)
        result[y_start:y_end, x_start:x_end] += region

#    visualize = cv2.addWeighted(bin, 0.2, result, 0.8, 20)
#    cv2.imwrite(imgname.rstrip('.png')+'_result0_0.png', visualize)

    # 3. Segment the boundaries of parts
    obj_region, regions_, part_tight_boundaries_ = segment_part_boundaries(bin, kernel_rec, kernel_horz3, kernel_cc, imgname, h_obj_th)
    part_tight_boundaries = []
    regions = []
    indices = []
    # sort
    parts_loc_ = []
    for x,y,w,h in parts_loc:
        parts_loc_.append([x,y,x+w,y+h])
    detection_result = np.asarray(parts_loc_)
    detection_num = detection_result.shape[0]
    morphology_result = np.asarray(part_tight_boundaries_)
    morphology_num = morphology_result.shape[0]

    detection_result = np.tile(np.expand_dims(detection_result, 0), (morphology_num, 1, 1))
    morphology_result = np.tile(np.expand_dims(morphology_result, 1), (1, detection_num, 1))

    dist = np.square(detection_result-morphology_result) # sum(i,j,:) = i-morphology_result - j-detection_result
    dist = np.sqrt(np.sum(dist, axis=2))
    indices = list(np.argmin(dist, axis=0)) # index according to the detection result
    morp_indices = list(np.argmin(dist, axis=1)) # index according to the morphology result

    if len(part_tight_boundaries_) > len(parts_loc): # detection result; false negative
##        print('false negative detected')
        for ind in indices:
            part_tight_boundaries.append(part_tight_boundaries_[ind])
            regions.append(regions_[ind])
        missing_ind = list(set(range(len(part_tight_boundaries_))) - set(indices))
        for ind in missing_ind:
            part_tight_boundaries.append(part_tight_boundaries_[ind])
            regions.append(regions_[ind])

    elif len(part_tight_boundaries_) < len(parts_loc): # detection result; false positive
##        print('false positive detected')
        for ind in range(len(parts_loc)):
            if ind in morp_indices:
                index = morp_indices.index(ind)
                part_tight_boundaries.append(part_tight_boundaries_[index])
                regions.append(regions_[index])
            else:
                part_tight_boundaries.append([])
                regions.append([])
    else:
        for ind in indices:
            part_tight_boundaries.append(part_tight_boundaries_[ind])
            regions.append(regions_[ind])

    if in_check:
        for j in range(len(regions)):
            if regions[j] == []:
                continue
            temp = regions[j].copy()
            mask = np.zeros((temp.shape[0]+2, temp.shape[1]+2), np.uint8)
            cv2.floodFill(temp, mask, (0,0), 255)
            temp = cv2.bitwise_not(temp)
            temp_fill = regions[j] | temp
#            temp=ndimage.binary_fill_holes(regions[j]).astype(int)
            regions[j] = temp_fill
##            cv2.imwrite(imgname.rstrip('.png')+'_fill%d.png' % j, (temp_fill).astype(np.uint8))

    # 4. Select fasteners only having common pixels with part boundaries
    fasteners = {}
    fasteners_num = 0
    final_region = np.zeros_like(gray)
    result = cv2.dilate(result, kernel_vert4, iterations=1)
##    visualize = cv2.addWeighted(bin, 0.2, result, 0.8, 20)
##    cv2.imwrite(imgname.rstrip('.png')+'_result0.png', visualize)
    labeled_fastener, num = ndimage.label(result)
    temp = (color.label2rgb(labeled_fastener)*255).astype(np.uint8)
    obj_fastener = ndimage.find_objects(labeled_fastener)
    candidates_fastener = list(range(1,num+1))
    candidates_fastener_final = []
    for cand in candidates_fastener:
        shape = obj_fastener[cand-1]
        x1,x2 = shape[1].start, shape[1].stop
        y1,y2 = shape[0].start, shape[0].stop
        region = labeled_fastener == cand
        check = region * (obj_region/255) # fasteners must overlap with the part boundary
        incheck = 0
        part_idx = -1
        for obj_idx, region_obj in enumerate(regions):
            if region_obj == []:
                continue
            tempcheck = np.sum(region * region_obj/255)
            if incheck < tempcheck:
                incheck = tempcheck
                part_idx = obj_idx
        if np.sum(check) != 0:
            temp_pos = np.where(check>0.5)
            ymid = y1+(y2-y1)/2
            cy1 = np.min(temp_pos[0])
            cy2 = np.max(temp_pos[0])
            cx1 = np.min(temp_pos[1])
            cx2 = np.max(temp_pos[1])
        # to check the actual image
        temp = gray[max(0, y1-20):min(y2+20, gray.shape[0]), max(0, x1-20):min(x2+20, gray.shape[1])]
        temp2 = (region[max(0, y1-20):min(y2+20, gray.shape[0]), max(0, x1-20):min(x2+20, gray.shape[1])]*255).astype(np.uint8)
        temp2 = cv2.cvtColor(temp2, cv2.COLOR_GRAY2RGB)
        temp2[:,:,1:] = 0
        temp3 = (check[max(0, y1-20):min(y2+20, gray.shape[0]), max(0, x1-20):min(x2+20, gray.shape[1])]*255).astype(np.uint8)
        temp3 = cv2.cvtColor(temp3, cv2.COLOR_GRAY2RGB)
        temp3[:,:,0] = 0
        temp3[:,:,2] = 0
        temp23 = temp2+temp3
        temp_vis = cv2.addWeighted(cv2.cvtColor(temp, cv2.COLOR_GRAY2RGB), 0.2, temp23, 0.8, 20)
        if np.sum(check) == 0: # no overlap region
            continue
        elif np.sum(region)-25 <= np.sum(check) <= np.sum(region)+25: # part boundary totally covers the region
#            print('bound_checked')
            continue
        elif np.sum(region)-25 <= incheck <= np.sum(region)+25 and in_check:
#            print('in_checked')
            continue
        else:
            check_up = 0
            check_down = 0
            for j in range(len(temp_pos[0])):
                cy, cx = temp_pos[0][j], temp_pos[1][j]
                if cy >= ymid:
                    check_up = 1
                else:
                    check_down = 1
            if ud_check and (check_up and check_down):
#                print('up and down', shape)
                continue

#            print('ok', '#%d#%d#%d#%d' % (x1,y1,x2,y2), shape, 'rate h w', (y2-y1)/(x2-x1), 'overlap(region, check)', np.sum(region), np.sum(check), np.sum(region)/np.sum(check), 'in(check)', incheck)
            # to check
            candidates_fastener_final.append(cand)

            fasten_type = 'up' if check_up else 'down'
            part_type = -1
            if len(parts_id) < len(part_tight_boundaries):
                total_num = len(part_tight_boundaries)
            else:
                total_num = len(parts_id)
            for part_num in range(total_num): #parts_loc)):
                if part_tight_boundaries[part_num] == []:
                    continue
                px1,py1,px2,py2 = part_tight_boundaries[part_num] #parts_loc[part_num]
                overlap_area_part = 0
                if cx1>=px1 and cx2<=px2 and cy1>=py1 and cy2<=py2:
                    if part_type != -1:
                        part_num = part_idx
#                        print('duplicate, #%d#%d#%d#%d, parts loc: %d,%d,%d,%d, overlp loc: %d,%d,%d,%d' % (x1,y1,x2,y2, px,py,px+pw,py+ph, cx1,cy1,cx2,cy2))

                    part_type = part_num
            if part_type == -1:
##                print('no! part_type=-1, #%d#%d#%d#%d' % (x1,y1,x2,y2))
##                plt.clf()
##                plt.imshow(temp_vis)
##                plt.savefig(imgname.rstrip('.png')+'_removed_%d#%d#%d#%d.png' % (x1,y1,x2,y2), bbox_inches='tight')
                continue
            part_type = parts_id[min(part_type, len(parts_id)-1)]
            if part_type in fasteners.keys():
                if part_type == 'step8':
                    fasten_type = 'up'
                fasteners[part_type].append(fasten_type+'#%d#%d#%d#%d#%d#%d' % (x1,y1,x2,y2,(cx1+cx2)//2,(cy1+cy2)//2))
            else:
                fasteners[part_type] = [fasten_type+'#%d#%d#%d#%d#%d#%d' % (x1,y1,x2,y2,(cx1+cx2)//2,(cy1+cy2)//2)]
            # visualize
##            plt.clf()
##            plt.imshow(temp_vis)
##            plt.savefig(imgname.rstrip('.png')+'_%s_%s_#%d#%d#%d#%d.png' % (part_type, fasten_type, x1, y1, x2, y2), bbox_inches='tight')
##            plt.close()

            region = labeled_fastener[y1:y2, x1:x2]
            region = ((region==cand)*255).astype(np.uint8)
            final_region[y1:y2, x1:x2] += region
            fasteners_num += 1

##    print('fastener', len(candidates_fastener_final))
    visualize = cv2.addWeighted(bin, 0.2, final_region, 0.8, 20)
    cv2.imwrite(imgname.rstrip('.png')+'_fastener.png', visualize)

    output = {}
    for key in fasteners.keys():
        output[key] = [len(fasteners[key]), fasteners[key]]

    return output

def sort_2d_holes(part_holes, lkey, lorder, ukey, uorder, direction):
    positions = np.asarray(part_holes)
    positions = (positions[:,5:7]).astype(np.int32)
    positions = np.concatenate((positions, np.ones([positions.shape[0], 1], np.float32)), axis=-1)
    if direction == 'left':
        positions = np.transpose(np.matmul(M1, np.transpose(positions)))
    else:
        positions = np.transpose(np.matmul(M2, np.transpose(positions)))
    positions_z = np.tile(np.expand_dims(positions[:,-1],-1), [1,3])
    positions = positions/positions_z
    positions = positions[:,:2]

    positions = positions.tolist()
    indices = [x for x in range(len(positions))]

    part_holes_sort = sorted(sorted(zip(positions, indices), key=lambda x: x[0][0] if lkey=='x' else x[0][1], reverse=lorder), key=lambda x: x[0][1] if ukey=='y' else x[0][0], reverse=uorder)
    part_holes_sort_ind = [x[1] for x in part_holes_sort]

    part_holes_sort = []
    for ind in part_holes_sort_ind:
        part_holes_sort.append(part_holes[ind])

    return part_holes_sort

def convert_view_assembly_to_CAD(id_assembly, parts_id, parts_pose, parts_loc, connector=None, step_num=1):
    # input process
    hole_ids = {}
    check=0
    connect=0
    for part_id in parts_id: #'aaa'
        for part_connect in part_connectivity: #'aaa#bbb'
            if part_id == part_connect.split('#')[0]:
                if part_connect.split('#')[1] in parts_id:
                    check=1
                    connectivity = part_connect
                    connect_dic_hole = part_holes[connectivity]
                    break # assumption: one part-connectivity in one step
    if check==0:
        connectivity = ''

    for i in range(len(parts_id)): # for each part
        part_id = parts_id[i]
        if connectivity != '':
            connect = 1
        if part_id not in id_assembly.keys():
##            print('%s not in id_assembly' % part_id)
            hole_ids[part_id] = hole_ids.get(part_id, []) + []
            continue

        part_pose = str(parts_pose[i]) ## CORRECTION! POSE_GT
        part_pose_direction = 'left' if parts_pose[i]%2==0 else 'right'

        if 'part' in part_id:
            dic_pose = pose_dic_part[part_pose]
        else:
            dic_pose = pose_dic[part_pose]

        dic_hole = part_holes[part_id]
        part_hole = id_assembly[part_id][1] ### SORTING     x1,y1,x2,y2,(cx1+cx2)//2,(cy1+cy2)//2
        if len(part_hole) == 0:
            hole_ids[part_id] = []
            continue
        ##### sorting part_hole, to match the locations of labeled holes ########################
        part_hole_up = [x.split('#') for x in part_hole if 'up' in x]
        part_hole_down = [x.split('#') for x in part_hole if 'down' in x]
        if 'part' in part_id:
            part_loc_dic = loc_dic_part[part_pose]
        else:
            part_loc_dic = loc_dic[part_pose] # [up-up, up-left, down-up, down-left]
        up_ukey, up_uorder = sort_key[part_loc_dic[0]]
        up_lkey, up_lorder = sort_key[part_loc_dic[1]]
        down_ukey, down_uorder = sort_key[part_loc_dic[2]]
        down_lkey, down_lorder = sort_key[part_loc_dic[3]]
#        print('up_ukey: %s, up_uorder: %d,, up_lkey: %s, up_lorder: %d,, down_ukey: %s, down_uorder: %d,, down_lkey: %s, down_lorder: %d' % (up_ukey, up_uorder, up_lkey, up_lorder, down_ukey, down_uorder, down_lkey, down_lorder))
        if straight[part_id]:
            if len(part_hole_up) > 1:
                xdif_base = np.asarray([int(x[5]) for x in part_hole_up])
                xdif1 = np.tile(np.expand_dims(xdif_base, 0), [len(xdif_base), 1])
                xdif2 = np.tile(np.expand_dims(xdif_base, 1), [1, len(xdif_base)])
                xdif = abs(np.sum((xdif1-xdif2)[:,0]))
                ydif_base = np.asarray([int(x[6]) for x in part_hole_up])
                ydif1 = np.tile(np.expand_dims(ydif_base, 0), [len(ydif_base), 1])
                ydif2 = np.tile(np.expand_dims(ydif_base, 1), [1, len(ydif_base)])
                ydif = abs(np.sum((ydif1-ydif2)[:,0]))
                if xdif >= ydif:
                    up_key = 'x'
                    up_order = up_uorder if up_ukey=='x' else up_lorder
                else:
                    up_key = 'y'
                    up_order = up_uorder if up_ukey=='y' else up_lorder
            else:
                up_key, up_order = up_ukey, up_uorder

            if len(part_hole_down) > 1:
                xdif_base = np.asarray([int(x[5]) for x in part_hole_down])
                xdif1 = np.tile(np.expand_dims(xdif_base, 0), [len(xdif_base), 1])
                xdif2 = np.tile(np.expand_dims(xdif_base, 1), [1, len(xdif_base)])
                xdif = abs(np.sum((xdif1-xdif2)[:,0]))
                ydif_base = np.asarray([int(x[6]) for x in part_hole_down])
                ydif1 = np.tile(np.expand_dims(ydif_base, 0), [len(ydif_base), 1])
                ydif2 = np.tile(np.expand_dims(ydif_base, 1), [1, len(ydif_base)])
                ydif = abs(np.sum((ydif1-ydif2)[:,0]))
                if xdif >= ydif:
                    down_key = 'x'
                    down_order = down_uorder if down_ukey=='x' else down_lorder
                else:
                    down_key = 'y'
                    down_order = down_uorder if down_ukey=='y' else down_lorder
            else:
               down_key, down_order = down_ukey, down_uorder
            part_hole_up = sorted(part_hole_up, key=lambda x: int(x[5]) if up_key=='x' else int(x[6]), reverse=up_order)
            part_hole_down = sorted(part_hole_down, key=lambda x: int(x[5]) if down_key=='x' else int(x[6]), reverse=down_order)
        else:
            if len(part_hole_up)>0:
                part_hole_up = sort_2d_holes(part_hole_up, up_lkey, up_lorder, up_ukey, up_uorder, part_pose_direction)
            if len(part_hole_down)>0:
                part_hole_down = sort_2d_holes(part_hole_down, down_lkey, down_lorder, down_ukey, down_uorder, part_pose_direction)
#            part_hole_up = sorted(sorted(part_hole_up, key=lambda x: int(x[5]) if up_lkey=='x' else int(x[6]), reverse=up_lorder), key=lambda x: int(x[6]) if up_ukey=='y' else int(x[5]), reverse=up_uorder) #sorted(sorted(left), up)
#            part_hole_down = sorted(sorted(part_hole_down, key=lambda x: int(x[5]) if down_lkey=='x' else int(x[6]), reverse=down_lorder), key=lambda x: int(x[6]) if down_ukey=='y' else int(x[5]), reverse=down_uorder)
        part_hole = part_hole_up + part_hole_down
        ################################

        hole_ids[part_id] = hole_ids.get(part_id, [])
        for p in part_hole:
            p_type = p[0]
            CAD_type = dic_pose[0] if p_type=='up' else dic_pose[1]
            if len(dic_hole)==0:
                continue
            else:
                hole_type_candidates = [x for x in dic_hole if x[-1] == CAD_type] #[0:2] == CAD_type]
                if len(hole_type_candidates) == 0:
##                    print('not hole', p)
                    continue
                if connector is not None:
                    hole_type_candidates = [x for x in hole_type_candidates if x[4] == connector] #.split('#')[2] == connector]
                if len(hole_type_candidates) == 0:
##                    print('not available hole', p) #step 4,5 : step4 - wrong dic_hole(5), wrong OCR result, step 9: wrong dic_hole
                    continue
                hole_type = hole_type_candidates[0]
                if connect and connectivity.split('#')[0] == part_id: # need to be concerned..
                    connect_part_id = connectivity.split('#')[1]
                    connect_hole_type = [x.split('#')[3] for x in connect_dic_hole if x.split('#')[1]==hole_type[0]]
##                    if len(connect_hole_type) > 1:
##                        print('multiple hole candidates!, len %d' % len(connect_hole_type), connect_hole_type, 'for %s' % part_id)
                    if len(connect_hole_type) >= 1:
                        connect_hole_type = connect_hole_type[0]
                        connect_hole_type = [x[0] for x in part_holes[connect_part_id] if x[0]==connect_hole_type][0]
                        hole_ids[connect_part_id] = hole_ids.get(connect_part_id, []) + [[connect_hole_type, []]]
##                    else: #len(connect_hole_type) == 0
##                         print('not connected hole!', hole_type)
                part_holes[part_id].remove(hole_type)
            hole_ids[part_id].append([hole_type[0], p])

    id_CAD = []
    for part_id in parts_id:
        if connect:
            # for remove duplicate
            num=0
            hole_id_temp = {}
            for k,v in hole_ids[part_id]:
                hole_id_temp[k] = hole_id_temp.get(k,[]) + v
            # for location-not specified holes
            for k in hole_id_temp.keys():
                if hole_id_temp[k] == []:
                    connect_id = [x for x in part_connectivity if part_id in x][0]
                    connect_part_id = connect_id.split('#')[0]
                    connect_hole_ids = part_holes[connect_id]
                    connect_hole_id = [x for x in connect_hole_ids if k in x.split('#')[3]][0] #[6:]][0]
                    connect_hole_loc = [x for x in hole_ids[connect_part_id] if x[0] == connect_hole_id.split('#')[1]] #connect_hole_id[0:5] in x[0]]
                    connect_hole_loc = connect_hole_loc[0][1]
                    temp_x1 = connect_hole_loc[1]
                    temp_y1 = str(int(connect_hole_loc[4])+100)
                    temp_x2 = connect_hole_loc[3]
                    temp_y2 = str(int(connect_hole_loc[4])+100+int(connect_hole_loc[4])-int(connect_hole_loc[2]))
                    hole_id_temp[k] = ['pp', temp_x1, temp_y1, temp_x2, temp_y2]
                    num += 1
            hole_id_list = []
            for k in hole_id_temp.keys():
                hole_id_list.append([k, hole_id_temp[k]])
        else:
            hole_id_list = hole_ids[part_id]

        id_CAD.append(hole_id_list)

    # hole pair print
    hole_pair_step = deepcopy(hole_pairs[str(step_num)])
    hole_pair_step_print = []

    candidate_holes = []
    for pid in range(len(parts_id)):
        pa_id = parts_id[pid]
        id_CAD_part = id_CAD[pid]
        temp = ['%s_1-hole_%s'%(pa_id, x[0]) for x in id_CAD_part] if 'part' in pa_id else [x[0] for x in id_CAD_part]
        candidate_holes.extend(temp)

    for hole_step in hole_pair_step:
        h = hole_step.split('#')
        check=1
        for hh in h:
            if 'C' in hh:
                continue
            else:
                if hh not in candidate_holes:
                    check=0
        if check:
            hole_pair_step_print.append(hole_step)

    return id_CAD, connectivity, hole_pair_step_print
