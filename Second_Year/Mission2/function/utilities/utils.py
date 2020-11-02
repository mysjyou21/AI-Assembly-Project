import os
import shutil
import time
import collections
import scipy.ndimage as ndimage
import math
from PIL import Image
from PIL.ImageOps import invert
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob
import tensorflow as tf


assembly_list = ['ekedalen', 'gamleby', 'idolf', 'ingatorp', 'ingolf', 'ivar', 'kaustby', 'lerhamn', 'nordmyra', 'nordviken', 'norraryd', 'norrnas', 'stefan', 'stefan_9', 'input', 'mission1', 'mission2']

# def save_group_intermediate() -이삭

def set_connectors(I):
    """ 
    set the list of connectors from the first page
    I: the first page containing the whole number and category of the connectors 
    return: connectors={key: value}, key=serial number of the connector, value: the number of the connector in this assembly """
    connectors = {}
    # Find the connectors(serial number, number of connectors)
    serial_nums = []
    num_of_connectors = []
    for i in range(len(serial_nums)):
        serial_num = serial_nums[i]
        num_of_connector = num_of_connectors[i]
        connectors[serial_num] = num_of_connector

    return connectors

def diamond_se(r):
        kernel = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.uint8)
        for i in range(0, r):
            kernel[i, r - i:r + i + 1] = 1
            kernel[2 * r - i, r - i:r + i + 1] = 1
        kernel[r, :] = 1
        return kernel
    
def set_steps_from_cut(I, current_step_num=1):
    """
    조립도가 순서대로 주어진다는 가정 하에 짰음.
    모폴로지 연산 이용하여 step number detection 수행.
    extract steps from a cut.

    I: a cut image
    return:
        is_steps: list, boolean
            나눈 이미지가 step 나타내는 이미지면 True, 연결자 소개하는 이미지면 False
        step_imgs: list, array
            images corresponding to each step
    """
    step_nums = []
    is_steps = []
    step_imgs = []

    img = np.copy(I)
    H, W, C = img.shape
    gray = img[:, :, 0]
    inv = 255 - gray
    inv_binary = (inv > 100).astype(int)

    ## erosion 이용하여 step number 위치 찾기
    _, inv = cv.threshold(inv, thresh=100, maxval=255, type=cv.THRESH_BINARY)
    ## diamond-shaped StructuringElement로 erosion
    r = int(4 * W / 1166)

    kernel = diamond_se(r)
    eroded = cv.erode(inv, kernel, iterations=1)
    ## erosion 후에도 남아있는 object들을 labeling
    label_array, num_feature = ndimage.label(eroded)
    objects = ndimage.find_objects(label_array)
    ## objects 중 조립도의 왼쪽 적당한 위치(step 번호가 위치할만한)에 있는 것만 keep
    obj_temp = []
    # conds = [int(W * 150 / 2480), int(W * 350 / 2480)]
    conds = [int(W * 0.06), int(W * 0.17), int(H * 0.8)]
    for obj in objects:
        if obj is None:
            break
        else:
            y = obj[0].start
            h = obj[0].stop - y
            x = obj[1].start
            w = obj[1].stop - x
            if x > conds[0] and x + w < conds[1] and y < conds[2]:
                obj_temp.append([x, y, w, h])

    ## 그것들 중 크기 너무 작은것들(가로 세로 둘중 하나라도 5px 이하)은 버림
    obj_final = []
    for obj in obj_temp:
        if obj is None:
            break
        else:
            x, y, w, h = obj
            if (w > 5) and (h > 5):
                obj_final.append(obj)

    ## step number 가 없으면 연결자만 소개된 cut
    ## step number 없으면 여기서 함수 종료.
    if len(obj_final) == 0:
        is_steps = [False]
        step_imgs = [img]
        return step_nums, is_steps, step_imgs

    ## step number가 존재할 때, 긴 가로선이 있으면 연결자 소개랑 step이랑 같이 나와있는 cut임
    inv_horizontal_sum = np.sum(inv_binary, axis=1)
    hor_line_idx = 0
    ## 가로선의 길이가 조립도 가로 길이의 0.8배 이상이면 구분선으로 간주하고 step image를 나눈다
    if inv_horizontal_sum.max() >= 0.8 * W:
        hor_line_idx = np.argmax(inv_horizontal_sum)
        # 가로선 위의 이미지를 append, margin은 5 pixels
        is_steps.append(False)
        step_imgs.append(img[0:hor_line_idx - 5])
        # 기존 img, inv를 가로선 아래의 이미지로 replace
        img = img[hor_line_idx + 5: ]
        inv = inv[hor_line_idx + 5: ]
        H, W, C = img.shape

    ## step number가 여러 개 존재하면 쪼개야 함.
    ## step number들의 height 위치가 두 종류 이상인가?
    ## y_list: step number들의 (왼쪽 위 점의) y좌표 list
    y_list = [obj[1] - hor_line_idx for obj in obj_final]
    y_list_temp = y_list.copy()
    y_list_final = [y_list[0]]
    for y in y_list:
        # step number의 y좌표가 5픽셀 이내로 차이나면 같은 step number로 취급
        # (ex. 10, 11 처럼 두자릿수 step number)
        if np.abs(y - y_list_final[-1]) > 5:
            y_list_final.append(y_list_temp[0])
        y_list_temp.remove(y)
    y_list = y_list_final
    y_list.append(-1)

    for i in range(len(y_list) - 1):
        is_steps.append(True)
        step_nums.append(current_step_num + i + 1)
        y_start = max(0, y_list[i] - 80)
        y_end = y_list[i+1] - 80
        step_imgs.append(img[y_start : y_end])
    return step_nums, is_steps, step_imgs


def grouping(circles, rectangles, connectors, connectors_serial, connectors_mult, tools, serials_OCR, mults_OCR):
    """
    [ [[element1_1], [element1_2]], [[element2_1]], [] ] 이런 식으로 grouping 수행.
    ---------------------------
    :param
    circles ~ tools: [[element1], [element1], ..., [elementN]]
    ---------------------------
    :returns
    circles_new: 인접한 원들은 하나의 bounding box로 뭉친 list.
        [[circle1], ..., [circleN]]
    connectors_list_new, connectors_serial_list_new, connectors_mult_list_new, tools_list_new: grouping된 list.
        [ [[element1_1], [element1_2]], [[element2_1]], [] ]
    """
    
    circles_list = circles.copy()
    rectangles_list = rectangles.copy()
    connectors_list = connectors.copy()
    connectors_serial_list = connectors_serial.copy()
    connectors_mult_list = connectors_mult.copy()
    tools_list = tools.copy()
    serials_OCR_list = serials_OCR.copy()
    mults_OCR_list = mults_OCR.copy()
    ### 1. rectangle 영역 안의 것들은 제거


    # NOTES : 이삭
    # rectangle 안에서 mult의 'X'가 발견되면 그건 사각형 모양 circle임(일반 사각형처럼 지워버리면 안되는 영역, grouping에 사용해야하는 영역)
    # 1년차 챌린지 1,2에서 받은 이미지들은 사각형 모양 circle이 발견되는 경우가 발생할 때 이미지 내에 원형 모양 circle이 없고, 일반 사각형이 없어서,
    # 이 경우 grouping을 할 때에는 이미지 내에서 circle를 하나도 못 찾은 경우와 동일하게 처리 (검출된 mult, serial 다 동일 group으로 처리
    #(circle을 못 찾아도 grouping은 정상 작동하도록 이렇게 코딩했었음))

    for rectangle in rectangles_list:
        x_rec, y_rec, w_rec, h_rec = rectangle
        for obj_list in [circles_list, connectors_list, connectors_serial_list, connectors_mult_list, tools_list]:
            for idx, obj in enumerate(obj_list):
                x_obj, y_obj, w_obj, h_obj = obj[:4]
                # obj의 왼쪽 위 좌표랑 오른쪽 아래 좌표가 모두 rectangle 내부에 존재하면 remove
                if (x_obj >= x_rec) and (x_obj + w_obj <= x_rec + w_rec) and (y_obj >= y_rec) and (
                        y_obj + h_obj <= y_rec + h_rec):
                    obj_list.remove(obj)
                    # obj_list 가 serial 또는 mult의 위치를 나타내는 list인 경우, 해당 OCR list도 처리
                    if obj_list == connectors_serial_list:
                        serials_OCR_list.remove(obj_list[idx])
                    if obj_list == connectors_mult_list:
                        mults_OCR_list.remove(obj_list[idx])
                    # print('removed')

    ### 2. 인접한 circle들끼리 grouping
    def is_connected(circle1, circle2):
        left1, right1, top1, bot1 = circle1[0], circle1[0] + circle1[2], circle1[1], circle1[1] + circle1[3]
        left2, right2, top2, bot2 = circle2[0], circle2[0] + circle2[2], circle2[1], circle2[1] + circle2[3]
        cond1 = (left1 > right2)
        cond2 = (right1 < left2)
        cond3 = (top1 > bot2)
        cond4 = (bot1 < top2)
        if cond1 or cond2 or cond3 or cond4:
            return 0
        else:
            return 1

    ## 2-1. is_connected_mtx의 (a,b) 성분: a번째 circle이랑 b번째 circle이 인접한 관계이면 1, 아니면 0
    num_circles = len(circles_list)
    is_connected_mtx = np.eye(num_circles, dtype=int)
    for i in range(0, num_circles - 1):
        for j in range(i + 1, num_circles):
            # print(i, j)
            circle_i = circles_list[i]
            circle_j = circles_list[j]
            is_connected_mtx[j, i] = is_connected_mtx[i, j] = is_connected(circle_i, circle_j)

    ## 2-2. reachable_mtx의 (a,b) 성분: circle a에서 circle b 까지 도달할 수 있는 경로의 수 (이게 0이 아니면 연결되어 있다는 뜻)
    reachable_mtx = np.zeros((num_circles, num_circles), dtype=int)
    for n in range(num_circles):
        reachable_mtx += np.linalg.matrix_power(is_connected_mtx, n + 1)

    ## 2-3. reachable mtx를 토대로 circle의 index들을 grouping
    group_list = []
    for n in range(num_circles):
        indices = np.argwhere(reachable_mtx[n] != 0)
        indices = sorted(np.reshape(indices, -1).tolist())
        if indices not in group_list:
            group_list.append(indices)

    ## 2-4. index group으로 원들 merging & grouping
    circles_new = []
    for group_idx, indices in enumerate(group_list):
        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []
        for idx in indices:
            x, y, w, h = circles[idx]
            x1_list.append(x)
            y1_list.append(y)
            x2_list.append(x + w)
            y2_list.append(y + h)
        x_new = np.min(x1_list)
        y_new = np.min(y1_list)
        w_new = np.max(x2_list) - x_new
        h_new = np.max(y2_list) - y_new
        # circles_new.append([x_new, y_new, w_new, h_new, group_idx])
        circles_new.append([x_new, y_new, w_new, h_new])

    # 인접한 원들끼리 grouping이 발생했으면 is_merged=True, 그렇지 않으면 False
    if len(circles) != len(circles_new):
        is_merged = True
    else:
        is_merged = False

    ### 3. 원의 index에 맞춰서 다른 object들도 grouping
    ## 원이 없는데 object는 있는 경우, 싹 다 index 0으로 grouping
    ## 거리가 가장 가까운 원으로 grouping.
    ## OCR 결과는 위치 정보가 없으므로 해당하는 serial/mult loc의 group을 따라감.
    for obj_list in [connectors_list, connectors_serial_list, connectors_mult_list, tools_list]:
        for n, obj in enumerate(obj_list):
            # 각 object의 중심 좌표 x, y를 구함.
            # connectors_mult_list의 경우 이미 중심 좌표가 구해져 있고,
            # 그 외 list에 대해서는 x와 y에 각각 w/2, h/2를 더해 중심 좌표로 바꿔줌
            x, y, w, h = obj[:4]
            if obj_list != connectors_serial_list:
                x = x + int(w/2)
                y = y + int(h/2)
            # 원이 없으면 index 무조건 0
            if len(circles_new) == 0:
                group_idx = 0
                obj.append(group_idx)
                # for OCR list
                if obj_list == connectors_serial_list:
                    try:
                        serials_OCR_list[n].append(group_idx)# Isaac
                    except:
                        serials_OCR_list = [list(x) for x in serials_OCR_list] # Isaac
                        serials_OCR_list[n].append(group_idx)# Isaac
                if obj_list == connectors_mult_list:
                    try:
                        mults_OCR_list[n].append(group_idx)# Isaac
                    except:
                        mults_OCR_list = [list(x) for x in mults_OCR_list]# Isaac
                        mults_OCR_list[n].append(group_idx)# Isaac
            else:
                dist_list = []
                for circle in circles_new:
                    x_cir, y_cir, w_cir, h_cir = circle[0:4]
                    dist = (x - x_cir)**2 + (y - y_cir)**2
                    dist_list.append(dist)
                group_idx = int(np.argmin(dist_list))
                obj.append(group_idx)
                # for OCR list
                if obj_list == connectors_serial_list:
                    serials_OCR_list[n] = [serials_OCR_list[n], group_idx]
                if obj_list == connectors_mult_list:
                    mults_OCR_list[n] = [mults_OCR_list[n], group_idx]

    ### 4. grouping 형식 수정
    ## [x, y, w, h, idx] 이런 식으로 뒤에 group index 붙이는 게 아니라
    ## [ [[element1_1], ..., [element1_n]], [ (no element in group 2) ], [[element3_1]] ] 이런 형식으로 수정
    # 우선 return할 list들을 빈 list로 초기화
    connectors_list_new = []
    connectors_serial_list_new = []
    connectors_mult_list_new = []
    tools_list_new = []
    serials_OCR_list_new = []
    mults_OCR_list_new = []
    # 만든 빈 list들에 원의 개수만큼 빈 list를 만들어 놓음
    for obj_list in [connectors_list_new, connectors_serial_list_new, connectors_mult_list_new, tools_list_new, serials_OCR_list_new, mults_OCR_list_new]:
        for _ in range(len(circles_new)):
            obj_list.append([])

    old_list = [connectors_list, connectors_serial_list, connectors_mult_list, tools_list, serials_OCR_list, mults_OCR_list]
    new_list = [connectors_list_new, connectors_serial_list_new, connectors_mult_list_new, tools_list_new, serials_OCR_list_new, mults_OCR_list_new]

    if len(np.array(new_list).shape) < 3: # Isaac
        temp = np.array(new_list)# Isaac
        temp = np.expand_dims(temp,1)# Isaac
        new_list = temp.tolist()# Isaac

    for old_obj_list, new_obj_list in zip(old_list, new_list):

        for old_obj in old_obj_list:
            if old_obj_list in [serials_OCR_list, mults_OCR_list]:
                ocr_result = old_obj[0]
                group_idx = old_obj[1]
                try: 
                    # new_obj_list[group_idx].append(ocr_result) # Isaac
                    new_obj_list[int(group_idx)].append(ocr_result) # Isaac
                except:
                    # import ipdb; ipdb.set_trace(context=21) #red # Isaac
                    print() # Isaac
            else:
                bb = old_obj[0:-1]
                
                group_idx = old_obj[-1]
                new_obj_list[group_idx].append(bb)

    # serial은 있는데 mult가 빈 list 인 경우, mult_list를 [['1']]로 replace
    for n, serial_OCR_list in enumerate(serials_OCR_list_new):
        if len(serials_OCR_list) != 0 and len(mults_OCR_list_new[n]) == 0:
            mults_OCR_list_new[n] = ['1']

    return circles_new, connectors_list_new, connectors_serial_list_new, connectors_mult_list_new, tools_list_new, serials_OCR_list_new, mults_OCR_list_new, is_merged


def div_cut(I):
    """ divide first cut if it has materials on its top """
    H, W, C = I.shape
    gray = I[:, :, 0]
    inv = 255 - gray
    # erosion to inverted gray-scale image with diamond-shaped-StructuringElement
    r = int(6 * W / 1166)
    r_d = int(3 * W / 1166)
    num_w, num_h = W / 6, H / 5

    # to set position of actions
    kernel_pos = diamond_se(r)
    erode_pos = cv.erode(inv, kernel_pos, iterations=1)
    # thresholding eroded image
    bin_pos = erode_pos > 200
    # labeling thresholded image and original image
    bin_labeled_array_pos, num_feature = ndimage.label(bin_pos)

    # to get detailed shape of actions
    kernel_detail = diamond_se(r_d)
    erode_detail = cv.erode(inv, kernel_detail, iterations=1)
    bin_detail = erode_detail > 200
    bin_labeled_array_detail, _ = ndimage.label(bin_detail)
    bin_objects_detail = ndimage.find_objects(bin_labeled_array_detail)

    for i in range(num_feature):
        # pos = bin_objects_pos[i]
        pos_y, pos_x = np.where(bin_labeled_array_pos == i + 1)
        num = bin_labeled_array_detail[pos_y[0], pos_x[0]]
        # except sequence numbers and too-weak action candidates
        shape = bin_objects_detail[num - 1]
        x = shape[1].start
        y = shape[0].start
        w = shape[1].stop - x
        h = shape[0].stop - y
        if 3.6 > h / w > 3 and max(pos_x) < num_w:
            if max(pos_y) > num_h:
                # I = I[min(pos_y) - 10:, :, :]
                cv.rectangle(I, (0, 0), (I.shape[1], min(pos_y) - 10), (255, 255, 255), -1)
    return I


def save_result_text(self):
    with open('./intermediate_results/results_rect.txt', 'a') as f:
        f.write(str(self.opt.assembly_name) + ' ')
        f.write(str(rect_output))
        f.write('\n')
    with open('./intermediate_results/results_bubble.txt', 'a') as f:
        f.write(str(self.opt.assembly_name) + ' ')
        f.write(str(bubble_output))
        f.write('\n')
    with open('./intermediate_results/results_num.txt', 'a') as f:
        f.write(str(self.opt.assembly_name) + ' ')
        f.write(str(num_output))
        f.write('\n')
    with open('./intermediate_results/results_mult.txt', 'a') as f:
        f.write(str(self.opt.assembly_name) + ' ')
        f.write(str(mult_output))
        f.write('\n')


def save_group_intermediate(self):
    print('\nsave group image')
    refresh_folder(self.opt.group_image_path)
    for i, cut in enumerate(self.cut):
        img = np.copy(cut)
        circle_group_loc = np.array(self.circle_group_loc[i])
        serial_loc = np.array(self.serial_loc[i])
        mult_loc = np.array(self.mult_loc[i])
        RECTANGLE = True
        if RECTANGLE:
            if self.rectangle_check[i] > 0:
                for rect_loc in self.rectangle_loc[i]:
                    x, y, w, h = rect_loc
                    cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 4)
        if np.sum(np.array(circle_group_loc)) != 0:
            for j in range(circle_group_loc.shape[0]):
                x, y, w, h, group_index = circle_group_loc[j]
                cv.rectangle(img, (x, y), (x + w, y + h), return_color_tuple(group_index), 3)
        if np.sum(np.array(serial_loc)) != 0:
            for j in range(serial_loc.shape[0]):
                x, y, group_index, angle, width, height = serial_loc[j]
                mid_point = (x, y)
                img = drawBoundingBox(img, mid_point, angle, width, height, group_index)
        if np.sum(np.array(mult_loc)) != 0:
            for j in range(mult_loc.shape[0]):
                x, y, w, h, group_index = mult_loc[j]
                cv.rectangle(img, (x, y), (x + w, y + h), return_color_tuple(group_index), 3)

        cv.imwrite(os.path.join(self.opt.group_image_path, self.opt.assembly_name + '_' + str(i).zfill(2) + '.png'), img)


def tostring(numbers):
    # numbers: 부품 번호, list 형식. [1, 0, 0, 0, 0, 1]
    result = ''
    for number in numbers:
        result += str(number)
    return result


def return_color_tuple(color):
    if color == 'red' or color == 1:
        color_tuple = (0, 0, 255)
    if color == 'green' or color == 2:
        color_tuple = (0, 255, 0)
    if color == 'blue' or color == 3:
        color_tuple = (255, 0, 0)
    if color == 'yellow' or color == 4:
        color_tuple = (0, 255, 255)
    if color == 'cyan' or color == 5:
        color_tuple = (255, 255, 0)
    if color == 'magenta' or color == 0:
        color_tuple = (255, 0, 255)
    return color_tuple


def get_serial_detect_values(x):
    # mid point
    mid_point = (x[0] + x[1]) / 2
    # angle (deg)
    angle_deg = x[2][1] % 360
    if angle_deg > 0 and angle_deg <= 90:
        angle_deg = angle_deg
    elif angle_deg > 90 and angle_deg <= 180:
        angle_deg = angle_deg - 180
    elif angle_deg > 180 and angle_deg <= 270:
        angle_deg = angle_deg - 180
    elif angle_deg > 270 and angle_deg <= 360:
        angle_deg = angle_deg - 360
    if np.absolute(angle_deg - 0) < 5:
        angle_deg = 0
    if np.absolute(angle_deg - 90) < 5:
        angle_deg = 90
    if np.absolute(angle_deg - (-90)) < 5:
        angle_deg = 90
    # width 
    width = x[2][0] * 12.2 / 10 * (x[3][0] + 6) / 6
    # height
    height = x[2][0] * 3.5 / 10
    return mid_point, angle_deg, width, height


def get_serial_detect_corners(mid_point, angle_deg, width, height):
    angle = np.deg2rad(-angle_deg)
    ul = (
        int(mid_point[0] - width / 2 * np.cos(angle) - height / 2 * np.sin(angle)),
        int(mid_point[1] - width / 2 * np.sin(angle) + height / 2 * np.cos(angle))
    )
    ur = (
        int(mid_point[0] + width / 2 * np.cos(angle) - height / 2 * np.sin(angle)),
        int(mid_point[1] + width / 2 * np.sin(angle) + height / 2 * np.cos(angle))
    )
    lr = (
        int(mid_point[0] + width / 2 * np.cos(angle) + height / 2 * np.sin(angle)),
        int(mid_point[1] + width / 2 * np.sin(angle) - height / 2 * np.cos(angle))
    )
    ll = (
        int(mid_point[0] - width / 2 * np.cos(angle) + height / 2 * np.sin(angle)),
        int(mid_point[1] - width / 2 * np.sin(angle) - height / 2 * np.cos(angle))
    )
    return ul, ur, lr, ll


def drawBoundingBox(img, base, angle_deg, width, height, color='red'):
    angle = np.deg2rad(-angle_deg)  # invert becasue opencv y-axis heads down
    ul = (
        int(base[0] - width / 2 * math.cos(angle) - height / 2 * math.sin(angle)),
        int(base[1] - width / 2 * math.sin(angle) + height / 2 * math.cos(angle))
    )
    ur = (
        int(base[0] + width / 2 * math.cos(angle) - height / 2 * math.sin(angle)),
        int(base[1] + width / 2 * math.sin(angle) + height / 2 * math.cos(angle))
    )
    lr = (
        int(base[0] + width / 2 * math.cos(angle) + height / 2 * math.sin(angle)),
        int(base[1] + width / 2 * math.sin(angle) - height / 2 * math.cos(angle))
    )
    ll = (
        int(base[0] - width / 2 * math.cos(angle) + height / 2 * math.sin(angle)),
        int(base[1] - width / 2 * math.sin(angle) - height / 2 * math.cos(angle))
    )

    thickness = 3
    cv.line(img, ul, ur, return_color_tuple(color), thickness)
    cv.line(img, ur, lr, return_color_tuple(color), thickness)
    cv.line(img, lr, ll, return_color_tuple(color), thickness)
    cv.line(img, ll, ul, return_color_tuple(color), thickness)
    return img


_Point = collections.namedtuple('Point', ['x', 'y'])


class Point(_Point):

    def __add__(self, p):
        return Point(self.x + p.x, self.y + p.y)

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def recenter(self, old_center, new_center):
        return self + (new_center - old_center)

    def rotate(self, center, angle_deg):
        angle = np.deg2rad(-angle_deg)
        x = math.cos(angle) * (self.x - center.x) - math.sin(angle) * (self.y - center.y) + center.x
        y = math.sin(angle) * (self.x - center.x) + math.cos(angle) * (self.y - center.y) + center.y
        return Point(x, y)


def getCenter(im):
    # center point of image
    return Point(*(d / 2 for d in im.size))


Bound = collections.namedtuple('Bound', ('left', 'upper', 'right', 'lower'))


def getBounds(points):
    # Bound object with min max x y
    xs, ys = zip(*points)
    return Bound(min(xs), min(ys), max(xs), max(ys))


def getBoundsPoints(bounds):
    # min max x y
    left, upper, right, lower = bounds
    points = []
    points.append(Point(left, upper))
    points.append(Point(right, upper))
    points.append(Point(left, lower))
    points.append(Point(right, lower))
    return points


def getBoundsCenter(bounds):
    # center point of rect
    return Point(
        (bounds.right - bounds.left) / 2 + bounds.left,
        (bounds.lower - bounds.upper) / 2 + bounds.upper
    )


def roundint(values):
    return tuple(int(round(v)) for v in values)


def getRotatedRectanglePoints(base_point, angle, width, height):
    # retrive corner points from mid_point, angle, widht, height
    angle = np.deg2rad(-angle)

    ul = Point(
        base_point.x - width / 2 * math.cos(angle) - height / 2 * math.sin(angle),
        base_point.y - width / 2 * math.sin(angle) + height / 2 * math.cos(angle)
    )
    ur = Point(
        base_point.x + width / 2 * math.cos(angle) - height / 2 * math.sin(angle),
        base_point.y + width / 2 * math.sin(angle) + height / 2 * math.cos(angle)
    )
    lr = Point(
        base_point.x + width / 2 * math.cos(angle) + height / 2 * math.sin(angle),
        base_point.y + width / 2 * math.sin(angle) - height / 2 * math.cos(angle)
    )
    ll = Point(
        base_point.x - width / 2 * math.cos(angle) + height / 2 * math.sin(angle),
        base_point.y - width / 2 * math.sin(angle) - height / 2 * math.cos(angle)
    )

    return tuple(Point(0, 0) + pt for pt in (ul, ur, lr, ll))


def crop(im, base, angle_deg, width, height):
    """ Return a new, cropped image.
     Args:
        im: a PIL.Image instance
        base: a (x,y) tuple for the center of the cropped area
        angle: angle, in radians, for which the cropped area should be rotated
        height: height in pixels of cropped area
        width: width in pixels of cropped area
    """
    base = Point(*base)
    points = getRotatedRectanglePoints(base, angle_deg, width, height)
    im_np = np.array(im)
    for i in range(4):
        cv.circle(im_np, (int(points[i].x), int(points[i].y)), 4, (0, 0, 255), -1)
    # show(im_np)
    return _cropWithPoints(im, angle_deg, points)


def _cropWithPoints(im, angle_deg, points):
    # crop from image
    crop_bound = getBounds(points)
    crop = im.crop(roundint(crop_bound))
    crop = invert(crop)
    crop_np = np.array(crop)
    # recenter crop points (because of crop)
    crop_bound_center = getBoundsCenter(crop_bound)
    # print(crop_bound_center)
    # print(points)
    crop_center = getCenter(crop)
    # print(crop_center)
    crop_points = [pt.recenter(crop_bound_center, crop_center) for pt in points]
    # print(crop_points)
    # print(crop_np.shape)
    for i in range(4):
        cv.circle(crop_np, (int(crop_points[i].x), int(crop_points[i].y)), 4, (0, 0, 255), -1)
    # show(crop_np)
    crop_rotated_expanded = crop.rotate(-angle_deg, resample=Image.BICUBIC, expand=True)
    crop_rotated_expanded_np = np.array(crop_rotated_expanded)
    # recenter crop points (because of rotation)
    # first get boundary of rotated_expanded image
    crop_bound_points = getBoundsPoints(crop_bound)
    # print(crop_bound_points)
    crop_bound_points = [pt.recenter(crop_bound_center, crop_center) for pt in crop_bound_points]
    # print(crop_bound_points)
    rotated_crop_bound_points = [pt.rotate(crop_center, angle_deg) for pt in crop_bound_points]
    # print(rotated_crop_bound_points)
    rotated_crop_bound_points_bound = getBounds(rotated_crop_bound_points)
    # print(rotated_crop_bound_points_bound)
    rotated_crop_bound_points_bound_center = getCenter(crop_rotated_expanded)
    # print(rotated_crop_bound_points_bound_center)
    rotated_crop_points = [pt.rotate(crop_center, -angle_deg) for pt in crop_points]
    # print(rotated_crop_points)
    rotated_crop_points_recentered = [pt.recenter(crop_center, rotated_crop_bound_points_bound_center) for pt in rotated_crop_points]
    # print(rotated_crop_points_recentered)
    for i in range(4):
        cv.circle(crop_rotated_expanded_np, (int(rotated_crop_points_recentered[i].x), int(rotated_crop_points_recentered[i].y)), 4, (0, 0, 255), -1)
    # show(crop_rotated_expanded_np)
    crop1 = crop_rotated_expanded.crop(roundint(getBounds(rotated_crop_points_recentered)))
    crop1 = invert(crop1)
    crop1_np = np.array(crop1)
    # show(crop1_np)
    return crop1


def show(img, name=''):
    cv.imshow(name, img)
    cv.waitKey()
    cv.destroyAllWindows()


def get_iou(rec1, rec2):
    """
    Calculate IOU between two bounding boxes.
    :param rec1: (x1, y1, h1, w1)
    :param rec2: (x2, y2, h2, w2)
    :return iou: IOU between rec1 and rec2
    """
    x11, y11, h1, w1 = rec1
    x21, y21, h2, w2 = rec2
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2
    # (x1, y1): left-upper point, (x2, y2): right-lower point
    x1 = max(x11, x21)
    y1 = max(y11, y21)
    x2 = min(x12, x22)
    y2 = min(y12, y22)
    h, w = y2 - y1, x2 - x1
    if h < 0 or w < 0:
        return 0
    else:
        s1 = h1 * w1
        s2 = h2 * w2
        s3 = h * w
        iou = s3 / (s1 + s2 - s3)
        return iou


def refresh_folder(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        os.makedirs(dir)


def return_label_matrix_rect(name):
    # label matrix of rectangles
    name = name.lower()
    assert name in assembly_list
    if name == 'ekedalen':
        label_mtx_rect = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0])
        return label_mtx_rect
    if name == 'gamleby':
        label_mtx_rect = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
        return label_mtx_rect
    if name == 'idolf':
        label_mtx_rect = np.array([0, 1, 1, 1, 0, 0, 1, 2, 2])
        return label_mtx_rect
    if name == 'ingatorp':
        label_mtx_rect = np.array([1, 3, 0, 1, 2, 2, 2, 0, 0, 2, 0, 1])
        return label_mtx_rect
    if name == 'ingolf':
        label_mtx_rect = np.array([0, 0, 0, 0, 0, 0])
        return label_mtx_rect
    if name == 'ivar':
        label_mtx_rect = np.array([0, 0, 0, 0, 2, 0])
        return label_mtx_rect
    if name == 'kaustby':
        label_mtx_rect = np.array([1, 0, 0, 0, 0, 2])
        return label_mtx_rect
    if name == 'lerhamn':
        label_mtx_rect = np.array([0, 1, 0, 2, 0, 0])
        return label_mtx_rect
    if name == 'nordmyra':
        label_mtx_rect = np.array([2, 1, 0, 0, 0])
        return label_mtx_rect
    if name == 'nordviken':
        label_mtx_rect = np.array([0, 0, 0, 1, 0, 1, 0])
        return label_mtx_rect
    if name == 'norraryd':
        label_mtx_rect = np.array([1, 0])
        return label_mtx_rect
    if name == 'norrnas':
        label_mtx_rect = np.array([0, 0, 2, 0, 2, 0])
        return label_mtx_rect
    if name == 'stefan':
        label_mtx_rect = np.array([0, 1, 0, 0, 0, 0, 0, 2, 0])
        return label_mtx_rect
    if name == 'stefan_9':
        label_mtx_rect = np.array([1, 0, 0, 0, 2, 0])
        return label_mtx_rect
    if name == 'input':
        label_mtx_rect = np.array([
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
            0, 0, 2, 0, 0, 3, 0, 1, 1, 0,
            3, 2, 0, 0, 0, 0, 0, 0, 2, 1,
            0, 0, 2, 1, 2, 2, 3, 1, 2, 0,
            0, 1
        ])
        return label_mtx_rect
    if name == 'mission1':
        label_mtx_rect = np.array([
            2, 0, 0, 3, 0, 1, 1, 0, 3, 2,
            0, 0, 0, 0, 0, 0, 2, 1, 0, 0,
            2, 1, 2, 2, 3, 1, 2, 0, 0, 1
        ])
        return label_mtx_rect
    if name == 'mission2':
        label_mtx_rect = np.array([
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
            0, 0
        ])
        return label_mtx_rect


def return_label_matrix_circle(name):
    # label matrix of circles
    name = name.lower()
    assert name in assembly_list
    if name == 'ekedalen':
        label_mtx_circle = np.array([1, 3, 0, 3, 3, 0, 3, 0, 1])
        return label_mtx_circle
    if name == 'gamleby':
        label_mtx_circle = np.array([1, 2, 2, 1, 1, 1, 0, 1, 0, 1])
        return label_mtx_circle
    if name == 'idolf':
        label_mtx_circle = np.array([1, 1, 0, 2, 1, 1, 0, 1, 1])
        return label_mtx_circle
    if name == 'ingatorp':
        label_mtx_circle = np.array([1, 1, 2, 2, 3, 1, 0, 2, 2, 0, 1, 0])
        return label_mtx_circle
    if name == 'ingolf':
        label_mtx_circle = np.array([3, 3, 4, 3, 1, 1])
        return label_mtx_circle
    if name == 'ivar':
        label_mtx_circle = np.array([1, 1, 1, 1, 0, 1])
        return label_mtx_circle
    if name == 'kaustby':
        label_mtx_circle = np.array([2, 0, 1, 1, 2, 2])
        return label_mtx_circle
    if name == 'lerhamn':
        label_mtx_circle = np.array([1, 2, 0, 1, 1, 1])
        return label_mtx_circle
    if name == 'nordmyra':
        label_mtx_circle = np.array([1, 1, 1, 1, 1])
        return label_mtx_circle
    if name == 'nordviken':
        label_mtx_circle = np.array([1, 2, 1, 1, 1, 0, 2])
        return label_mtx_circle
    if name == 'norraryd':
        label_mtx_circle = np.array([1, 1])
        return label_mtx_circle
    if name == 'norrnas':
        label_mtx_circle = np.array([3, 4, 2, 3, 4, 2])
        return label_mtx_circle
    if name == 'stefan':
        label_mtx_circle = np.array([1, 1, 1, 1, 1, 1, 2, 0, 2])
        return label_mtx_circle
    if name == 'stefan_9':
        label_mtx_circle = np.array([1, 1, 1, 2, 0, 3])
        return label_mtx_circle
    if name == 'input':
        label_mtx_circle = np.array([
            0, 1, 1, 1, 0, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 2, 1, 1, 1, 1, 2, 1, 1, 1,
            1, 1, 2, 1, 1, 2, 1, 0, 1, 1,
            1, 1
        ])
        return label_mtx_circle
    if name == 'mission1':
        label_mtx_circle = np.array([
            1, 1, 1, 1, 1, 1, 1, 1, 0, 2,
            1, 1, 1, 1, 2, 1, 1, 1, 1, 1,
            2, 1, 1, 2, 1, 0, 1, 1, 1, 1
        ])
        return label_mtx_circle
    if name == 'mission2':
        label_mtx_circle = np.array([
            0, 1, 1, 1, 0, 1, 1, 1, 0, 1,
            1, 1
        ])
        return label_mtx_circle


def return_label_matrix_serial(name):
    # label matrix of rectangles
    name = name.lower()
    assert name in assembly_list
    if name == 'ekedalen':
        label_mtx_num = np.array([2, 2, 0, 2, 2, 0, 2, 0, 1])
        return label_mtx_num
    if name == 'gamleby':
        label_mtx_num = np.array([1, 2, 4, 1, 2, 1, 0, 1, 0, 1])
        return label_mtx_num
    if name == 'idolf':
        label_mtx_num = np.array([0, 4, 0, 2, 3, 2, 0, 4, 2])
        return label_mtx_num
    if name == 'ingatorp':
        label_mtx_num = np.array([2, 2, 1, 2, 2, 1, 0, 3, 4, 0, 1, 0])
        return label_mtx_num
    if name == 'ingolf':
        label_mtx_num = np.array([2, 2, 4, 2, 1, 1])
        return label_mtx_num
    if name == 'ivar':
        label_mtx_num = np.array([1, 2, 2, 2, 0, 1])
        return label_mtx_num
    if name == 'kaustby':
        label_mtx_num = np.array([2, 0, 2, 2, 0, 1])
        return label_mtx_num
    if name == 'lerhamn':
        label_mtx_num = np.array([2, 4, 0, 2, 2, 1])
        return label_mtx_num
    if name == 'nordmyra':
        label_mtx_num = np.array([0, 2, 2, 2, 2])
        return label_mtx_num
    if name == 'nordviken':
        label_mtx_num = np.array([1, 2, 1, 2, 2, 0, 1])
        return label_mtx_num
    if name == 'norraryd':
        label_mtx_num = np.array([2, 1])
        return label_mtx_num
    if name == 'norrnas':
        label_mtx_num = np.array([2, 2, 3, 2, 3, 1])
        return label_mtx_num
    if name == 'stefan':
        label_mtx_num = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1])
        return label_mtx_num
    if name == 'stefan_9':
        label_mtx_num = np.array([1, 1, 1, 0, 0, 2])
        return label_mtx_num
    if name == 'input':
        label_mtx_num = np.array([
            1, 1, 3, 1, 2, 0, 1, 2, 2, 2,
            1, 1, 2, 2, 2, 1, 1, 1, 1, 6,
            5, 2, 3, 1, 2, 2, 2, 1, 2, 2,
            1, 3, 2, 1, 1, 4, 1, 4, 2, 2,
            2, 3])
        return label_mtx_num
    if name == 'mission1':
        label_mtx_num = np.array([
            2, 2, 2, 2, 1, 1, 1, 6, 5, 2,
            3, 1, 2, 2, 2, 1, 2, 2, 1, 3,
            2, 1, 1, 4, 1, 4, 2, 2, 2, 3])
        return label_mtx_num
    if name == 'mission2':
        label_mtx_num = np.array([
            1, 1, 3, 1, 2, 0, 1, 2, 2, 2,
            1, 1
        ])
        return label_mtx_num


def return_label_matrix_mult(name):
    # label matrix of rectangles
    name = name.lower()
    assert name in assembly_list
    if name == 'ekedalen':
        label_mtx_mult = np.array([1, 1, 0, 1, 1, 0, 1, 0, 1])
        return label_mtx_mult
    if name == 'gamleby':
        label_mtx_mult = np.array([1, 2, 2, 1, 1, 1, 0, 1, 0, 1])
        return label_mtx_mult
    if name == 'idolf':
        label_mtx_mult = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1])
        return label_mtx_mult
    if name == 'ingatorp':
        label_mtx_mult = np.array([1, 1, 2, 1, 2, 1, 0, 2, 2, 0, 1, 0])
        return label_mtx_mult
    if name == 'ingolf':
        label_mtx_mult = np.array([2, 1, 3, 1, 1, 1])
        return label_mtx_mult
    if name == 'ivar':
        label_mtx_mult = np.array([1, 1, 1, 1, 0, 1])
        return label_mtx_mult
    if name == 'kaustby':
        label_mtx_mult = np.array([2, 0, 1, 1, 0, 1])
        return label_mtx_mult
    if name == 'lerhamn':
        label_mtx_mult = np.array([1, 2, 0, 1, 1, 1])
        return label_mtx_mult
    if name == 'nordmyra':
        label_mtx_mult = np.array([1, 1, 1, 1, 1])
        return label_mtx_mult
    if name == 'nordviken':
        label_mtx_mult = np.array([1, 2, 1, 1, 1, 0, 1])
        return label_mtx_mult
    if name == 'norraryd':
        label_mtx_mult = np.array([1, 1])
        return label_mtx_mult
    if name == 'norrnas':
        label_mtx_mult = np.array([2, 2, 2, 2, 2, 1])
        return label_mtx_mult
    if name == 'stefan':
        label_mtx_mult = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0])
        return label_mtx_mult
    if name == 'stefan_9':
        label_mtx_mult = np.array([1, 1, 1, 0, 0, 1])
        return label_mtx_mult
    if name == 'input':
        label_mtx_mult = np.array([
            1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 2, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
            1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
            1, 1
        ])
        return label_mtx_mult
    if name == 'mission1':
        label_mtx_mult = np.array([
            1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 2, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 1, 1, 1, 1, 1, 1
        ])
        return label_mtx_mult
    if name == 'mission2':
        label_mtx_mult = np.array([
            1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
            1, 1
        ])
        return label_mtx_mult


def ten_print(arr):
    arr = arr.astype(np.int)
    N = len(arr)
    M = N // 10
    for m in range(M):
        print(arr[10 * m:10 * (m + 1)])
    R = int(N % 10)
    print(arr[-R:])
