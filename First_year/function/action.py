import numpy as np
from scipy import ndimage
import cv2 as cv
from collections import OrderedDict

def action_checker(I, actionpath, image_order=0, highlight=None):
    # TODO: 은지
    """
    classify which action is in image I (no action = 0)
    :param I: current stefan cut image
    :param highlight: (x, y, w, h) list of location of highlight bounding box
    :return: loc : (c, x, y, w, h) list of location of action c
    """
    H, W, C = I.shape
    order = 1
    loc = []
    gray = I[:, :, 0]
    inv = 255 - gray
    # erosion to inverted gray-scale image with diamond-shaped-StructuringElement
    r = int(4 * W / 1166)
    r_d = int(2 * W / 1166)
    num_w, num_h = 200 * W / 1166, 200 * W / 1166
    num_x, num_y = 0, 0

    def diamond_se(r):
        kernel = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.uint8)
        for i in range(0, r):
            kernel[i, r - i:r + i + 1] = 1
            kernel[2 * r - i, r - i:r + i + 1] = 1
        kernel[r, :] = 1
        return kernel

    # to set position of actions
    kernel_pos = diamond_se(r)
    erode_pos = cv.erode(inv, kernel_pos, iterations=1)
    # thresholding eroded image
    bin_pos = erode_pos > 200
    # labeling thresholded image and original image
    bin_labeled_array_pos, num_feature = ndimage.label(bin_pos)
    bin_objects_pos = ndimage.find_objects(bin_labeled_array_pos)

    # to get detailed shape of actions
    kernel_detail = diamond_se(r_d)
    erode_detail = cv.erode(inv, kernel_detail, iterations=1)
    bin_detail = erode_detail > 200
    bin_labeled_array_detail, _ = ndimage.label(bin_detail)
    bin_objects_detail = ndimage.find_objects(bin_labeled_array_detail)

    prev_num = [-1]
    prev_pos = [[-1000, -1000]]
    x, y, h, w, c = 0, 0, 0, 0, 0
    num_xs, num_ys, num_n = 0, 0, 0

    def cut_num(pos_x, pos_y, num):
        """ verify this part is CUT NUMBER or not """
        shape = bin_objects_detail[num - 1]
        x_ = shape[1].start
        y_ = shape[0].start
        w_ = shape[1].stop - x_
        h_ = shape[0].stop - y_
        candidate = x_ + w_ < num_w or y_ + h_ < num_h or (W / 2 < x_ and w_ < num_w and H / 2 < y_ and h_ < num_h)
        is_num = abs(w_ - num_x) < 10 and abs(h_ - num_y) < 8

        return candidate * is_num

    for i in range(num_feature):
        # pos = bin_objects_pos[i]
        pos_y, pos_x = np.where(bin_labeled_array_pos == i + 1)
        num = bin_labeled_array_detail[pos_y[0], pos_x[0]]
        numbers = len(pos_x)
        # except sequence numbers and too-weak action candidates
        if max(pos_x) >= num_w or max(pos_y) >= num_h:
            if not cut_num(pos_x, pos_y, num):
                if (numbers > 10 and num_feature > 2) or (num_feature <= 2):
                    # to remove duplicates
                    if not num in prev_num:
                        shape = bin_objects_detail[num - 1]
                        x = shape[1].start
                        y = shape[0].start
                        w = shape[1].stop - x
                        h = shape[0].stop - y
                        # to remove duplicates
                        pos_check = sum([(abs(p[0] - x) < 5 and abs(p[1] - y) < 5) for p in prev_pos])
                        # to remove 'X'-shape
                        x_check = (w / h > 1.15 or w / h < 0.89) and (0.89 > h / w or h / w > 1.11)
                        if pos_check == 0 and x_check:
                            temp_array = bin_labeled_array_detail == num
                            cropped = temp_array[y:y + h, x:x + w]

                            cv.imwrite(actionpath + '/' + '%d_%d.png' % (image_order, order),
                                       (255 * cropped).astype(np.uint8))
                            order += 1
                            I = cv.rectangle(I.astype(np.uint8), (x, y), (x + w, y + h), (0, 0, 255), 3)
                            if w != 0:
                                c = 1
                                loc.append([c, x, y, h, w])
                            prev_num.append(num)
                            prev_pos.append([x, y])
        else: # to filter CUT NUMBER cases
            shape = bin_objects_detail[num - 1]
            x = shape[1].start
            y = shape[0].start
            w = shape[1].stop - x
            h = shape[0].stop - y
            num_x = w
            num_y = h
            num_xs += num_x
            num_ys += num_y
            num_n += 1

#    if num_n > 0:
#        print('num_x', num_xs / num_n, 'num_y', num_ys / num_n, 'w', W, 'h', H, (num_xs / num_n) / W, (num_ys / num_n) / H)
    if c == 0:
        loc.append([c, x, y, h, w])
    cv.imwrite(actionpath + '/bin_%d.png' % image_order, (bin_pos * 255).astype(np.uint8))
    cv.imwrite(actionpath + '/pred_%d.png' % image_order, I)

    return loc


def map_action(circle_labs, circle_nums, act_dic):
    """ input: material numbers in a circle, output: corresponding action(s) """

    circle_action = []
    circle_num = []

    # first, collect actions of all materials in a circle
    circle_actions = []
    circle_tools = []
    for material_lab in circle_labs:
        act = act_dic[material_lab]
        circle_actions.append(act[0])
        circle_tools.append('' if len(act) == 1 else act[1])

    # apply manual rules
    if ('Cylindrical' in circle_tools) and ('Tanned' in circle_tools):
        circle_action.append('A004')
        ide = circle_tools.index('Cylindrical')
        circle_num.append(circle_nums[ide])
        circle_actions.remove(circle_actions[ide])
        circle_tools.remove(circle_tools[ide])
        circle_nums.remove(circle_nums[ide])
    if 'Nut' in circle_tools:
        if 'Hex' in circle_tools:
            circle_action.append('A012')
            # remove already appended action
            # for i in range(circle_tools.count('Nut')): # not sure
            ide = circle_tools.index('Nut')
            # find the number
            circle_num.append(circle_nums[ide])
            circle_actions.remove(circle_actions[ide])
            circle_tools.remove(circle_tools[ide])
            circle_nums.remove(circle_nums[ide])
            ide = circle_tools.index('Hex')
            circle_actions.remove(circle_actions[ide])
            circle_tools.remove(circle_tools[ide])
            circle_nums.remove(circle_nums[ide])

    # simple actions
    circle_actions_pair = list(filter(lambda x: x[0] != '0', list(zip(circle_actions, circle_nums))))
    circle_actions = [x for x, _ in circle_actions_pair]
    circle_nums = [x for _, x in circle_actions_pair]

    # process duplicates
    temp_dic = OrderedDict()
    temp_mult = OrderedDict()
    for i in range(len(circle_actions)):
        idx = circle_actions[i]
        temp_dic[idx] = True
        if idx in temp_mult:
            temp_mult[idx] += circle_nums[i]
        else:
            temp_mult[idx] = circle_nums[i]
    circle_actions = list(temp_dic.keys())  # ordered
    circle_nums = list(temp_mult.values())

    # final action results
    circle_action += circle_actions
    circle_num += circle_nums

    if len(circle_action) == 0 and len(circle_num) > 0:
        circle_action = ['A015']
        circle_num = [circle_nums[0]]

    return circle_action, circle_num
