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


def div_cut(I):
    """ divide first cut if it has materials on its top """
    H, W, C = I.shape
    gray = I[:, :, 0]
    inv = 255 - gray
    # erosion to inverted gray-scale image with diamond-shaped-StructuringElement
    r = int(6 * W / 1166)
    r_d = int(3 * W / 1166)
    num_w, num_h = W / 6, H / 5

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
    mid_point = (x[0] + x[1]) / 2
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
    width = x[2][0] * 12.2 / 10 * (x[3][0] + 6) / 6
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


def show(img, opencv=True):
    if opencv == True:
        h = img.shape[0]
        w = img.shape[1]
        if h > 1000:
            h_new = 1000
            w_new = int(w * 1000 / h)
        else:
            h_new = h
            w_new = w
        img = cv.resize(img, (w_new, h_new))
        cv.imshow('', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if len(img.shape) == 3:
            plt.imshow(img)
            plt.show()
        else:
            plt.imshow(img, cmap='gray')
            plt.show()


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
