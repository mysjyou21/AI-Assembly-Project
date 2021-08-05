import os, glob
import sys
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage

def calculate_dist_pair(fastenerInfo, p1_hole, p2_hole):
    fastener_x = fastenerInfo[1][0]
    fastener_y_up = fastenerInfo[1][1]
    fastener_y_down = fastenerInfo[2][1]

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

def calculate_dist_point(fastenerInfo, p_hole):

    fastener_x = fastenerInfo[2][0]
    fastener_y_down = fastenerInfo[2][1]

    p_hole_x = p_hole[1]
    p_hole_y = p_hole[2]

    dist = ((fastener_x - p_hole_x)**2 + (fastener_y_down - p_hole_y)**2)**(1/2)

    return dist

def change_to_point_matching(recomb):
    count = 0
    idx_list = list()
    for i, comb in enumerate(recomb):
        f_coord_zip = sorted(comb[0], key=lambda x:x[1])
        h_coord_zip = sorted(comb[1], key=lambda x:x[1])
        fu = f_coord_zip[0]
        fd = f_coord_zip[1]
        hu = h_coord_zip[0]
        hd = h_coord_zip[1]
        d1 = ((fu[0]-hu[0])**2 + (fu[1]-hu[1])**2)**(1/2)
        d2 = ((fd[0]-hd[0])**2 + (fd[1]-hd[1])**2)**(1/2)
        if abs(d1-d2) > 150:
            count += 1
            idx_list.append(i)
    return count, idx_list