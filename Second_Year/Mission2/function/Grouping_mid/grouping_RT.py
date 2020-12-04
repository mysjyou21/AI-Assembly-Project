import numpy as np
from itertools import permutations

def baseRT_to_midRT(base_hole_dict, mid_hole_dict, HOLE_DIST_THR=1):
    ######## step_num-1의 중간산출물의 hole 위치 및 hole pair 찾기 ##########
    base_id_list = list(base_hole_dict.keys())
    mid_id_list = list(mid_hole_dict.keys())
    mid_permu = list(permutations(mid_id_list, 2))

    ####### 중간산출물의 기본부품2의 hole을 기본부품1의 RT로 이동한 위치가 기본부품2의 RT로 기본부품2를 이동한 위치와 가까운지 확인####
    mid_RT = np.zeros([3,4])
    find_mid = False
    for i, mid_pair in enumerate(mid_permu):
        mp1, mp2 = mid_pair
        if mp1 in base_id_list and mp2 in base_id_list:
            mid_RT1 = holepair_to_RT(base_hole_dict[mp1], mid_hole_dict[mp1])
            mid_hole_2 = transform_hole(mid_RT1, mid_hole_dict[mp2])

            dist = hole_distance(mid_hole_2, base_hole_dict[mp2])
            if dist.min() < HOLE_DIST_THR:
                mid_RT = mid_RT1
                find_mid = True
                break

    ######## 중간산출물을 이루는 기본부품간의 pose가 일치하는게 없을시, 중간산출물을 이루는 아무 파트중 하나의 pose로 통일 ####
    mid_base = list(set(mid_id_list).intersection(base_id_list))
    if not find_mid and len(mid_base) > 0:
        id = mid_base[0]
        mid_RT = holepair_to_RT(base_hole_dict[id], mid_hole_dict[id])
        find_mid = True

    return mid_RT, mid_id_list, find_mid



def transform_hole(RT, hole):
    return (RT @ np.hstack([hole, np.ones([hole.shape[0], 1])]).T).T

def hole_distance(hole1, hole2):
    return np.linalg.norm(hole1 - hole2, 2, 0)

def holepair_to_RT(dst, src):
    ######## src, dst == 3xN ################
    ####### R @ src + t = dst
    dst = dst.T
    src = src.T
    src_cent = np.mean(src, 1)[:, np.newaxis]
    dst_cent = np.mean(dst, 1)[:, np.newaxis]
    A = (dst - dst_cent) @ (src - src_cent).T
    U, S, VT = np.linalg.svd(A)
    V = VT.T

    R = U @ VT
    if np.linalg.det(R) < 0:
        V[:, -1] = -V[:, -1]
        R = U @ VT
    t = np.mean(dst, 1) - R @ np.mean(src, 1)
    #t = t * scale
    RT = np.concatenate([R,t[...,np.newaxis]], 1)
    return RT
