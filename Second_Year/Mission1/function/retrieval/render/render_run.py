import numpy as np
import os
import glob
import sys
import time
import cv2
import argparse
import shutil
import platform
from pathlib import Path


def stl_to_obj(args):
    """ read .STL files in cad_path and save .obj files (FreeCAD) """
    print('\nConverting CAD models to .obj ...')
    # Using freecad
    mute = True  # mute freecad output
    # flags to send to freecad python script
    flag = '-cad_ext ' + '.STL' + ' -cad_path ' + args.opt.cad_path
    # run freecad with python script
    if mute:
        os.system('freecadcmd ./function/retrieval/render/convert_cad_format.py > ./function/retrieval/render/render_stdout.txt -- ' + flag + ' 2>&1')
    else:
        os.system('freecadcmd ./function/retrieval/render/convert_cad_format.py -- ' + flag)


def center_model(args):
    """ read .obj files in cad_path and save after centering (Blender) """
    cad_path = args.opt.cad_path
    print('\nCentering CAD models ...')
    mute = True  # mute Blender output
    # flags to send to the Blender python script
    flag = ' -cad_path ' + cad_path  # cad file directory
    # Run Blender with python script
    if mute:
        os.system('blender -b -P ./function/retrieval/render/center_cad_model.py > ./function/retrieval/render/render_stdout.txt -- ' + flag + ' 2>&1')
    else:
        os.system('blender -b -P ./function/retrieval/render/center_cad_model.py -- ' + flag)


def create_rendering(args, list_added_obj, render_type):
    """ create renderings from .obj files (Blender) """
    #======================
    #    CONFIGURATION
    #======================
    assert render_type in ['views', 'views_gray_black']

    cad_path = args.opt.cad_path  # cad file directory
    point_cloud_path = args.opt.point_cloud_path  # point cloud(.asc) directory
    retrieval_views_path = args.opt.retrieval_views_path  # rendering output directory for retrieval network(VIEWS_GRAY)
    pose_views_path = args.opt.pose_views_path  # rendering output directory for pose network(VIEWS)
    pose_krt_path = args.opt.pose_krt_path  # K, Rt output directory
    pose_krt_imgs_path = args.opt.pose_krt_imgs_path  # K, Rt images output directory for validating K, Rt outputs

    mute = True  # mute Blender output

    #========================
    #       RENDER
    #========================
    # set rendering output directory
    if render_type == 'views_gray_black':  # rendering type : gray body, black background
        render_output_path = retrieval_views_path
    elif render_type == 'views':  # rendering type : lines
        render_output_path = pose_views_path
    else:
        raise Exception('wrong render type')

    # create rendering directory
    if not os.path.exists(render_output_path):
        os.makedirs(render_output_path)
    if not os.path.exists(pose_krt_path):
        os.makedirs(pose_krt_path)
    if not os.path.exists(pose_krt_imgs_path):
        os.makedirs(pose_krt_imgs_path)

    # skip rendering for already existing CAD name folders in rendering output directory
    output_subfolders = sorted(glob.glob(os.path.join(render_output_path + '/*')))
    output_subfolders = [os.path.basename(s) for s in output_subfolders]
    input_subfolders = sorted(glob.glob(os.path.join(cad_path + '/*')))
    input_subfolders = [os.path.basename(s) for s in input_subfolders]
    input_subfolders = [os.path.splitext(s)[0] for s in input_subfolders]
    intersection_ = sorted(list(set(output_subfolders) & set(input_subfolders)))
    intersection = []
    for inter in intersection_:
        temp_num = len(glob.glob(os.path.join(render_output_path, inter, '*.png')))
        if temp_num == 48:
            intersection.append(inter)
        else:
            os.system('rm -r '+os.path.join(render_output_path, inter))
    print('\nRendering : {}'.format(render_type))
    print('SKIP for already existing folders : {}'.format(intersection))

    # flags to send to the Blender python script
    list_added_obj_str = ''
    for added_obj in list_added_obj:
        list_added_obj_str += added_obj + ' '
    flag = '-render_type ' + render_type + ' -list_added_obj ' + list_added_obj_str + ' -render_output_path '\
        + render_output_path + ' -cad_path ' + cad_path + ' -point_cloud_path ' + point_cloud_path + ' -krt_path ' + pose_krt_path\
        + ' -krt_imgs_path ' + pose_krt_imgs_path
    # Run Blender with python script
    if mute:
        os.system('blender -b -P ./function/retrieval/render/render_views.py > ./function/retrieval/render/render_stdout.txt -- ' + flag + ' 2>&1')
    else:
        os.system('blender -b -P ./function/retrieval/render/render_views.py -- ' + flag)

    # OPTIONAL: additional operation for VIEWS rendering
    # rendered VIEWS are currently 448 x 448 -> resize to 224 x 224
    if render_type == 'views':
        H, W = 224, 224
        img_adrs = []
        for root, dirs, files in os.walk(render_output_path):
            for file in files:
                img_adrs.append(os.path.join(root, file))
        for img_adr in img_adrs:
            if platform.system() == 'Windows':
                img_adr = str(Path(img_adr))
            else:
                pass
            img = cv2.imread(img_adr)
            option = True
            if option:
                # clip border
                non_zero = np.nonzero(255 - img)
                y_min = np.min(non_zero[0])
                y_max = np.max(non_zero[0])
                x_min = np.min(non_zero[1])
                x_max = np.max(non_zero[1])
                # crop image
                img = img[y_min:y_max + 1, x_min:x_max + 1]
                long_side = np.max(img.shape)
                # make 150 x 150
                ratio = 150 / long_side
                img = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
                pad_left = int(np.ceil((150 - img.shape[1]) / 2))
                pad_right = int(np.floor((150 - img.shape[1]) / 2))
                pad_top = int(np.ceil((150 - img.shape[0]) / 2))
                pad_bottom = int(np.floor((150 - img.shape[0]) / 2))
                img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, None, [255, 255, 255])
                # make 224 x 224
                img = cv2.copyMakeBorder(img, 37, 37, 37, 37, cv2.BORDER_CONSTANT, None, [255, 255, 255])
            else:
                # resize
                img = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
            cv2.imwrite(img_adr, img)

    print('rendering complete')


def create_pointcloud(args, list_added):
    """ create point cloud(.asc) from .obj files (CloudCompare) """
    NUM_POINTS = 10000
    print('\nCreating Point Clouds ...')
    if not os.path.exists(args.opt.point_cloud_path):
        os.makedirs(args.opt.point_cloud_path)
    for cad_filename in list_added:
        if not os.path.exists(args.opt.point_cloud_path + '/' + os.path.splitext(cad_filename)[0] + '_SAMPLED_POINTS.asc'):
            command = 'CloudCompare -SILENT -c_EXPORT_FMT ASC -ADD_HEADER -ADD_PTS_COUNT -NO_TIMESTAMP -O '+ os.path.join(args.opt.cad_path, cad_filename) + ' -SAMPLE_MESH POINTS ' + str(NUM_POINTS) + ' > ./function/retrieval/render/render_stdout.txt'
            os.system(command)
    if platform.system() == 'Windows':
        os.system('move /y ' + str(Path(args.opt.cad_path) / '*.asc') + ' ' + str(Path(args.opt.point_cloud_path)))
    else:
        os.system('mv ' + args.opt.cad_path + '/*.asc ' + args.opt.point_cloud_path + '/')
