import os
import glob
import time
import cv2
import json
import numpy as np
import platform
from pathlib import Path


def create_scenes(args):
    """ save scene_models.json"""

    # specify min, max number of cads in single scene
    min_num_cad_per_scene = int(args.num_cad_range.split(',')[0].strip())
    max_num_cad_per_scene = int(args.num_cad_range.split(',')[1].strip())

    # find radii of cad objects
    center_model(args)
    def files(path, ext=''):
        temp = sorted(glob.glob(os.path.join(path, '*' + ext)))
        return [x for x in temp if os.path.isfile(x)]
    cad_adrs = files(args.cad_path, '.obj')
    def get_obj_radius(cad_adr):
        with open(cad_adr, 'r') as f:
            xyz = []
            lines = f.readlines()
            for line in lines:
                split = line.split(' ')
                # vertices
                if split[0] == 'v':
                    xyz.append([float(x.rstrip('\n')) for x in split[1:]])
            xyz = np.array(xyz, dtype=np.float32)
        xyz /= args.scale
        radius = np.max(np.linalg.norm(xyz, axis=1))
        return radius
    cad_radii = [get_obj_radius(x) for x in cad_adrs]

    # save & read cam_data.json
    mute = args.mute
    H = args.rendering_size.split(',')[0].strip()
    W = args.rendering_size.split(',')[1].strip()
    res_ratio = min(int(H)/int(W), int(W)/int(H))

    flag = ' -views_path ' + args.views_path + ' -views_gray_black_path ' + args.views_gray_black_path + ' -views_gray_black_occlu_path ' + args.views_gray_black_occlu_path\
        + ' -cad_path ' + args.cad_path + ' -point_cloud_path ' + args.point_cloud_path + ' -krt_image_path ' + args.krt_image_path + ' -H ' + H + ' -W ' + W + ' -scale ' + str(args.scale)\
        + ' -return_camera_intrinsics True' + ' -intermediate_results_path ' + args.intermediate_results_path
    if mute:
        os.system(args.blender + ' -b -P ./render_views.py > ./render_stdout.txt -- ' + flag + ' 2>&1')
    else:
        os.system(args.blender + ' -b -P ./render_views.py -- ' + flag)
    with open('./cam_data.json', 'r') as f:
        load_dict = json.load(f)
    angle = np.arctan(load_dict['sensor_size_in_mm'] * 0.5 / load_dict['f_in_mm'])
    angle_narrow = np.arctan(res_ratio * load_dict['sensor_size_in_mm'] * 0.5 / load_dict['f_in_mm'])


    # create scene data
    hyper = [0.8, 0.5, 0.9, 0.9]
    scene_dict = {}
    for scene_index in range(args.num_scene):
        scene_name = str(scene_index).zfill(6)
        scene_dict[scene_name] = []
        num_obj = np.random.randint(min_num_cad_per_scene, max_num_cad_per_scene + 1)
        for _ in range(num_obj):
            obj_dict = {}
            # select random cad
            obj_index = np.random.randint(0, len(cad_radii))
            obj_dict['obj_id'] = obj_index
            # random location
            max_z = hyper[0] * load_dict['radius'] - cad_radii[obj_index] / np.sin(angle_narrow)
            min_z = -hyper[1] * load_dict['radius']
            loc_z = np.random.uniform(min_z, max_z)
            max_y = hyper[2] * (load_dict['radius'] - loc_z) * np.tan(angle_narrow) - cad_radii[obj_index] / np.cos(angle_narrow)
            loc_y = np.random.uniform(-max_y, max_y)
            max_x = hyper[3] * (load_dict['radius'] - loc_z) * np.tan(angle) - cad_radii[obj_index] / np.cos(angle)
            loc_x = np.random.uniform(-max_x, max_x)
            loc = [loc_x, loc_y, loc_z]
            obj_dict['location_XYZ'] = [int(x * args.scale) for x in loc]
            # random rotation
            rot_theta = np.pi * np.random.uniform(0, 1)
            rot_phi = 2 * np.pi * np.random.uniform(0, 1)
            axis_rot = [np.sin(rot_theta) * np.cos(rot_phi), np.sin(rot_theta) * np.sin(rot_phi), np.cos(rot_theta)]
            angle_rot = [int(360 * np.random.uniform(0, 1))]
            axis_angle = angle_rot + [round(x, 4) for x in axis_rot]
            obj_dict['axisangle_WXYZ'] = axis_angle
            # append
            scene_dict[scene_name].append(obj_dict)
    with open('./scene_models.json', 'w') as f:
        json.dump(scene_dict, f, sort_keys=True, indent=2)


def stl_to_obj(args):
    """ read .STL files in cad_path and save .obj files (FreeCAD) """
    print('\nConverting CAD models to .obj ...')
    # mute freecad output
    mute = True  
    # flags to send to freecad python script
    flag = '-cad_ext ' + '.STL' + ' -cad_path ' + args.cad_path
    # run freecad with python script
    if mute:
        os.system('freecadcmd ./convert_cad_format.py > ./render_stdout.txt -- ' + flag + ' 2>&1')
    else:
        os.system('freecadcmd ./convert_cad_format.py -- ' + flag)


def center_model(args):
    """ read .obj files in cad_path and save after centering (Blender) """
    print('\nCentering CAD models ...')
    # mute Blender output
    mute = True  
    # flags to send to the Blender python script
    flag = ' -cad_path ' + args.cad_path  # cad file directory
    # Run Blender with python script
    if mute:
        os.system(args.blender + ' -b -P ./center_cad_model.py > ./render_stdout.txt -- ' + flag + ' 2>&1')
    else:
        os.system(args.blender + ' -b -P ./center_cad_model.py -- ' + flag)


def create_pointcloud(args, cad_basenames):
    """ create point cloud(.asc) from .obj files (CloudCompare) """
    # center model before making pointcloud
    exist = True
    for cad_basename in cad_basenames:
        if not os.path.exists(args.point_cloud_path + '/' + os.path.splitext(cad_basename)[0] + '_SAMPLED_POINTS.asc'):
            exist = False
    if exist:
        return
    center_model(args)
    # create point clouds
    NUM_POINTS = 100000
    print('\nCreating Point Clouds ...')
    for cad_basename in cad_basenames:
        if not os.path.exists(args.point_cloud_path + '/' + os.path.splitext(cad_basename)[0] + '_SAMPLED_POINTS.asc'):
            os.system(args.cloudcompare + ' -SILENT -c_EXPORT_FMT ASC -ADD_HEADER -ADD_PTS_COUNT -NO_TIMESTAMP -O '+ os.path.join(args.cad_path, cad_basename) + ' -SAMPLE_MESH POINTS ' + str(NUM_POINTS) + ' > ./render_stdout.txt')
    # move point clouds
    if platform.system() == 'Windows':
        os.system('move /y ' + str(Path(args.cad_path) / '*.asc') + ' ' + str(Path(args.point_cloud_path)))
    else:
        os.system('mv ' + args.cad_path + '/*.asc ' + args.point_cloud_path + '/')


def create_rendering(args, cad_basenames):
    """ create renderings from .obj files (Blender) """
    
    # mute Blender output
    mute = args.mute

    # flags to send to the Blender python script
    H = args.rendering_size.split(',')[0].strip()
    W = args.rendering_size.split(',')[1].strip()
    single_scene_mode = 'True' if len(args.scene_name) else 'False'
    scene_name = args.scene_name if len(args.scene_name) else 'dummy'

    flag = ' -views_path ' + args.views_path + ' -views_gray_black_path ' + args.views_gray_black_path + ' -views_gray_black_occlu_path ' + args.views_gray_black_occlu_path\
        + ' -cad_path ' + args.cad_path + ' -point_cloud_path ' + args.point_cloud_path + ' -krt_image_path ' + args.krt_image_path + ' -H ' + H + ' -W ' + W + ' -scale ' + str(args.scale)\
        + ' -intermediate_results_path ' + args.intermediate_results_path + ' -scene_name ' + scene_name + ' -single_scene_mode ' + single_scene_mode + ' -thickness 2.0 '\
        + ' -json ' + args.json
    
    # Run Blender with python script
    if mute:
        os.system(args.blender + ' -b -P ./render_views.py > ./render_stdout.txt -- ' + flag + ' 2>&1')
    else:
        os.system(args.blender + ' -b -P ./render_views.py -- ' + flag)


def postprocessing(args):
    """ line weighting. save output (mask, mask_visib, rgb) """

    def ensure_paths(paths):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    # read scene_models.json
    with open('./' + args.json + '.json', 'r') as f:
        scene_dict = json.load(f)

    # read number of camera orientations
    CAMERA_ORIENTATIONS = int(np.loadtxt('./num_cam.txt'))

    print('POST PROCESSING : total scene ' + str(len(scene_dict)) + ' total camera_orientations ' + str(CAMERA_ORIENTATIONS))
    for scene_name in sorted(scene_dict):
        # create folders
        mask_path = os.path.join(args.output_path, scene_name, 'mask')
        mask_visib_path = os.path.join(args.output_path, scene_name, 'mask_visib')
        rgb_path = os.path.join(args.output_path, scene_name, 'rgb')
        ensure_paths([mask_path, mask_visib_path, rgb_path])

        
        for camera_index in range(CAMERA_ORIENTATIONS):
            print('postprocessing scene ' + scene_name + ' camera_index ' + str(camera_index))
            # load scene filenames
            views_gray_black_adrs = sorted(glob.glob(os.path.join(args.views_gray_black_path, scene_name, str(camera_index).zfill(6) + '*.png')))
            views_gray_black_occlu_adrs = sorted(glob.glob(os.path.join(args.views_gray_black_occlu_path, scene_name, str(camera_index).zfill(6) + '*.png')))
            views_adrs = sorted(glob.glob(os.path.join(args.views_path, scene_name, str(camera_index).zfill(6) + '*.png')))
            assert len(views_adrs) == 1, views_adrs
            view_adr = views_adrs[0]

            # load images
            masks = [cv2.imread(x) for x in views_gray_black_adrs]
            mask_visibs = [cv2.imread(x) for x in views_gray_black_occlu_adrs]
            rgb = cv2.imread(view_adr)

            # create mask
            def create_mask(mask):
                threshold = 1
                mask = 0 * (mask <= threshold) + 255 * (mask > threshold)
                mask = mask.astype(np.uint8)
                return mask
            masks = [create_mask(x) for x in masks]
            # save label


            # create mask_visib
            def create_mask_visib(mask):
                threshold = 254
                mask = 255 * (mask <= threshold) + 0 * (mask > threshold)
                mask = mask.astype(np.uint8)
                return mask
            mask_visibs = [create_mask_visib(x) for x in mask_visibs]

            # create rgb
            for mask, mask_visib in zip(masks, mask_visibs):
                gray = np.amax(mask, axis=2)
                gray_v = np.amax(mask_visib, axis=2)
                _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                _, contours_v, hierarchy_v = cv2.findContours(gray_v, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                blank = np.ones_like(gray) * 255
                blank_v = np.ones_like(gray_v) * 255
                try:
                    for i in range(len(hierarchy[0])):
                        blank = cv2.drawContours(blank, contours, i, color=0, thickness=9)
                except:
                    pass
                try:
                    for i in range(len(hierarchy_v[0])):
                        blank_v = cv2.drawContours(blank_v, contours_v, i, color=0, thickness=9)
                except:
                    pass
                blank_both = blank.astype(np.int) + blank_v.astype(np.int)
                blank_both = np.stack((blank_both, blank_both, blank_both), axis=-1)
                rgb = 0 * (blank_both == 0) + rgb * (blank_both > 0)
                rgb = rgb.astype(np.uint8)

            # save output
            mask_adrs = [os.path.join(mask_path, os.path.basename(x)) for x in views_gray_black_adrs]
            for img, adr in zip(masks, mask_adrs):
                cv2.imwrite(adr, img)
            mask_visib_adrs = [os.path.join(mask_visib_path, os.path.basename(x)) for x in views_gray_black_occlu_adrs]
            for img, adr in zip(mask_visibs, mask_visib_adrs):
                cv2.imwrite(adr, img)
            rgb_adr = os.path.join(rgb_path, os.path.basename(view_adr))
            cv2.imwrite(rgb_adr, rgb)


def create_labels(args):
    """save labels.json

    {"scene_name":
        {"rgb_name":
            [obj0_labels, obj1_labels, ...]
        }
    }
    
    obj_labels = {
        "bbox_obj" : bounding box of object
        "bbox_visib" : bounding box if visible part of object
        "obj_id" : object index in cad_path
        "obj_type" : New, Mid
    }
    """

    # read krt.json
    with open(args.intermediate_results_path + '/krt.json', 'r') as f:
        krt_dict = json.load(f)

    label_dict = {}

    def files(path, ext=''):
        temp = sorted(glob.glob(os.path.join(path, '*' + ext)))
        return [x for x in temp if os.path.isfile(x)]

    def directories(path):
        temp = sorted(glob.glob(os.path.join(path, '*')))
        return [x for x in temp if os.path.isdir(x)]

    def bbox(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        try:
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            x, y, w, h = cmin.item(), rmin.item(), (cmax-cmin).item(), (rmax-rmin).item()
            return [x, y, w, h]
        except:
            return []

    cad_adrs = files(args.cad_path)
    cad_basenames = [os.path.basename(x) for x in cad_adrs]

    # scenes
    scene_adrs = directories(args.output_path)
    scene_names = [os.path.basename(x) for x in scene_adrs]
    with open('./scene_models.json', 'r') as f:
        scene_models = json.load(f)
    for scene_adr, scene_name in zip(scene_adrs, scene_names):
        label_dict[scene_name] = {}
        # scene file addresses
        mask_path = scene_adr + '/mask'
        mask_visib_path = scene_adr + '/mask_visib'
        rgb_path = scene_adr + '/rgb'
        mask_adrs = files(mask_path)
        mask_visib_adrs = files(mask_visib_path)
        rgb_adrs = files(rgb_path)
        for rgb_name in [os.path.basename(x).split('.')[0] for x in rgb_adrs]:
            print('scene_name : {}, rgb_name : {}'.format(scene_name, rgb_name))
            label_dict[scene_name][rgb_name] = []
            rgb_mask_adrs = [x for x in mask_adrs if os.path.basename(x).startswith(rgb_name)]
            rgb_mask_visib_adrs = [x for x in mask_visib_adrs if os.path.basename(x).startswith(rgb_name)]
            rgb_masks = [cv2.imread(x) for x in rgb_mask_adrs]
            rgb_mask_visibs = [cv2.imread(x) for x in rgb_mask_visib_adrs]
            assert len(scene_models[scene_name]) == len(rgb_masks), '{} != {}'.format(len(scene_models[scene_name]), len(rgb_masks))
            for i in range(len(rgb_masks)):
                bbox_obj = bbox(rgb_masks[i])
                bbox_visib = bbox(rgb_mask_visibs[i])
                obj_id = scene_models[scene_name][i]['obj_id']
                obj_type = 'New' if 'part' in cad_basenames[obj_id] else 'Mid'
                rgb_dict = {}
                rgb_dict['bbox_obj'] = bbox_obj
                rgb_dict['bbox_visib'] = bbox_visib
                rgb_dict['obj_id'] = obj_id
                rgb_dict['obj_type'] = obj_type
                krt_temp_dict = krt_dict[scene_name][rgb_name][i]
                rgb_dict['cam_K'] = krt_temp_dict['cam_K']
                rgb_dict['cam_R_m2c'] = krt_temp_dict['cam_R_m2c']
                rgb_dict['cam_t_m2c'] = krt_temp_dict['cam_t_m2c']
                label_dict[scene_name][rgb_name].append(rgb_dict)
    with open('./' + args.label_json + '.json', 'w') as f:
        json.dump(label_dict, f, sort_keys=True, indent=2)
        



    

