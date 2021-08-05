""" file runs in blender """
import bpy
print(bpy.app.binary_path_python)
import os
import sys
import glob
import math
import time
import argparse
import numpy as np
import matplotlib
import cv2
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from mathutils import Matrix
from scipy.spatial.transform import Rotation as Rot

if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-views_path', default='', help='views rendering path')
    parser.add_argument('-views_gray_black_path', default='', help='views_gray_black rendering path')
    parser.add_argument('-views_gray_black_occlu_path', default='', help='views_gray_black_occlu rendering path')
    parser.add_argument('-cad_path', default='', help='cad input path')
    parser.add_argument('-point_cloud_path', default='', help='point cloud path')
    parser.add_argument('-krt_image_path', default='', help='path for saving krt images')
    parser.add_argument('-intermediate_results_path', default='', help='for saving some labels')
    parser.add_argument('-H', default='100', help='height of rendering')
    parser.add_argument('-W', default='100', help='width of rendering')
    parser.add_argument('-scale', default='100', help='cad_model = 1/Scale * cad_model (original cad is too big)')
    parser.add_argument('-thickness', default='', help='line thickness of line rendering')
    parser.add_argument('-return_camera_intrinsics', default='', help='not for rendering. for scenes generation stage')
    parser.add_argument('-moving_camera_mode', default='True', help='...')
    parser.add_argument('-view_scene_mode', default='', help='view scenes in blender GUI')
    parser.add_argument('-single_scene_mode', default='', help='render single scene')
    parser.add_argument('-scene_name', default='', help='scene_name that you wish to view')
    parser.add_argument('-json', default='scene_models', help='which scene_models.json to read...')
    args = parser.parse_known_args(argv)[0]
print('# render views.py')

D = bpy.data
C = bpy.context
O = bpy.ops

krt_dict = {}

#---------------------- settings ----------------------------------------

# resolution
C.scene.render.resolution_y = int(args.H)
C.scene.render.resolution_x = int(args.W)

def main():
    if bool(args.view_scene_mode):
        # read scene_models.json
        with open('./' + args.json + '.json', 'r') as f:
            scene_dict = json.load(f)

        # initialize scene
        init_scene()

        for scene_name in sorted(scene_dict):
            if scene_name != args.scene_name:
                continue
            else:
                init_ops(scene_dict[scene_name])
                delete_camera_and_origin()

    elif bool(args.return_camera_intrinsics):
        # for scene generation stage
        radius = camera_orientation()
        f_in_mm, sensor_size_in_mm = blender_K()
        save_dict = {}
        save_dict['radius'] = radius
        save_dict['f_in_mm'] = f_in_mm
        save_dict['sensor_size_in_mm'] = sensor_size_in_mm
        with open('./cam_data.json', 'w') as f:
            json.dump(save_dict, f)
    else:
        tic_total = time.time()
        
        # read scene_models.json
        with open('./' + args.json + '.json', 'r') as f:
            scene_dict = json.load(f)

        # initialize scene
        init_scene()

        for scene_name in sorted(scene_dict):
            if args.single_scene_mode == 'True':
                if not scene_name == args.scene_name:
                    continue
            tic_part = time.time()
            render(scene_name, scene_dict[scene_name])
            render_individual(scene_name, scene_dict[scene_name])
            render_individual_occlu(scene_name, scene_dict[scene_name])
            toc_part = time.time()
            print(scene_name, toc_part - tic_part, 'seconds')

        # save krt label
        with open(args.intermediate_results_path + '/krt.json', 'w') as f:
            json.dump(krt_dict, f, sort_keys=True, indent=2)
        toc_total = time.time()

def camera_orientation():
    """
    # camera orientations

    Radius : camera position

    theta : camera start at (0, 0, Radius), rotate about world y-axis

    phi : camera at (Rsin(t), 0, Rcos(t)), rotate about world z-axis
          camera ends at (Rsin(t)cos(p), Rsin(t)sin(p), Rcos(t))

    psi : when camera is facing object (camera z-axis), the UP side of rendered image heads toward world z-axis
            rotate camera along camera negative z-axis
    
    # additional angle correction

    alpha : right rotation (rotation axis heading UP of image)

    beta : down rotation (rotation axis heading RIGHT of image)
    

    """
    radius = [22, 25, 28]
    if bool(args.moving_camera_mode):
        theta_phi = [(0, 0), (90, 0), (90, 90), (90, 180), (90, 270), (180, 0)]
        psi = [0, 90 ,180, 270]
        alpha = [25]
        beta = [25]
        direction = ['left', 'right']
    else:
        theta_phi = [(0, 0)]
        psi = [0]
        alpha = [0]
        beta = [0]
        direction = ['left'] # doesn't mean anything (alpha, beta = 0)
    cameras = [(r, tp[0], tp[1], p, a, b, d) for r in radius for tp in theta_phi for p in psi for a in alpha for b in beta for d in direction]
    np.savetxt('./num_cam.txt', np.array([len(cameras)]))
    if bool(args.return_camera_intrinsics):
        return radius[0]
    return cameras



def render_individual_settings():
    render_setting = C.scene.render
    C.scene.render.engine = 'BLENDER_WORKBENCH'
    render_setting.use_freestyle = False
    shading_setting = C.scene.display.shading
    shading_setting.light = 'MATCAP'
    shading_setting.color_type = 'SINGLE'
    render_setting.image_settings.color_mode = 'RGB'
    D.scenes["Scene"].view_settings.view_transform = 'Standard'
    D.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1.0
    # set object color to gray (intended white)
    shading_setting.single_color = (1, 1, 1)  
    # set background color to black
    D.worlds["World"].color = (0, 0, 0)

def render_individual_occlu_settings():
    render_setting = C.scene.render
    C.scene.render.engine = 'BLENDER_EEVEE'
    render_setting.use_freestyle = False
    shading_setting = C.scene.display.shading
    D.scenes["Scene"].view_settings.view_transform = 'Standard'
    D.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1.0
    # set background color to white
    D.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)

def render_settings():
    render_setting = C.scene.render
    C.scene.render.engine = 'BLENDER_EEVEE'
    render_setting.use_freestyle = True
    D.scenes["Scene"].view_settings.view_transform = 'Standard'
    D.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1.0
    D.linestyles["LineStyle"].caps = 'ROUND'
    D.scenes["Scene"].render.line_thickness_mode = 'ABSOLUTE'
    F = D.scenes["Scene"].view_layers[0].freestyle_settings.linesets[0]
    D.scenes["Scene"].render.line_thickness = float(args.thickness)
    F.select_silhouette = True
    F.select_border = False
    F.select_contour = False
    F.select_suggestive_contour = False
    F.select_ridge_valley = False
    F.select_crease = True
    F.select_edge_mark = False
    F.select_external_contour = False
    F.select_material_boundary = False
    # set background color to white
    D.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)


def init_scene():
    """ delete all components in scene. initialize camera. fix camera to origin """
    ''' delete all objects in Blender, add camera '''
    O.object.select_all(action='SELECT')
    O.object.delete()

    O.object.add(type='CAMERA')
    C.scene.camera = C.object
    
    ''' initialize camera '''
    cam = D.objects['Camera']
    C.view_layer.objects.active = cam
    # perspective projection
    C.object.data.type = 'PERSP'

    ''' fix camera to origin '''
    # create origin
    origin_name = 'Origin'
    try:
        origin = D.objects[origin_name]
    except KeyError:
        O.object.empty_add(type='SPHERE')
        D.objects['Empty'].name = origin_name
        origin = D.objects[origin_name]
        pass
    origin.location = (0, 0, 0)
    # fix camera to origin
    cam = D.objects['Camera']
    C.view_layer.objects.active = cam
    if 'Track To' not in cam.constraints:
        O.object.constraint_add(type='TRACK_TO')
    cam.constraints['Track To'].target = origin
    cam.constraints['Track To'].target_space = 'LOCAL'
    cam.constraints['Track To'].owner_space = 'LOCAL'
    cam.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'  # camera direction is -Z axis
    cam.constraints['Track To'].up_axis = 'UP_Y'  # GUI의 UP_X와 동일 --> bug?
    
#----------------------------------- render ----------------------------------------
def select_render_engine(render_type):
    if render_type  == 'render':
        render_settings()
    elif render_type == 'render_individual':
        render_individual_settings()
    elif render_type == 'render_individual_occlu':
        render_individual_occlu_settings()
    else:
        raise Exception("select_render_engine : wrong render type")
        
def render_individual(scene_name, scene):
    krt_dict[scene_name] = {}
    cameras = camera_orientation()
    for camera_index in range(len(cameras)):
        krt_dict[scene_name][str(camera_index).zfill(6)] = []
    select_render_engine('render_individual')
    for model_index in range(len(scene)):
        init_ops(scene, model_index)
        for camera_index, cam in enumerate(cameras):
            K, R, T = create_krt_image(cam, scene, scene_name, camera_index, model_index)
            krt_temp_dict = {}
            krt_temp_dict['cam_K'] = K
            krt_temp_dict['cam_R_m2c'] = R
            krt_temp_dict['cam_t_m2c'] = T
            krt_dict[scene_name][str(camera_index).zfill(6)].append(krt_temp_dict)
            move_camera(cam)
            tilt_model(cam)
            do_render('render_individual', scene_name, camera_index, model_index)
            untilt_model(cam)
        delete_scene()

def render_individual_occlu(scene_name, scene):
    select_render_engine('render_individual_occlu')
    init_ops(scene)
    cameras = camera_orientation()
    cad_adrs = sorted(glob.glob(args.cad_path + '/*.obj'))
    cad_names = [os.path.basename(x).split('.')[0] for x in cad_adrs]
    materials = [x for x in D.materials if x.name != 'Material']
    for model_index, model in enumerate(scene):
        for i, material in enumerate(materials):
            # set only one object to black
            color = (0, 0, 0, 0) if i == model_index else (1, 1, 1, 1)
            material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = color
        for camera_index, cam in enumerate(cameras):
            move_camera(cam)
            tilt_model(cam)
            do_render('render_individual_occlu', scene_name, camera_index, model_index)
            untilt_model(cam)
    delete_scene()

def render(scene_name, scene):
    select_render_engine('render')
    init_ops(scene)
    cameras = camera_orientation()
    for camera_index, cam in enumerate(cameras):
        move_camera(cam)
        tilt_model(cam)
        do_render('render', scene_name, camera_index)
        untilt_model(cam)
    delete_scene()

def do_render(render_type, scene_name, camera_index, model_index=None):
    if render_type == 'render_individual':
        render_output_path = os.path.abspath(os.path.join(args.views_gray_black_path, scene_name))
        render_filename = str(camera_index).zfill(6) + '_' + str(model_index).zfill(6) + '.png'
    elif render_type == 'render_individual_occlu':
        render_output_path = os.path.abspath(os.path.join(args.views_gray_black_occlu_path, scene_name))
        render_filename = str(camera_index).zfill(6) + '_' + str(model_index).zfill(6) + '.png'
    elif render_type == 'render':
        render_output_path = os.path.abspath(os.path.join(args.views_path, scene_name))
        render_filename = str(camera_index).zfill(6) + '.png'
    else:
        raise Exception('do_render : wrong render type')
    C.scene.render.filepath = os.path.join(render_output_path, render_filename)
    O.render.render(write_still=True)

#------------------------------- object ops -------------------------------------

def init_ops(scene, index=None):
    """ load model, shrink model size, set models to default pose"""
    if index is not None:
        i = index
        # load model
        load_models(scene, i)
        # shrink model
        shrink_models()
        # set models to default pose
        set_models_to_default_pose(scene, i)
    else:
        # load models
        load_models(scene)
        # shrink models
        shrink_models()
        # set models to default pose
        set_models_to_default_pose(scene)

def load_models(scene, model_index=None):
    """ load models """
    obj_id_list = [x['obj_id'] for x in scene]
    cad_adrs = sorted(glob.glob(args.cad_path + '/*.obj'))
    cad_adrs = [cad_adrs[x] for x in obj_id_list]
        
    for i, cad_adr in enumerate(cad_adrs):
        if model_index is not None:
            # for single model
            if i != model_index:
                continue
        cad_name = os.path.basename(cad_adr).split('.')[0]
        print('index : ' + str(i) + ', loading : ' + cad_name)
        O.import_scene.obj(filepath=cad_adr, filter_glob='*.obj')
        # object name format : index@cad_name
        for obj in [x for x in D.objects]:
            if cad_name in obj.name and '@' not in obj.name:
                obj.name = str(i) + '@' + cad_name

    # VIEWS setting : set object color to white
    for material in D.materials:
        material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (1, 1, 1, 1)

    # discard .001 when reimporting
    D_objects = [x for x in D.objects]
    for obj in D_objects:
        obj.name = obj.name.split('.')[0]

def shrink_models():
    """ scale model """
    model_objects = [obj for obj in D.objects if obj != D.objects['Camera'] and obj != D.objects['Origin']]
    
    for obj in model_objects:
        dim = obj.dimensions
        print('original dim:' + str(dim))
        dim = dim / int(args.scale)
        obj.dimensions = dim
        print('new dim:' + str(dim))


def set_models_to_default_pose(scene, model_index=None):
    """ read default pose from json, and set models """
    cad_adrs = sorted(glob.glob(args.cad_path + '/*.obj'))
    cad_names = [os.path.basename(x).split('.')[0] for x in cad_adrs]
    for i, model in enumerate(scene):
        if model_index is not None:
            # single model
            if i != model_index:
                continue
        cad_name = cad_names[model['obj_id']]
        obj_name = str(i) + '@' + cad_name
        obj = D.objects[obj_name]
        obj.rotation_mode = 'QUATERNION'
        axis_angle = model['axisangle_WXYZ'].copy()
        axis_angle[0] = deg2rad(axis_angle[0])
        rotation = Rot.from_rotvec(axis_angle[0] * np.array(axis_angle[1:]))
        quaternion = np.roll(rotation.as_quat(), 1)
        obj.rotation_quaternion = quaternion
        lx, ly, lz = model['location_XYZ'].copy()
        obj.location = (lx / int(args.scale), ly / int(args.scale), lz / int(args.scale))


def delete_scene():
    """ delete model """
    for obj in D.objects:
        if obj.name != 'Camera' and obj.name != 'Origin':
            D.objects.remove(obj)
    for mat in D.materials:
        if mat.name != 'Material':
            D.materials.remove(mat)

def delete_camera_and_origin():
    """ delete model """
    for obj in D.objects:
        if obj.name == 'Camera' or obj.name == 'Origin':
            D.objects.remove(obj)

#------------------------ camera ------------------------------------------
def theta_phi_correction(theta, phi):
    """change theta, phi value to less erronous values
    
    theta : 0 -> 1, 180 -> 179
    phi : 0 -> 1

    Args:
        theta: (int) degrees
        phi: (int) degrees
    Returns:
        theta : (int) degrees
        phi : (int) degrees
    """
    if theta == 0:
        theta = deg2rad(1)
    if phi == 0:
        phi = deg2rad(1)
    if theta == 180:
        theta = deg2rad(179)
    return theta, phi

def move_camera(cam):
    """ set camera location, r-axis rotation """
    # move camera
    r = cam[0]
    theta, phi = cam[1], cam[2]
    theta, phi = theta_phi_correction(theta, phi)
    theta, phi = deg2rad(theta), deg2rad(phi)    
    angle = deg2rad(cam[3])

    C.view_layer.objects.active = D.objects['Camera']
    loc_x, loc_y, loc_z = circular_loc(r, theta, phi)
    D.objects['Camera'].location = (loc_x, loc_y, loc_z)
    D.objects['Camera'].rotation_mode = 'AXIS_ANGLE'
    D.objects['Camera'].rotation_axis_angle = (angle, loc_x, loc_y, loc_z)


def circular_loc(r, theta, phi):
    """ set camera location """
    loc_x = r * math.sin(theta) * math.cos(phi)
    loc_y = r * math.sin(theta) * math.sin(phi)
    loc_z = r * math.cos(theta)
    return loc_x, loc_y, loc_z


def deg2rad(deg):
    return deg * math.pi / 180.0

def tilt_model(cam, reverse=False):
    """
    first rotation : right alpha (y-axis)
    second rotation : down beta (x-axis)
    """
    theta, phi, psi, alpha, beta, direction = cam[1:]
    theta, phi = theta_phi_correction(theta, phi)
    alpha = alpha if direction == 'right' else -alpha
    # set origin of all model objects to (0, 0, 0)
    model_objects = [obj for obj in D.objects if obj != D.objects['Camera'] and obj != D.objects['Origin']]
    for obj in model_objects:
        obj.select_set(True)
    O.object.origin_set(type='ORIGIN_CURSOR')

    # figure out rotation axes
    # normalized forward vector (target --> eye)
    loc_x = np.sin(deg2rad(theta)) * np.cos(deg2rad(phi))
    loc_y = np.sin(deg2rad(theta)) * np.sin(deg2rad(phi))
    loc_z = np.cos(deg2rad(theta))
    f = np.array([loc_x, loc_y, loc_z], dtype=np.float32)

    # side vector of camera
    u_world = np.array([0, 0, 1], dtype=np.float32)
    s = np.cross(u_world, f)
    norm_s = np.linalg.norm(s)
    if norm_s == 0:
        s = np.array([0, 1, 0], dtype=np.float32) * np.dot(u_world, f)
    else:
        s /= norm_s

    # up vector of camera
    u = np.cross(f,s)
    
    # rotate camera
    rotation = Rot.from_rotvec(np.deg2rad(psi) * f)
    u = rotation.apply(u).astype(np.float32)
    s = rotation.apply(s).astype(np.float32)
    for obj in model_objects:
        
        # first rotation
        first_rotation = Rot.from_rotvec(deg2rad(alpha) * u)

        # second rotation
        second_rotation = Rot.from_rotvec(deg2rad(beta) * s)

        # apply additional angle correction
        rotation = second_rotation * first_rotation
        quaternion = np.roll(rotation.as_quat(), 1)
        
        if reverse:
            # un-tilt model
            obj.delta_rotation_quaternion = -quaternion
        else:
            # tilt model
            obj.delta_rotation_quaternion = quaternion

def untilt_model(cam):
    tilt_model(cam, reverse=True)



#-------------------------- KRT -----------------------------------------

def cam2RT(cam):
    """ rotation matrix from camera orientation 
    
    camera z-axis : camera point to target point
    camera y-axis : facing down
    camera x-axis : facing right
    """
    radius, theta, phi, psi, alpha, beta, direction = cam
    theta, phi = theta_phi_correction(theta, phi)
    alpha = alpha if direction == 'right' else -alpha
    # normalized forward vector (target --> eye)
    loc_x = np.sin(deg2rad(theta)) * np.cos(deg2rad(phi))
    loc_y = np.sin(deg2rad(theta)) * np.sin(deg2rad(phi))
    loc_z = np.cos(deg2rad(theta))
    f = np.array([loc_x, loc_y, loc_z], dtype=np.float32)

    # side vector of camera
    u_world = np.array([0, 0, 1], dtype=np.float32)
    s = np.cross(u_world, f)
    norm_s = np.linalg.norm(s)
    if norm_s == 0:
        s = np.array([0, 1, 0], dtype=np.float32) * np.dot(u_world, f)
    else:
        s /= norm_s

    # up vector of camera
    u = np.cross(f,s)

    # rotate camera
    rotation = Rot.from_rotvec(np.deg2rad(psi) * f)
    u = rotation.apply(u).astype(np.float32)
    s = rotation.apply(s).astype(np.float32)

    # rotation matrix
    cam_z = -f
    cam_y = -u
    cam_x = s
    R = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

    # translation matrix
    C = np.array(radius * np.array([loc_x, loc_y, loc_z]), dtype=np.float32).reshape(3, 1)
    T = -R @ C
    
    # additional angle correction
    first_rotation = Rot.from_rotvec(deg2rad(alpha) * u)
    second_rotation = Rot.from_rotvec(deg2rad(beta) * s)
    rotation = second_rotation * first_rotation
    R = R @ rotation.as_matrix()

    RT = np.concatenate((R, T), axis=1)

    return RT


def set_models_to_default_pose(scene, model_index=None):
    """ read default pose from json, and set models """
    cad_adrs = sorted(glob.glob(args.cad_path + '/*.obj'))
    cad_names = [os.path.basename(x).split('.')[0] for x in cad_adrs]
    for i, model in enumerate(scene):
        if model_index is not None:
            # single model
            if i != model_index:
                continue
        cad_name = cad_names[model['obj_id']]
        obj_name = str(i) + '@' + cad_name
        obj = D.objects[obj_name]
        obj.rotation_mode = 'QUATERNION'
        axis_angle = model['axisangle_WXYZ'].copy()
        axis_angle[0] = deg2rad(axis_angle[0])
        rotation = Rot.from_rotvec(axis_angle[0] * np.array(axis_angle[1:]))
        quaternion = np.roll(rotation.as_quat(), 1)
        obj.rotation_quaternion = quaternion
        lx, ly, lz = model['location_XYZ'].copy()
        obj.location = (lx / int(args.scale), ly / int(args.scale), lz / int(args.scale))


def create_krt_image(cam, scene, scene_name, camera_index, model_index):
    """ save images for checking whether calculated KRT are correct """
    
    img = np.zeros((int(args.H), int(args.W))).astype(np.uint8)

    # read point cloud
    obj = [x for x in D.objects if x.name != 'Camera' and x.name != 'Origin']
    assert len(obj) == 1
    obj =  obj[0]
    obj_name = obj.name.split('@')[-1]
    pt_cld_adrs = sorted(glob.glob(args.point_cloud_path + '/*'))
    pt_cld_adr = [x for x in pt_cld_adrs if os.path.basename(x).startswith(obj_name)][0]
    pt_cld_data = np.loadtxt(pt_cld_adr, skiprows=2, usecols=(0, 1, 2))
    pt_cld_data /= int(args.scale)
    ones = np.ones((pt_cld_data.shape[0], 1))
    xyz = np.append(pt_cld_data[:, :3], ones, axis=1)  # Nx4 : (x, y, z, 1)

    # projection
    obj.select_set(True)
    O.object.origin_set(type='ORIGIN_CENTER_OF_MASS')

    model = scene[model_index]
    axis_angle = model['axisangle_WXYZ'].copy()
    axis_angle[0] = deg2rad(axis_angle[0])
    rotation_default = Rot.from_rotvec(axis_angle[0] * np.array(axis_angle[1:]))
    R_default = rotation_default.as_matrix()
    lx, ly, lz = model['location_XYZ'].copy()
    T_default =  np.array([lx / int(args.scale), ly / int(args.scale), lz / int(args.scale)]).reshape(3, 1)
    RT_default = np.concatenate((R_default, T_default), axis=1)
    RT_default_4x4 = np.concatenate((RT_default, np.array([0, 0, 0, 1]).reshape(1,4)), axis=0)

    K = blender_K()
    RT = cam2RT(cam)
    RT = RT @ RT_default_4x4
    xy = K @ RT @ xyz.T
    coord = xy[:2, :] / xy[2, :]
    coord = (np.floor(coord)).T.astype(int)
    x = np.clip(coord[:, 0], 0, int(args.W) - 1)
    y = np.clip(coord[:, 1], 0, int(args.H) - 1)
    img[y, x] = 200

    # save image
    view = cv2.imread(os.path.join(args.views_path, scene_name, str(camera_index).zfill(6) + '.png'))
    save_path = os.path.join(args.krt_image_path, scene_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = str(camera_index).zfill(6) + '_' + str(model_index).zfill(6) + '.png'
    save_adr = os.path.join(save_path, filename)
    cv2.imwrite(save_adr, img + view[..., 0])

    # return labels
    K = K.flatten().tolist()
    R = RT[:, :3].flatten().tolist()
    T = RT[:, 3].tolist()

    return K, R, T


def blender_K():
    """ get K """
    cam = bpy.data.objects['Camera']
    if bool(args.return_camera_intrinsics):
        return get_calibration_matrix_K_from_blender(cam.data)
    K = get_calibration_matrix_K_from_blender(cam.data)
    K = np.array(K)
    return K


#======================================
# 3x4 P matrix from Blender camera
# scrapped from google
#======================================

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

'''
Build intrinsic camera parameters from Blender camera data

See notes on this in
blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
as well as
https://blender.stackexchange.com/a/120063/3581
'''

def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene

    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0  # only use rectangular pixels
    Enlarge_Factor = 1
    s_u /= Enlarge_Factor
    s_v /= Enlarge_Factor
    u_0 /= Enlarge_Factor
    v_0 /= Enlarge_Factor

    K = Matrix(
        ((s_u, skew, u_0),
         (0, s_v, v_0),
         (0, 0, 1)))
    if bool(args.return_camera_intrinsics):
        return f_in_mm, sensor_size_in_mm
    return K
#---------------------------------------------------------------

if __name__ == '__main__':
    main()
