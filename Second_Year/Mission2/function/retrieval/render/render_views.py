""" file runs in blender """
import bpy
from mathutils import Matrix
import os
import sys
import glob
import math
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append('./function/Pose')
from pose_gt import POSE_TO_LABEL
import platform
from pathlib import Path
from PIL import Image

if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-render_type', help='render type - VIEWS_GRAY_BLACK : surface rendering, VIEWS : line rendering ')
    parser.add_argument('-list_added_obj', nargs='+', help='added .obj files in cad_path list')
    parser.add_argument('-render_output_path', help='render_output path')
    parser.add_argument('-cad_path', help='cad input path')
    parser.add_argument('-point_cloud_path', help='point cloud output path')
    parser.add_argument('-krt_path', help='krt output path')
    parser.add_argument('-krt_imgs_path', help='krt imgs output path. for validating krt outputs')
    args = parser.parse_known_args(argv)[0]
print(args.list_added_obj)

#=================================================================================
#                              Hyperparameters
#=================================================================================
Radius = 2  # radius of camera position
Scale = 1000  # cad_model = 1/Scale * cad_model (applied to VIEWS)
Line_Thickness = 2.0 # line thickness of line rendering
Ortho_Scale = 1.5 # hyperparameter for orthographic projection
H, W = 224, 224  # rendering resolution
Enlarge_Factor = 2  # OPTIONAL : rendering resolution 448 x 448 for VIEWS
if args.render_type == 'views':
    H, W = Enlarge_Factor * H, Enlarge_Factor * W
A1, A2 = 25, 25  # tilting angles for additional angle correction
#=================================================================================

D = bpy.data
C = bpy.context
O = bpy.ops

# delete all objects in Blender, add camera
O.object.select_all(action='SELECT')
O.object.delete()
O.object.add(type='CAMERA')
C.scene.camera = C.object

# rendering settings
shading_settting = C.scene.display.shading
shading_settting.color_type = 'SINGLE'  # VIEWS_GRAY_BLACK : render settings
shading_settting.single_color = (1, 1, 1)  # VIEWS_GRAY_BLACK : set object color to gray (intended white)
shading_settting.light = 'MATCAP'  # VIEWS_GRAY_BLACK : render settings
render_setting = C.scene.render
render_setting.image_settings.color_mode = 'RGB'  # ?
D.scenes["Scene"].view_settings.view_transform = 'Standard'  # ?
D.worlds["World"].color = (0, 0, 0) if 'black' in args.render_type.lower() else (1, 1, 1)  # VIEWS_GRAY_BLACK : set background color to black
D.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)  # VIEWS : set background color to white

# VIEWS settings
D.scenes["Scene"].render.line_thickness_mode = 'ABSOLUTE'  # VIEWS : ['RELATIVE', 'ABSOLUTE'] : line thickness settings
F = D.scenes["Scene"].view_layers[0].freestyle_settings.linesets[0]
F.linestyle.thickness = Line_Thickness  # VIEWS : line thickness
# VIEWS : ? trial and error...
F.select_silhouette = True
F.select_border = False
F.select_contour = False
F.select_suggestive_contour = False
F.select_ridge_valley = False
F.select_crease = True
F.select_edge_mark = False
F.select_external_contour = False
F.select_material_boundary = False

# select render engine
if args.render_type in ['views']:
    # VIEWS (line rendering)
    C.scene.render.engine = 'BLENDER_EEVEE'
    render_setting.use_freestyle = True

elif args.render_type in ['views_gray_black']:
    # VIEWS_GRAY_BLACK (surface rendering)
    C.scene.render.engine = 'BLENDER_WORKBENCH'
    render_setting.use_freestyle = False
else:
    raise Exception("wrong render type")


# resolution
render_setting.resolution_y = H
render_setting.resolution_x = W


# camera orientations
"""
theta : camera start at (0, 0, R), rotate about world y-axis

phi : camera at (Rsin(t), 0, Rcos(t)), rotate about world z-axis
      camera ends at (Rsin(t)cos(p), Rsin(t)sin(p), Rcos(t))

alpha : when camera is facing object (camera z-axis), the UP side of rendered image heads toward world z-axis
        rotate camera along camera negative z-axis
"""

theta_phi = [(0, 0), (90, 0), (90, 90), (90, 180), (90, 270), (180, 0)]
alpha = [0, 90, 180, 270]
cameras = [(tp[0], tp[1], a) for tp in theta_phi for a in alpha]


def main():
    tic_total = time.time()

    existing_folders = sorted(glob.glob(os.path.join(args.render_output_path + '/*')))
    existing_folders = [os.path.basename(s) for s in existing_folders]

    # render
    for added_obj in args.list_added_obj:
        tic_part = time.time()
        added_obj_name = os.path.splitext(added_obj)[0]
        # skip for already existing CAD name folders in rendering output directory
        if added_obj_name in existing_folders:
            print('{} already exists in {}, skipping rendering'.format(added_obj_name, args.render_output_path))
            continue
        init_camera()
        fix_camera_to_origin()
        do_model(os.path.join(args.cad_path, added_obj))
        toc_part = time.time()
        print(added_obj_name, toc_part - tic_part, 'seconds')
    toc_total = time.time()


def init_camera():
    """ initialize camera settings """
    cam = D.objects['Camera']
    C.view_layer.objects.active = cam
    if args.render_type == 'views':
        # perspective projection
        C.object.data.type = 'PERSP'
    else:
        # orthographic projection
        C.object.data.type = 'ORTHO'
        C.object.data.ortho_scale = Ortho_Scale


def fix_camera_to_origin():
    """ fix camera to origin """
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


def do_model(path):
    # load model
    name = load_model(path)
    if D.objects[0] == D.objects['Camera']:
        if D.objects[1] == D.objects['Origin']:
            model_object = D.objects[2]
        else:
            model_object = D.objects[1]
    elif D.objects[1] == D.objects['Camera']:
        if D.objects[0] == D.objects['Origin']:
            model_object = D.objects[2]
        else:
            model_object = D.objects[0]
    elif D.objects[2] == D.objects['Camera']:
        if D.objects[0] == D.objects['Origin']:
            model_object = D.objects[1]
        else:
            model_object = D.objects[0]
    else:
        print(D.objects)
        print(D.objects[0])
        print(D.objects[1])
        print(D.objects[2])
        raise Exception('blender model loading error')
    # center model
    center_model(model_object)
    print(path)

    # shrink or normalize model
    if args.render_type == 'views':
        # VIEWS : scale down model size with 'Scale'
        shrink_model(model_object)
    else:
        # VIEWS_GRAY_BLACK : normalize model size
        normalize_model(model_object)

    # load point cloud (for KRT visualization)
    pt_cld_file = args.point_cloud_path + '/' + os.path.splitext(os.path.basename(path))[0] + '_SAMPLED_POINTS.asc'

    # rotate model
    imported = C.selected_objects[0]
    imported.rotation_mode = 'XYZ'
    imported.rotation_euler = (0, 0, 0)

    image_subdir = os.path.join(args.render_output_path, name)
    image_subdir_abs = os.path.abspath(image_subdir)
    krt_image_subdir = os.path.join(args.krt_imgs_path, name)
    krt_image_subdir_abs = os.path.abspath(krt_image_subdir)
    if not os.path.exists(krt_image_subdir_abs):
        os.makedirs(krt_image_subdir_abs)
    for material in bpy.data.materials:
        material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (1, 1, 1, 1)  # VIEWS : set object color to white
    for c in cameras:
        # move camera
        move_camera(c)
        c = np.array(c)
        c_s = c.astype('str')
        c_s = np.char.zfill(c_s, 3)
        # rotate model
        imported.rotation_euler = (0, 0, 0)
        # additional angle correction
        tilt_model(imported, c, 'left')
        # render
        C.scene.render.filepath = os.path.join(image_subdir_abs, '{}_{}_{}_{}_left.png'.format(name, c_s[0], c_s[1], c_s[2]))
        O.render.render(write_still=True)
        if platform.system() == 'Windows':
            # compenstate for alpha rotation error in windows
            # camera and image rotation is opposite
            filepath = str(Path(os.path.join(image_subdir_abs, '{}_{}_{}_{}_left.png'.format(name, c_s[0], c_s[1], c_s[2]))))
            rendering_img = np.array(plt.imread(filepath))
            rot90_num = 4 - (c[2] // 90)
            rendering_img = np.rot90(rendering_img, rot90_num)
            im = Image.fromarray((rendering_img * 255).astype(np.uint8))  # plt saves RGBA, Image saves RGB
            im.save(filepath)
        print('saved {}_{}_{}_{}_left.png'.format(name, c_s[0], c_s[1], c_s[2]))
        if args.render_type == 'views':
            # save KRT validation imgs
            K, _ = get_KRT()
            RT, RT_default = calculate_RT(c, 'left')
            print('RT')
            print(RT)
            pose = '{}_{}_{}_{}'.format(c_s[0], c_s[1], c_s[2], 'left')
            save_KRT(K, RT_default, RT, pose)
            img = get_image_from_pose(pt_cld_file, K, RT)
            plt.imsave(os.path.join(krt_image_subdir_abs, '{}_{}_{}_{}_zl.png'.format(name, c_s[0], c_s[1], c_s[2])), img, cmap='gray')

        # rotate model
        imported.rotation_euler = (0, 0, 0)
        # additional angle correction
        tilt_model(imported, c, 'right')
        # render
        C.scene.render.filepath = os.path.join(image_subdir_abs, '{}_{}_{}_{}_right.png'.format(name, c_s[0], c_s[1], c_s[2]))
        O.render.render(write_still=True)
        if platform.system() == 'Windows':
            # compenstate for alpha rotation error in windows
            # camera and image rotation is opposite
            filepath = str(Path(os.path.join(image_subdir_abs, '{}_{}_{}_{}_right.png'.format(name, c_s[0], c_s[1], c_s[2]))))
            rendering_img = np.array(plt.imread(filepath))
            rot90_num = 4 - (c[2] // 90)
            rendering_img = np.rot90(rendering_img, rot90_num)
            im = Image.fromarray((rendering_img * 255).astype(np.uint8))  # plt saves RGBA, Image saves RGB
            im.save(filepath)
        print('saved {}_{}_{}_{}_right.png'.format(name, c_s[0], c_s[1], c_s[2]))
        if args.render_type == 'views':
            # save KRT validation images
            K, _ = get_KRT()
            RT, RT_default = calculate_RT(c, 'right')
            print('RT')
            print(RT)
            pose = '{}_{}_{}_{}'.format(c_s[0], c_s[1], c_s[2], 'right')
            save_KRT(K, RT_default, RT, pose)
            img = get_image_from_pose(pt_cld_file, K, RT)
            plt.imsave(os.path.join(krt_image_subdir_abs, '{}_{}_{}_{}_zr.png'.format(name, c_s[0], c_s[1], c_s[2])), img, cmap='gray')
        imported.rotation_euler = (0, 0, 0)
    delete_model()


def tilt_model(imported, c, direction):
    """
    first rotation : right a1
    second rotation : down a2
    """
    a1, a2 = -A1, A2
    a1 = -a1 if direction == 'right' else a1

    c0, c1, c2 = c
    """axis_dict = {(theta, phi) : right direction axis (M-axis), up direction axis (L-axis), remaining axis, sign for axis_dict[smth][0], sign for axis_dict[smth][1]}"""
    axis_dict = {
        (0, 0): ('X', 'Y', 'Z', -1, 1),
        (90, 0): ('Z', 'Y', 'X', 1, 1),
        (90, 90): ('Z', 'X', 'Y', 1, -1),
        (90, 180): ('Z', 'Y', 'X', 1, -1),
        (90, 270): ('Z', 'X', 'Y', 1, 1),
        (180, 0): ('X', 'Y', 'Z', 1, 1)
    }
    ax = axis_dict[(c0, c1)]
    # order of rotation
    imported.rotation_mode = ax[0] + ax[1] + ax[2] if c2 % 180 == 0 else ax[1] + ax[0] + ax[2]
    lst = [(1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)] # imitating circular rotation of [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    index = lst.index((ax[3], ax[4]))
    lst1 = [0, 90, 180, 270]
    index1 = lst1.index(c2)
    sign = lst[index + index1]
    lst2 = ['X', 'Y', 'Z']
    idx0 = lst2.index(ax[0])
    idx1 = lst2.index(ax[1])
    a = [0, 0, 0]
    if c2 % 180 == 0:
        a[idx0] = sign[0] * a1
        a[idx1] = sign[1] * a2
    else:
        a[idx1] = sign[0] * a1
        a[idx0] = sign[1] * a2
    imported.rotation_euler = deg2rad(np.array(a))
    print(c, imported.rotation_mode, a)


def load_model(path):
    """ load model """
    name = os.path.basename(path).split('.')[0]
    if name not in D.objects:
        print('loading :' + name)
        O.import_scene.obj(filepath=path, filter_glob='*.obj')
    return name


def center_model(obj):
    """ center model """
    O.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    obj.location = (0, 0, 0)


def normalize_model(obj):
    """ normalize model """
    dim = obj.dimensions
    print('original dim:' + str(dim))
    if max(dim) > 0:
        dim = dim / max(dim)
    obj.dimensions = dim
    print('new dim:' + str(dim))


def shrink_model(obj):
    """ scale model """
    dim = obj.dimensions
    print('original dim:' + str(dim))
    dim = dim / Scale
    obj.dimensions = dim
    print('new dim:' + str(dim))


def delete_model():
    """ delete model """
    for ob in C.scene.objects:
        if ob.type == 'MESH':
            ob.select_set(True)
        else:
            ob.select_set(False)
    O.object.delete()


def deg2rad(deg):
    return deg * math.pi / 180.0


def circular_loc(theta, phi):
    """ set camera location """
    r = Radius
    loc_x = r * math.sin(theta) * math.cos(phi)
    loc_y = r * math.sin(theta) * math.sin(phi)
    loc_z = r * math.cos(theta)
    return loc_x, loc_y, loc_z


def move_camera(coord):
    """ set camera location, r-axis rotation """
    cam = D.objects['Camera']
    C.view_layer.objects.active = cam
    # move camera
    theta, phi = deg2rad(coord[0]), deg2rad(coord[1])
    if theta == 0:
        theta = deg2rad(1)
    if phi == 0:
        phi = deg2rad(1)
    if theta == 180:
        theta = deg2rad(179)
    a = deg2rad(coord[2])

    loc_x, loc_y, loc_z = circular_loc(theta, phi)
    cam.location = (loc_x, loc_y, loc_z)
    # rotate camera (r-axis) ## WARNING : works at ubuntu, doesn't work at windows
    cam.rotation_mode = 'AXIS_ANGLE'
    cam.rotation_axis_angle = (a, loc_x, loc_y, loc_z)


def rotX(deg):
    """ for RT computation """
    rad = deg2rad(deg)
    return np.array([
        [1, 0, 0],
        [0, np.cos(rad), -np.sin(rad)],
        [0, np.sin(rad), np.cos(rad)]
    ])


def rotY(deg):
    """ for RT computation """
    rad = deg2rad(deg)
    return np.array([
        [np.cos(rad), 0, np.sin(rad)],
        [0, 1, 0],
        [-np.sin(rad), 0, np.cos(rad)]
    ])


def rotZ(deg):
    """ for RT computation """
    rad = deg2rad(deg)
    return np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad), np.cos(rad), 0],
        [0, 0, 1]
    ])


def calculate_RT(c, direction):
    """ calculate RT from theta, phi, alpha, additional angle correction direction (left, right)
        camera is pinned at (0, 0, Radius)
        only the CAD model rotates at location (0, 0, 0)
    """
    c0, c1, c2 = c
    # default
    r0 = rotZ(-90) @ rotY(180)
    # phi, theta, alpha
    r1 = rotZ(c1)
    r2 = rotX(-c0)
    r3 = rotZ(c2)
    # additional angle correction
    r4 = rotY(A1) if direction == 'left' else rotY(-A1)
    r5 = rotX(A2)

    R = r5 @ r4 @ r3 @ r2 @ r1 @ r0
    R_default = r0
    t = np.array([[0, 0, Radius]]).T
    RT = np.around(np.concatenate((R, t), axis=1), 4)
    RT_default = np.around(np.concatenate((R_default, t), axis=1), 4)
    return RT, RT_default


def get_image_from_pose(pt_cld_file, K, RT):
    """ make pose images(krt validation images) for checking calculated RT outputs """
    img = np.zeros((224, 224)).astype(np.uint8)

    # Read point Point Cloud Data
    pt_cld_data = np.loadtxt(pt_cld_file, skiprows=2, usecols=(0, 1, 2))
    pt_cld_data /= Scale
    ones = np.ones((pt_cld_data.shape[0], 1))
    homogenous_coordinate = np.append(pt_cld_data[:, :3], ones, axis=1)  # Nx4 : (x, y, z, 1)

    # Perspective Projection to obtain 2D coordinates for masks
    # 3xN = 3x3 @ 3x4 @ 4xN
    K = np.array(K)
    RT = np.array(RT)
    homogenous_2D = K @ RT @ homogenous_coordinate.T
    # 2xN
    coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
    # Nx2 (x, y)
    coord_2D = (np.floor(coord_2D)).T.astype(int)
    x_2d = np.clip(coord_2D[:, 0], 0, 223)
    y_2d = np.clip(coord_2D[:, 1], 0, 223)
    img[y_2d, x_2d] = 255

    return img


def get_KRT():
    """ get K, Rt. but only use K """
    cam = bpy.data.objects['Camera']
    P, K, RT = get_3x4_P_matrix_from_blender(cam)
    return K, RT


def save_KRT(K, RT_default, RT, pose):
    """ save K, Rt """
    np.savetxt(args.krt_path + '/K.txt', K, '%10.5f')
    np.savetxt(args.krt_path + '/RT_default.txt', RT_default, '%10.5f')
    pose_index = POSE_TO_LABEL[pose]
    np.savetxt(args.krt_path + '/RT_' + str(pose_index).zfill(2) + '.txt', RT, '%10.5f')


#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
# scrapped from google
#---------------------------------------------------------------

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

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581


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

    s_u /= Enlarge_Factor
    s_v /= Enlarge_Factor
    u_0 /= Enlarge_Factor
    v_0 /= Enlarge_Factor

    K = Matrix(
        ((s_u, skew, u_0),
         (0, s_v, v_0),
         (0, 0, 1)))
    return K

# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction


def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
    ))
    return RT


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT

#---------------------------------------------------------------

if __name__ == '__main__':
    main()
