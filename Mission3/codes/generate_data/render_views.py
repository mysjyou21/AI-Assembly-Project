# file runs in blender
import bpy
import os
import sys
import glob
import math
import time
import argparse
import numpy as np

sys.path.append('./')
from render_run import H, W

if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser('go to render_run.py for better descriptions')
    parser.add_argument('-render_type', help='render type')
    parser.add_argument('-target_models_path', help='target_models_path')
    parser.add_argument('-render_output_path', help='render_output path')
    parser.add_argument('-cad_ext', help='cad extension')
    args = parser.parse_known_args(argv)[0]

is_stl = True if args.cad_ext == '.STL' else False
is_ply = True if args.cad_ext == '.ply' else False

D = bpy.data
C = bpy.context
O = bpy.ops

shading_settting = C.scene.display.shading
shading_settting.color_type = 'SINGLE'
shading_settting.single_color = (1, 1, 1)  # surface : object color
shading_settting.light = 'MATCAP'
O.object.select_all(action='SELECT')
O.object.delete()
O.object.add(type='CAMERA')
C.scene.camera = C.object
render_setting = C.scene.render
render_setting.image_settings.color_mode = 'RGB'
D.worlds["World"].color = (0, 0, 0) if 'black' in args.render_type.lower() else (1, 1, 1)  # surface : background color
D.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)  # line : background color
D.scenes["Scene"].view_settings.view_transform = 'Standard'
D.scenes["Scene"].render.line_thickness_mode = 'RELATIVE'  # 'ABSOLUTE'
F = D.scenes["Scene"].view_layers[0].freestyle_settings.linesets[0]
# F.linestyle.thickness = 2.0
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
if args.render_type in ['views', 'views_black']:
    # line
    C.scene.render.engine = 'BLENDER_EEVEE'
    render_setting.use_freestyle = True

elif args.render_type in ['views_gray', 'views_gray_black']:
    # surface
    C.scene.render.engine = 'BLENDER_WORKBENCH'
    render_setting.use_freestyle = False
else:
    raise Exception(f"wrong render type")


# resolution
render_setting.resolution_x = W
render_setting.resolution_y = H

# camera orientations
'''
theta : camera start at (0, 0, R), rotate about global y-axis

phi : camera at (Rsin(t), 0, Rcos(t)), rotate about global z-axis
      camera ends at (Rsin(t)cos(p), Rsin(t)sin(p), Rcos(t))

alpha : when camera is facing object (camera z-axis), the UP side of rendered image heads toward global z-axis
        rotate camera along camera negative z-axis
'''

# new viewpoint

theta_phi = [(0, 0), (90, 0), (90, 90), (90, 180), (90, 270), (180, 0)]
alpha = [0, 90, 180, 270]
cameras = [(tp[0], tp[1], a) for tp in theta_phi for a in alpha]
use_tilt_model = True

# old viewpoint

# theta = [60, 120]
# phi = [i for i in range(0, 360, 30)]
# alpha = [0]
# cameras = [(t, p, a) for t in theta for p in phi for a in alpha]
# use_tilt_model = False


def main():
    tic_total = time.time()
    if is_stl:
        files = sorted(glob.glob(os.path.join(args.target_models_path, '*.STL')))
    elif is_ply:
        files = sorted(glob.glob(os.path.join(args.target_models_path, '*.ply')))
    else:
        files = sorted(glob.glob(os.path.join(args.target_models_path, '*.obj')))

    # render
    for file in files:
        tic_part = time.time()
        init_camera()
        fix_camera_to_origin()
        do_model(file)
        toc_part = time.time()
        name = os.path.basename(file)
        print(name, toc_part - tic_part, 'seconds')
    toc_total = time.time()


def init_camera():
    cam = D.objects['Camera']
    C.view_layer.objects.active = cam
    C.object.data.type = 'ORTHO'
    C.object.data.ortho_scale = 1.5


def fix_camera_to_origin():
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
    cam.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
    cam.constraints['Track To'].up_axis = 'UP_Y'


def do_model(path):
    name = load_model(path)
    center_model(name)
    normalize_model(name)
    imported = C.selected_objects[0]
    imported.rotation_mode = 'XYZ'
    imported.rotation_euler = (0, 0, 0)
    image_subdir = os.path.join(args.render_output_path, name)
    image_subdir_abs = os.path.abspath(image_subdir)
    for material in bpy.data.materials:
        material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (1, 0, 0, 0) if 'black' in args.render_type.lower() else (1, 1, 1, 1)  # line : object color
    for c in cameras:
        move_camera(c)
        c = np.array(c)
        c_s = c.astype('str')
        c_s = np.char.zfill(c_s, 3)
        if use_tilt_model:
            tilt_model(imported, c)
        C.scene.render.filepath = os.path.join(image_subdir_abs, f'{name}_{c_s[0]}_{c_s[1]}_{c_s[2]}.png')
        O.render.render(write_still=True)
    delete_model(name)


def tilt_model(imported, c):

    c0, c1, c2 = c
    assert c2 in [0, 90, 180, 270]
    a1, a2 = 0, 0
    a1 = 20
    a2 = -30
    if [c0, c1] == [0, 0]:
        imported.rotation_mode = 'XYZ'
        if c2 == 0:
            s1, s2 = 1, -1
            a = (a1, -a2, 0)
        elif c2 == 90:
            s1, s2 = -1, -1
            a = (-a1, -a2, 0)
        elif c2 == 180:
            s1, s2 = -1, 1
            a = (-a1, a2, 0)
        elif c2 == 270:
            s1, s2 = 1, 1
            a = (a1, a2, 0)
    elif [c0, c1] == [90, 0]:
        imported.rotation_mode = 'ZYX'
        if c2 == 0:
            s1, s2 = -1, -1
            a = (0, -a2, -a1)
        elif c2 == 90:
            s1, s2 = 1, -1
            a = (0, -a2, a1)
        elif c2 == 180:
            s1, s2 = 1, 1
            a = (0, a2, a1)
        elif c2 == 270:
            s1, s2 = -1, 1
            a = (0, a2, -a1)
    elif [c0, c1] == [90, 90]:
        imported.rotation_mode = 'ZXY'
        if c2 == 0:
            s1, s2 = -1, 1
            a = (a2, 0, -a1)
        elif c2 == 90:
            s1, s2 = 1, 1
            a = (a2, 0, a1)
        elif c2 == 180:
            s1, s2 = 1, -1
            a = (-a2, 0, a1)
        elif c2 == 270:
            s1, s2 = -1, -1
            a = (-a2, 0, -a1)
    elif [c0, c1] == [90, 180]:
        imported.rotation_mode = 'ZYX'
        if c2 == 0:
            s1, s2 = -1, 1
            a = (0, a2, -a1)
        elif c2 == 90:
            s1, s2 = 1, 1
            a = (0, a2, a1)
        elif c2 == 180:
            s1, s2 = 1, -1
            a = (0, -a2, a1)
        elif c2 == 270:
            s1, s2 = -1, -1
            a = (0, -a2, -a1)
    elif [c0, c1] == [90, 270]:
        imported.rotation_mode = 'ZXY'
        if c2 == 0:
            s1, s2 = -1, -1
            a = (-a2, 0, -a1)
        elif c2 == 90:
            s1, s2 = 1, -1
            a = (-a2, 0, a1)
        elif c2 == 180:
            s1, s2 = 1, 1
            a = (a2, 0, a1)
        elif c2 == 270:
            s1, s2 = -1, 1
            a = (a2, 0, -a1)
    elif [c0, c1] == [180, 0]:
        imported.rotation_mode = 'XYZ'
        if c2 == 0:
            s1, s2 = -1, -1
            a = (-a1, -a2, 0)
        elif c2 == 90:
            s1, s2 = 1, -1
            a = (a1, -a2, 0)
        elif c2 == 180:
            s1, s2 = 1, 1
            a = (a1, a2, 0)
        elif c2 == 270:
            s1, s2 = -1, 1
            a = (-a1, a2, 0)
    else:
        raise Exception('not predefined viewpoints')

    imported.rotation_euler = deg2rad(np.array(a))


def load_model(path):
    d = os.path.dirname(path)
    ext = path.split('.')[-1]
    name = os.path.basename(path).split('.')[0]
    if name not in D.objects:
        print('loading :' + name)
        if is_stl:
            O.import_mesh.stl(filepath=path, filter_glob='*.STL')
        elif is_ply:
            O.import_mesh.ply(filepath=path, filter_glob='*.ply')
        else:
            O.import_scene.obj(filepath=path, filter_glob='*.obj')
    print(D.objects[0])
    print(D.objects[1])
    print(D.objects[2])
    return name


def center_model(name):

    O.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    try:
        D.objects[name].location = (0, 0, 0)
    except:
        name = name.replace('_', ' ').title()
        D.objects[name].location = (0, 0, 0)


def normalize_model(name):
    try:
        obj = D.objects[name]
    except:
        name = name.replace('_', ' ').title()
        obj = D.objects[name]
    dim = obj.dimensions
    print('original dim:' + str(dim))
    if max(dim) > 0:
        dim = dim / max(dim)
    obj.dimensions = dim
    print('new dim:' + str(dim))


def delete_model(name):
    for ob in C.scene.objects:
        if ob.type == 'MESH':  # and ob.name.startswith(name):
            ob.select_set(True)
        else:
            ob.select_set(False)
    O.object.delete()


def deg2rad(deg):
    return deg * math.pi / 180.0


def circular_loc(theta, phi):
    r = 3.
    loc_x = r * math.sin(theta) * math.cos(phi)
    loc_y = r * math.sin(theta) * math.sin(phi)
    loc_z = r * math.cos(theta)
    return loc_x, loc_y, loc_z


def move_camera(coord):
    cam = D.objects['Camera']
    C.view_layer.objects.active = cam

    theta, phi = deg2rad(coord[0]), deg2rad(coord[1])
    if theta == 0:
        theta = deg2rad(1)
    if phi == 0:
        phi = deg2rad(1)
    a = deg2rad(coord[2])

    loc_x, loc_y, loc_z = circular_loc(theta, phi)
    cam.location = (loc_x, loc_y, loc_z)
    cam.rotation_mode = 'AXIS_ANGLE'
    cam.rotation_axis_angle = (a, loc_x, loc_y, loc_z)

if __name__ == '__main__':
    main()


'''

# NOTES 

LineSetV.select_silhouette = True # soso
LineSetV.select_border = False # nothing
LineSetV.select_contour = False # contour
LineSetV.select_suggestive_contour = False # bad
LineSetV.select_ridge_valley = False # bad
LineSetV.select_crease = True # good
LineSetV.select_edge_mark = False # nothing
LineSetV.select_external_contour = False # contour
LineSetV.select_material_boundary = False # nothing
LineSetV.linestyle.thickness = 2.0          

'''
