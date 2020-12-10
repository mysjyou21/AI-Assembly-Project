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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from mathutils import Matrix
from scipy.spatial.transform import Rotation as Rot
from copy import deepcopy


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-cad_path', default='./input/stefan/cad', help='cad input path')
    parser.add_argument('-json_path', default='./input/stefan/cad', help='json path')
    parser.add_argument('-cad_adrs', default='', help='only added cad_adrs')
    parser.add_argument('-initial', type=str2bool, default=True, help='True: ignore existed json file, False: load existed json file')
    args = parser.parse_known_args(argv)[0]


print('# render views.py')
for k, v in sorted(args.__dict__.items()):
    print(k, '=', v, '({})'.format(type(v)))


D = bpy.data
C = bpy.context
O = bpy.ops


def main():
    if (not args.initial) and os.path.exists(os.path.join(args.json_path, 'center.json')):
        with open(os.path.join(args.json_path, 'center.json'), 'r') as f:
            obj_centers = json.load(f)
    else:
        obj_centers = {}

    if args.cad_adrs == '':
        obj_files = glob.glob(os.path.join(args.cad_path, '*.STL')) + glob.glob(os.path.join(args.cad_path, '*.obj'))
    else:
        cad_adrs = args.cad_adrs.split(',')
        obj_files = [os.path.join(args.cad_path, x) for x in cad_adrs]
    obj_files = sorted(obj_files)
    print(obj_files)

    for obj_file in obj_files:
        obj_name = os.path.basename(obj_file).replace('.STL','').replace('.obj','')
        if 'obj' in obj_name:
            set_models_to_default_pose(obj_file)
        init_scene()
        load_models(obj_file)
        center = find_scene_surface_center()
        obj_centers[obj_name] = center

    with open(os.path.join(args.json_path, 'center.json'), 'w') as f:
        json.dump(obj_centers, f, indent=2)

def init_scene():
    """ Delete all components in scene. Add origin at (0, 0, 0)"""
    O.object.select_all(action='SELECT')
    O.object.delete()

def load_models(cad_adr): #scene, model_index=None):
    """ load model """
    cad_name = os.path.basename(cad_adr).split('.')[0]
    print('loading: '+ cad_name)

    file_type = os.path.basename(cad_adr).split('.')[1]
    if file_type == 'obj':
        O.import_scene.obj(filepath=cad_adr, filter_glob='/*.'+file_type)
    elif file_type == 'STL':
        O.import_mesh.stl(filepath=cad_adr, filter_glob='/*.'+file_type)
    else:
        print('Not available type cad file')

#    for obj in [x for x in D.objects]:
#        obj.name = cad_name

#    # discard .001 when reimporting
#    D_objects = [x for x in D.objects]
#    for obj in D_objects:
#        obj.name = obj.name.split('.')[0]


def shrink_models():
    """ scale model """
    model_objects = [obj for obj in D.objects]

    for obj in model_objects:
        dim = obj.dimensions
        print('original dim:' + str(dim))
        dim = dim / int(args.scale)
        obj.dimensions = dim
        print('new dim:' + str(dim))


def set_models_to_default_pose(cad_adr): #scene, model_index=None):
    """ read default pose from json, and set models """
    assert len(D.objects) == 1
    obj = D.objects[0]
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = [1, 0, 0, 0]

#    cad_adrs = sorted(glob.glob(args.cad_path + '/*.obj'))
#    cad_names = [os.path.basename(x).split('.')[0] for x in cad_adrs]
#    for i, model in enumerate(scene):
#        if model_index is not None:
#            # single model
#            if i != model_index:
#                continue
##        cad_name = cad_names[model['obj_id'] - args.OBJ_ID_CORRECTION]
##        obj_name = str(i) + '@' + cad_name
##        obj = D.objects[obj_name]
#        obj.rotation_mode = 'QUATERNION'
##        axis_angle = model['axisangle_WXYZ'].copy()
##        axis_angle[0] = deg2rad(axis_angle[0])
##        rotation = Rot.from_rotvec(axis_angle[0] * np.array(axis_angle[1:]))
##        quaternion = np.roll(rotation.as_quat(), 1)
#        obj.rotation_quaternion = [1, 0, 0, 0] #quaternion
#        lx, ly, lz = model['location_XYZ'].copy()
#        obj.location = (lx / int(args.scale), ly / int(args.scale), lz / int(args.scale))

def deg2rad(deg):
    return deg * math.pi / 180.0


def delete_scene():
    """ delete model """
    for obj in D.objects:
        if obj.name != 'Camera' and obj.name != 'Origin':
            D.objects.remove(obj)
    for mat in D.materials:
        if mat.name != 'Material':
            D.materials.remove(mat)


def find_scene_surface_center():
    """ set origin of all model objects to surface center """
    #Deselect all
    O.object.select_all(action='DESELECT')
    objs = [obj for obj in D.objects if obj.type == 'MESH']
    for obj in objs:
        # Select all mesh objects
        obj.select_set(True)
    # Make one active
    C.view_layer.objects.active = obj
    # Join objects
#    O.object.join()
    O.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    objs = [obj for obj in D.objects]
    assert len(objs) == 1, 'More than one object'
    scene = objs[0]
    surface_center = [x for x in deepcopy(scene.location)]
#    surface_center = args.scale * np.array([x for x in deepcopy(scene.location)])

    return surface_center


if __name__ == '__main__':
    main()

