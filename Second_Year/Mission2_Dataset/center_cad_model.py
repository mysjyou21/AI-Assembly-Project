# file runs in blender
import bpy
import os
import sys
import glob
import time
import argparse
import numpy as np

if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-cad_path', help='cad input path. output path is the same')
    args = parser.parse_known_args(argv)[0]


D = bpy.data
C = bpy.context
O = bpy.ops

# delete all objects in Blender
O.object.select_all(action='SELECT')
O.object.delete()


def main():
    # import .obj
    files = sorted(glob.glob(os.path.join(args.cad_path, '*.obj')))
    # center model
    for file in files:
        create_origin()
        do_model(file)

def create_origin():
    """ create origin """
    origin_name = 'Origin'
    try:
        origin = D.objects[origin_name]
    except KeyError:
        O.object.empty_add(type='SPHERE')
        D.objects['Empty'].name = origin_name
        origin = D.objects[origin_name]
        pass
    origin.location = (0, 0, 0)

def do_model(path):
    # load model
    name = load_model(path)
    if D.objects[0] == D.objects['Origin']:
        model_object = D.objects[1]
    elif D.objects[1] == D.objects['Origin']:
        model_object = D.objects[0]
    else:
        print(D.objects)
        print(D.objects[0])
        print(D.objects[1])
        raise Exception('blender model loading error')
    # center model
    center_model(model_object)
    # save model
    save_model(path)
    # delete model
    delete_model()


def load_model(path):
    """load model"""
    d = os.path.dirname(path)
    ext = path.split('.')[-1]
    name = os.path.basename(path).split('.')[0]
    if name not in D.objects:
        print('loading :' + name)
        O.import_scene.obj(filepath=path, filter_glob='*.obj')
    return name


def center_model(obj):
    """center model"""
    O.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    obj.location = (0, 0, 0)


def save_model(save_path):
    """save model"""
    for ob in C.scene.objects:
        if ob.type == 'MESH':
            ob.select_set(True)
        else:
            ob.select_set(False)
    O.export_scene.obj(filepath=save_path, use_selection=True, use_materials=False)

def delete_model():
    """delete model"""
    for ob in C.scene.objects:
        if ob.type == 'MESH':
            ob.select_set(True)
        else:
            ob.select_set(False)
    O.object.delete()

if __name__ == '__main__':
    main()

