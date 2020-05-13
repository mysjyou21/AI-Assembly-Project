import os
import glob
import sys
import time
import argparse
import shutil

parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', '-n', default='stefan_parts', help='folder name in ../../data to render')
parser.add_argument('--data_path', '-d', default='../../data', help='data folder directory')
parser.add_argument('--render_type', '-r', default='views_gray', help='rendering type. {views, views_gray, views_black, views_gray_black}')
parser.add_argument('--cad_type', '-c', default='obj', help='input CAD file extension. {off, step, obj, igs, stl, ply}')
parser.add_argument('--mute', '-m', default='True', help='mute rendering(blender) output {True, False}')
args = parser.parse_known_args()[0]


#----------------------------------------------------------#
# configuration

# render output type
assert args.render_type in ['views', 'views_gray', 'views_black', 'views_gray_black']

# input cad file type
assert args.cad_type in ['off', 'step', 'obj', 'igs', 'iges', 'stl', 'ply']

# input dir / output dir
folder_path = os.path.join(args.data_path, args.name)
target_models_path = os.path.join(folder_path, 'TARGET_MODELS/models')
images_path = os.path.join(folder_path, 'IMAGES')
views_path = os.path.join(folder_path, 'VIEWS')
views_black_path = os.path.join(folder_path, 'VIEWS_BLACK')
views_gray_path = os.path.join(folder_path, 'VIEWS_GRAY')
views_gray_black_path = os.path.join(folder_path, 'VIEWS_GRAY_BLACK')
path_list = [images_path, views_path, views_gray_path, views_black_path, views_gray_black_path]


# rendered output resolution
H, W = 224, 224

# print list of loaded cad files at end of execution
print_filelist = True

# mute rendering output
assert args.mute in ['True', 'False']

#----------------------------------------------------------#


def main():
    # create subfolders
    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)

    # 2d-projection rendering (obj --> png)
    print(f'Rendering {(args.render_type).upper()}...')
    if args.render_type == 'views':
        render_output_path = views_path
    elif args.render_type == 'views_black':
        render_output_path = views_black_path
    elif args.render_type == 'views_gray':
        render_output_path = views_gray_path
    elif args.render_type == 'views_gray_black':
        render_output_path = views_gray_black_path
    else:
        raise Exception('wrong render type')

    # CAD file type conversion (off, step --> obj)
    cad_ext_dict = {'off': '.off', 'step': '.STEP', 'obj': '.obj', 'igs': '.IGS', 'iges': '.IGS', 'stl': '.STL', 'ply': '.ply'}
    if args.cad_type == 'off':  # .off
        off_files = sorted(glob.glob(os.path.join(target_models_path, '*.off')))
        for off_file in off_files:
            os.system('off2obj ' + off_file[:-4] + '.off -o ' + off_file[:-4] + '.obj')
    elif args.cad_type == 'step':  # .step
        flag = '-cad_ext ' + cad_ext_dict[args.cad_type] + ' -target_models_path ' + target_models_path
        os.system('freecadcmd convert_cad_format.py -- ' + flag)
    elif args.cad_type == 'igs':  # .igs
        flag = '-cad_ext ' + cad_ext_dict[args.cad_type] + ' -target_models_path ' + target_models_path
        os.system('freecadcmd convert_cad_format.py -- ' + flag)
    elif args.cad_type == 'stl':  # .stl
        flag = '-cad_ext ' + cad_ext_dict[args.cad_type] + ' -target_models_path ' + target_models_path
    elif args.cad_type == 'obj':  # .obj
        pass

    flag = '-render_type ' + args.render_type + ' -target_models_path ' + target_models_path + ' -render_output_path ' + render_output_path + ' -cad_ext ' + cad_ext_dict[args.cad_type]
    if args.mute == 'True':
        os.system('blender -b -P render_views.py 1 > nul -- ' + flag)
    else:
        os.system('blender -b -P render_views.py -- ' + flag)

    # create cla. files
    flag = '-folder_path ' + folder_path
    os.system('python create_cla_files.py -- ' + flag)

    # print stdout
    print()
    print('\n------------------------------\n')
    filelist = sorted(glob.glob(os.path.join(target_models_path, '*' + cad_ext_dict[args.cad_type])))
    print("Finished '''%s''' Rendering" % args.render_type)
    print()
    print("from %s type file" % cad_ext_dict[args.cad_type])
    if print_filelist:
        print(filelist)
    print()
    print('target_models_path = %s' % target_models_path)
    print('render_output_path = %s' % render_output_path)
    print()


if __name__ == '__main__':
    tic = time.time()
    main()
    toc = time.time()
    print('total %f sec' % (toc - tic))
    print('\n------------------------------\n')
