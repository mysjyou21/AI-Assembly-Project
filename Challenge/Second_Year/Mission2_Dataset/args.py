import argparse
import os
import shutil

def define_args():
    parser = argparse.ArgumentParser()
    # important settings
    parser.add_argument('--mode', default='2345', help='1: scene generation, 2: rendering, 3: post-processing, 4: labels')
    parser.add_argument('--num_scene', type=int, default=3, help='number of scenes to generate')
    parser.add_argument('--num_cad_range', default='2,5', help='number of cads to put in scene. min, max')
    parser.add_argument('--delete_previous', type=bool, default=False, help='delete previous dataset results')
    parser.add_argument('--cloudcompare', default='CloudCompare', help='set to your CloudCompare command')
    parser.add_argument('--blender', default='blender', help='set to your blender command')
    parser.add_argument('--scene_name', default='', help='for example, "2-5". If scene_name is specified, render only single scene')


    # settings
    parser.add_argument('--rendering_size', default='2480,1748', help='size of rendering (h,w)')
    parser.add_argument('--scale', type=int, default=100, help='cad_model = 1/Scale * cad_model (original cad is too big)')
    parser.add_argument('--mute', type=bool, default=False, help='mute blender output')

    # input
    parser.add_argument('--model_type', default='cad', help='model type')

    # intermediate results
    parser.add_argument('--intermediate_results_path', default='./intermediate_results', help='intermeidate results path')
    parser.add_argument('--point_cloud_path', default='./intermediate_results/point_cloud', help='randomly sampled points on cad suface path')
    parser.add_argument('--krt_image_path', default='./intermediate_results/krt_image', help='create images of pointcloud transformed with K, R, t in labels')
    parser.add_argument('--views_path', default='./intermediate_results/VIEWS', help='line rendering with white background path')
    parser.add_argument('--views_gray_black_path', default='./intermediate_results/VIEWS_GRAY_BLACK', help='surface rendering with gray object and black background path')
    parser.add_argument('--views_gray_black_occlu_path', default='./intermediate_results/VIEWS_GRAY_BLACK_OCCLU', help='surface rendering with gray object and black background path (occlusion)')
    parser.add_argument('--masked_components_path', default='./masked_components', help='non part components path')

    # json files
    parser.add_argument('--json', default='scene_models', help='name of json with scene gt')

    # output
    parser.add_argument('--train_type', default='blender', help='train type')
    
    args = parser.parse_args()
    args = add_args(args)
    if args.delete_previous:
        delete_previous(args)
    ensure_paths(args)
    return args

def add_args(args):
    # input path
    args.cad_path = './models_' + args.model_type # cad models directory
    # output path
    args.output_path = './train_' + args.train_type # output directory
    # masked components path
    args.cut_num_path = os.path.join(args.masked_components_path, 'cut_num')
    args.bottom_num_path = os.path.join(args.masked_components_path, 'bottom_num')
    # json files
    parsed_scene_models_name = args.json.split('_')
    if len(parsed_scene_models_name) <= 2:
        args.label_json = 'label'
    else:
        args.label_json = 'label_' + '_'.join(parsed_scene_models_name[2:])
    return args

def delete_previous(args):
    """ deleted previously created dataset files"""
    path_list = [args.intermediate_results_path, args.output_path]
    file_list = []
    for path in path_list:
        try:
            shutil.rmtree(path)
        except:
            pass
    for file in file_list:
        try:
            os.remove(file)
        except:
            pass

def ensure_paths(args):
    """create directories for path arguments in args
    
    argument name must include 'path' or 'dir'
    """
    path_list = []
    for arg in vars(args):
        if 'path' in arg or 'dir' in arg:
            path_list.append(getattr(args, arg))
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)

if __name__ == '__main__':
    """create directories, print arguments """
    args = define_args()
    print(args)