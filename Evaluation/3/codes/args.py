import argparse
import os
import shutil
from function.utils import *

def define_args(description='2020 Quantitative Report (Part 3 : Pose)'):
    parser = argparse.ArgumentParser(description=description)
    
    # mode
    parser.add_argument('--mode', default='test', 
        help='\
        mode in ...\
        \
        detection : run until detection output\
        retrieval : run until retrieval output\
        pose : run until pose output\
        \
        detection_unit_test : unit test detection (same as mode==detection)\
        retrieval_unit_test : unit test retrieval (GT detection output)\
        pose_unit_test      : unit test pose (GT detection,retrieval output)\
        \
        test_data : create test data (npy)\
        test      : execute test (no image output)'
        )

    # path
    parser.add_argument('--data_path', default='../data')
    parser.add_argument('--model_path', default='./model', help='default directory for model weights')
    parser.add_argument('--intermediate_results_path', default='./intermediate_results', help='directory to save unit test outputs')
    parser.add_argument('--output_path', default='./output', help='final output path')

    # options
    parser.add_argument('--detection_visualization', type=str2bool, default=False, help='Whether to save intermediate_results images to visualize the result.')
    parser.add_argument('--retrieval_visualization', type=str2bool, default=False, help='Whether to save intermediate_results images to visualize the result.')
    parser.add_argument('--pose_visualization', type=str2bool, default=False, help='Whether to save intermediate_results images to visualize the result.')
    parser.add_argument('--output_visualization', type=str2bool, default=False, help='Whether to save output images to visualize the test result.')
    parser.add_argument('--delete_previous', type=str2bool, default=True, help='Whether to delete previously created intermediate_results/outputs')

    # settings
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--detection_bbox_margin', type=int, default=10, help='how much to enlarge detection bbox')

    args = parser.parse_args()
    args = add_args(args)
    if args.delete_previous:
        delete_previous(args)
    ensure_paths(args)
    return args


def add_args(args):
    """add more arguments based on initially defined arguments
    
    Args:
        args: (namespace) args
    
    Returns:
        (namespace) args
    """
    # model path
    args.detection_model_path = os.path.join(args.model_path, 'detection')
    args.detection_config_path = os.path.join(args.model_path, 'detection/3.pickle')
    args.retrieval_model_path = os.path.join(args.model_path, 'retrieval')
    args.pose_model_path = os.path.join(args.model_path, 'pose')

    # intermediate resutls path
    args.detection_intermediate_results_path = os.path.join(args.intermediate_results_path, 'detection')
    args.retrieval_intermediate_results_path = os.path.join(args.intermediate_results_path, 'retrieval')
    args.pose_intermediate_results_path = os.path.join(args.intermediate_results_path, 'pose')

    # test data path
    # general
    args.binary_path = os.path.join(args.data_path, 'input', 'npy')
    args.cad_path = os.path.join(args.data_path, 'input', 'cad')
    args.image_path = os.path.join(args.data_path, 'input', 'image')
    args.label_path = os.path.join(args.data_path, 'input', 'label')
    args.input_bbox_path = os.path.join(args.data_path, 'input', 'bbox_answer')
    # detection
    args.detection_anns_label_path = os.path.join(args.data_path, 'detection', 'bbox_answer')
    # retrieval
    args.retrieval_gt_path = os.path.join(args.data_path, 'retrieval', 'ground_truth')
    # pose
    args.pose_data_path = args.data_path + '/pose'
    args.cornerpoints_adr = args.pose_data_path + '/cornerpoints.npy'
    args.RTs_adr = args.pose_data_path + '/RTs.npy'
    args.view_imgs_adr = args.pose_data_path + '/view_imgs.npy'
    
    return args


def delete_previous(args):
    """delete previously created intermediate_results / outputs

    delete folders:
        intermediate_results
        output
    delete files:
        None
    
    Args:
        args: (namespace) args
    """
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


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    """check wheter arguments are well defined"""
    args = define_args()
    print('==========')
    print('   ARGS   ')
    print('==========')
    for key, val in args.__dict__.items():
        print(key, '=', val)
