import argparse
import os


def parse_args(description='Robot'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--assembly_name', default='stefan')
    parser.add_argument('--input_path', default='./input')
    parser.add_argument('--intermediate_results_path', default='./intermediate_results')
    parser.add_argument('--det_config1_name', default='./model/detection/fine_tuned/12.pickle')
    parser.add_argument('--det_config2_name', default='./model/detection/fine_tuned/3.pickle')
    parser.add_argument('--ocr_flip_weight_path', default=os.path.join('./model', 'OCR', 'flip_weight'))
    parser.add_argument('--ocr_class_weight_path', default=os.path.join('./model', 'OCR', 'weight'))
    parser.add_argument('--retrieval_model_path', default=os.path.join('./model', 'retrieval', 'ckpt'))
    parser.add_argument('--pose_model_path', default=os.path.join('./model', 'pose', 'mission1'))

    parser.add_argument('--csv_dir', default='./output')
    parser.add_argument('--eval_print', default=False)
    parser.add_argument('--save_detection', default=True)
    parser.add_argument('--save_serial', default=True)
    parser.add_argument('--save_serial_whole', default=False)
    parser.add_argument('--save_serial_black', default=False)
    parser.add_argument('--save_serial_white', default=False)
    parser.add_argument('--save_serial_npy', default=False)
    parser.add_argument('--print_serial_progress', default=False)
    parser.add_argument('--save_mult', default=True)
    parser.add_argument('--save_mult_npy', default=False)
    parser.add_argument('--save_mult_black', default=False)
    parser.add_argument('--save_group_image', default=False)
    parser.add_argument('--save_part_id_pose', default=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--print_time', default=False)

    parser.add_argument('--step_num', default=False, type=int)
    parser.add_argument('--add_cad', default=False)

    opt = parser.parse_args()

    return opt


def init_args(description='Robot'):
    opt = parse_args(description)

    opt.assembly_path = os.path.join(opt.input_path, opt.assembly_name)
    opt.cut_path = os.path.join(opt.input_path, opt.assembly_name, 'cuts')
    opt.cad_path = os.path.join(opt.input_path, opt.assembly_name, 'cad')
    opt.cadinfo_path = os.path.join(opt.input_path, opt.assembly_name, 'cad_info')
    opt.point_cloud_path = os.path.join(opt.cad_path, 'point_cloud')
    opt.retrieval_views_path = os.path.join(opt.input_path, opt.assembly_name, 'views', 'VIEWS_GRAY_BLACK')
    opt.pose_views_path = os.path.join(opt.input_path, opt.assembly_name, 'views', 'VIEWS')
    opt.pose_krt_path = os.path.join(opt.input_path, opt.assembly_name, 'views', 'KRT')
    opt.pose_krt_imgs_path = os.path.join(opt.input_path, opt.assembly_name, 'views', 'KRT', 'krt_imgs')
    opt.textpath = opt.intermediate_results_path

    opt.intermediate_results_path = os.path.join(opt.intermediate_results_path, opt.assembly_name)
    opt.detection_path = os.path.join(opt.intermediate_results_path, 'detection')
    opt.circle_path = os.path.join(opt.intermediate_results_path, 'circle')
    opt.rectangle_path = os.path.join(opt.intermediate_results_path, 'rectangle')
    opt.action_path = os.path.join(opt.intermediate_results_path, 'action')
    opt.serial_path = os.path.join(opt.intermediate_results_path, 'serial')
    opt.serial_whole_path = os.path.join(opt.intermediate_results_path, 'serial_whole')
    opt.serial_black_path = os.path.join(opt.intermediate_results_path, 'serial_black')
    opt.serial_white_path = os.path.join(opt.intermediate_results_path, 'serial_white')
    opt.mult_path = os.path.join(opt.intermediate_results_path, 'mult')
    opt.mult_black_path = os.path.join(opt.intermediate_results_path, 'mult_black')
    opt.group_image_path = os.path.join(opt.intermediate_results_path, 'group_image')
    opt.part_id_pose_path = os.path.join(opt.intermediate_results_path, 'part_id_pose')
    opt.part_hole_path = os.path.join(opt.intermediate_results_path, 'part_hole')

    opt.csv_dir = os.path.join(opt.csv_dir, opt.assembly_name)

    opt.eval_print = string2bool(opt.eval_print)
    opt.save_serial = string2bool(opt.save_serial)
    opt.save_serial_whole = string2bool(opt.save_serial_whole)
    opt.save_serial_black = string2bool(opt.save_serial_black)
    opt.save_serial_white = string2bool(opt.save_serial_white)
    opt.save_serial_npy = string2bool(opt.save_serial_npy)
    opt.save_mult = string2bool(opt.save_mult)
    opt.save_mult_npy = string2bool(opt.save_mult_npy)
    opt.save_mult_black = string2bool(opt.save_mult_black)
    opt.print_serial_progress = string2bool(opt.print_serial_progress)
    opt.save_mult_black = string2bool(opt.save_mult_black)
    opt.save_group_image = string2bool(opt.save_group_image)

    return opt


def string2bool(string):
    if type(string) == bool:
        bool_type = string
    else:
        assert string == 'True' or string == 'False'

        bool_type = True if string == 'True' else False

    return bool_type
