import argparse
import os


def parse_args(description='Robot'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--assembly_name', default='stefan')
    parser.add_argument('--input_path', default='./input')
    parser.add_argument('--intermediate_results_path', default='./intermediate_results')
    parser.add_argument('--det_config1_name', default='./model/detection/fine_tuned/12.pickle')
    parser.add_argument('--det_config2_name', default='./model/detection/mission2/parts_1.pickle')
    parser.add_argument('--parts_threshold', default=0.7, type=float, help='confidence threshold value for parts detection')
    parser.add_argument('--ocr_flip_weight_path', default=os.path.join('./model', 'OCR', 'flip_weight'))
    parser.add_argument('--ocr_class_weight_path', default=os.path.join('./model', 'OCR', 'weight'))
    parser.add_argument('--retrieval_model_path', default=os.path.join('./model', 'retrieval', 'ckpt'))
    parser.add_argument('--pose_model_path', default=os.path.join('./model', 'pose', 'mission2'))
    parser.add_argument('--pose_model_adr', default='', help='If defined, overrides pose_model_path')
    parser.add_argument('--fastener_model_path', default=os.path.join('./model', 'fastener', 'mission2'))
    parser.add_argument('--part4dir_model_path', default=os.path.join('./model', 'part4dir'))

    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--eval_print', type=str2bool, default=False)
    parser.add_argument('--save_detection', type=str2bool, default=True)
    parser.add_argument('--save_serial', type=str2bool, default=True)
    parser.add_argument('--save_serial_whole', type=str2bool, default=False)
    parser.add_argument('--save_serial_black', type=str2bool, default=False)
    parser.add_argument('--save_serial_white', type=str2bool, default=False)
    parser.add_argument('--save_serial_npy', type=str2bool, default=False)
    parser.add_argument('--print_serial_progress', type=str2bool, default=False)
    parser.add_argument('--save_mult', type=str2bool, default=True)
    parser.add_argument('--save_mult_npy', type=str2bool, default=False)
    parser.add_argument('--save_mult_black', type=str2bool, default=False)
    parser.add_argument('--save_circle', type=str2bool, default=False)
    parser.add_argument('--save_rectangle', type=str2bool, default=False)
    parser.add_argument('--save_group_image', type=str2bool, default=True)
    parser.add_argument('--save_part_image', type=str2bool, default=True)
    parser.add_argument('--save_pose_prediction_maps', type=str2bool, default=True)
    parser.add_argument('--save_pose_visualization', type=str2bool, default=True)
    parser.add_argument('--save_part4dir', type=str2bool, default=True)
    parser.add_argument('--save_part_id_pose', type=str2bool, default=True)
    parser.add_argument('--save_fastener_prediction_maps', type=str2bool, default=True)
    parser.add_argument('--save_fastener_visualization', type=str2bool, default=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--print_time', type=str2bool, default=False)
    parser.add_argument('--mid_RT_on', type=str2bool, default=True)
    parser.add_argument('--hole_detection_on', type=str2bool, default=True)
    parser.add_argument('--use_fine_tuned_model_detection', type=str2bool, default=True)
    parser.add_argument('--use_challenge_detection_model', type=str2bool, default=False)
    parser.add_argument('--use_challenge_pose_model', type=str2bool, default=False)
    parser.add_argument('--use_part4_direction_estimation', type=str2bool, default=True)
    parser.add_argument('--save_pose_indiv', type=str2bool, default=True)
    parser.add_argument('--print_pose_flip', type=str2bool, default=False)
    parser.add_argument('--no_hole', type=str2bool, default=False, help='skip hole, action grouping')

    ############# temp ###########
    parser.add_argument('--temp', type=str2bool, default=False)
    ###############################
    ######### print_pred ###########
    parser.add_argument('--print_pred', type=str2bool, default=False)
    parser.add_argument('--pred_dir', default=None)
    ################################
    parser.add_argument('--blender', default='blender')

    parser.add_argument('--step_num', default=False, type=int)

    parser.add_argument('--starting_cut', default=-1, type=int)

    opt = parser.parse_args()

    return opt


def init_args(description='Robot'):
    opt = parse_args(description)

    opt.assembly_path = os.path.join(opt.input_path, opt.assembly_name)
    opt.hole_path = os.path.join(opt.assembly_path, 'cad_info')
    opt.hole_path_2 = os.path.join(opt.assembly_path, 'cad_info2')
    opt.cut_path = os.path.join(opt.input_path, opt.assembly_name, 'cuts')
    opt.cad_path = os.path.join(opt.input_path, opt.assembly_name, 'cad')
    opt.point_cloud_path = os.path.join(opt.cad_path, 'point_cloud')
    opt.retrieval_views_path = os.path.join(opt.input_path, opt.assembly_name, 'views', 'VIEWS_GRAY_BLACK')
    opt.pose_views_path = os.path.join(opt.input_path, opt.assembly_name, 'views', 'VIEWS')
    opt.pose_krt_path = os.path.join(opt.input_path, opt.assembly_name, 'views', 'KRT')
    opt.pose_krt_imgs_path = os.path.join(opt.input_path, opt.assembly_name, 'views', 'KRT', 'krt_imgs')
    opt.pose_data_path = './function/Pose/data'
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
    opt.part_image_path = os.path.join(opt.intermediate_results_path, 'part_image')
    opt.part_id_pose_path = os.path.join(opt.intermediate_results_path, 'part_id_pose')
    opt.part_hole_path = os.path.join(opt.intermediate_results_path, 'part_hole')
    opt.initial_pose_estimation_path = os.path.join(opt.intermediate_results_path, 'initial_pose_estimation')
    opt.initial_pose_estimation_adr = os.path.join(opt.initial_pose_estimation_path, 'initial_pose_estimation.json')
    opt.initial_pose_estimation_prediction_maps_path = os.path.join(opt.initial_pose_estimation_path, 'prediction_maps')
    opt.initial_pose_estimation_visualization_path = os.path.join(opt.initial_pose_estimation_path, 'visualization')
    opt.initial_pose_estimation_part4dir_path = os.path.join(opt.initial_pose_estimation_path, 'part4_dir')
    opt.fastener_detection_path = os.path.join(opt.intermediate_results_path, 'fastener_detection')
    opt.fastener_detection_prediction_maps_path = os.path.join(opt.fastener_detection_path, 'prediction_maps')
    opt.fastener_detection_visualization_path = os.path.join(opt.fastener_detection_path, 'visualization')

    opt.output_dir = os.path.join(opt.output_dir, opt.assembly_name)

    opt.eval_print = str2bool(opt.eval_print)
    opt.save_serial = str2bool(opt.save_serial)
    opt.save_serial_whole = str2bool(opt.save_serial_whole)
    opt.save_serial_black = str2bool(opt.save_serial_black)
    opt.save_serial_white = str2bool(opt.save_serial_white)
    opt.save_serial_npy = str2bool(opt.save_serial_npy)
    opt.save_mult = str2bool(opt.save_mult)
    opt.save_mult_npy = str2bool(opt.save_mult_npy)
    opt.save_mult_black = str2bool(opt.save_mult_black)
    opt.print_serial_progress = str2bool(opt.print_serial_progress)
    opt.save_mult_black = str2bool(opt.save_mult_black)
    opt.save_detection = str2bool(opt.save_detection)
    opt.save_circle = str2bool(opt.save_circle)
    opt.save_rectangle = str2bool(opt.save_rectangle)
    opt.save_group_image = str2bool(opt.save_group_image)
    opt.save_part_image = str2bool(opt.save_part_image)
    opt.save_pose_prediction_maps = str2bool(opt.save_pose_prediction_maps)
    opt.save_part_id_pose = str2bool(opt.save_part_id_pose)
    opt.mid_RT_on = str2bool(opt.mid_RT_on)
    opt.hole_detection_on = str2bool(opt.hole_detection_on)
    opt.use_fine_tuned_model_detection = str2bool(opt.use_fine_tuned_model_detection)

    if not opt.use_fine_tuned_model_detection:
        opt.det_config2_name = './model/detection/mission2/parts_nft.pickle'
    if opt.use_challenge_detection_model:
        opt.det_config2_name = './model/detection/mission2/parts.pickle'
    if opt.use_challenge_pose_model:
        opt.pose_model_adr = './model/pose/mission2/correspondence_block_finetuned.pt'

    ####### Temp ########
    opt.temp = str2bool(opt.temp)
    #####################

    return opt


def str2bool(string):
    if type(string) == bool:
        bool_type = string
    else:
        assert string == 'True' or string == 'False'
        bool_type = True if string == 'True' else False
    return bool_type


if __name__ == '__main__':
    args = init_args()
    for k, v in sorted(args.__dict__.items()):
        print(k, '=', v)
