import argparse
import os

def parse_args(description='Robot'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--assembly_name', default='input')
    parser.add_argument('--cutpath', default='./assemblies_300dpi')
    parser.add_argument('--resultpath', default='./results')  # intermediate
    parser.add_argument('--ocr_modelpath', default=os.path.join('./model','OCR'))
    parser.add_argument('--csv_dir', default='./output')
    parser.add_argument('--eval_print', default=False)
    parser.add_argument('--save_serial_whole', default=False)
    parser.add_argument('--save_serial_black', default=False)
    parser.add_argument('--save_serial_white', default=False)
    parser.add_argument('--print_serial_progress', default=True)
    parser.add_argument('--save_mult_black', default=False)
    parser.add_argument('--save_group_image', default=False)
    parser.add_argument('--gpu', default='0')

    opt = parser.parse_args()

    return opt

def init_args(description='Robot'):
    opt = parse_args(description)

    opt.cutpath = os.path.join(opt.cutpath, opt.assembly_name, 'cuts')
    opt.textpath = opt.resultpath

    opt.resultpath = os.path.join(opt.resultpath, opt.assembly_name)
    opt.circlepath = os.path.join(opt.resultpath, 'circle')
    opt.rectanglepath = os.path.join(opt.resultpath, 'rectangle')
    opt.actionpath = os.path.join(opt.resultpath, 'action')
    opt.serialpath = os.path.join(opt.resultpath, 'serial')
    opt.serial_whole_path = os.path.join(opt.resultpath, 'serial_whole')
    opt.serial_black_path = os.path.join(opt.resultpath, 'serial_black')
    opt.serial_white_path = os.path.join(opt.resultpath, 'serial_white')
    opt.multpath = os.path.join(opt.resultpath, 'mult')
    opt.mult_black_path = os.path.join(opt.resultpath, 'mult_black')
    opt.group_image_path = os.path.join(opt.resultpath, 'group_image')

    opt.csv_dir = os.path.join(opt.csv_dir, opt.assembly_name)

    opt.eval_print = string2bool(opt.eval_print)
    opt.save_serial_whole = string2bool(opt.save_serial_whole)
    opt.save_serial_black = string2bool(opt.save_serial_black)
    opt.save_serial_white = string2bool(opt.save_serial_white)
    opt.print_serial_progress = string2bool(opt.print_serial_progress)
    opt.save_mult_black = string2bool(opt.save_mult_black)
    opt.save_group_image = string2bool(opt.save_group_image)

    return opt

def string2bool(string):
    if type(string)==bool:
        bool_type = string
    else:
        assert string=='True' or string=='False'

        bool_type = True if string=='True' else False

    return bool_type

