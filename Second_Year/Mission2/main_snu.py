from Robot import Assembly
import glob
import os
import cv2
import time
import platform
import pickle

from config import *
opt = init_args()

AUTO=False
new_cad_list = [['step1_a.STL', 'step1_b.STL'], ['step2.STL'], ['step3.STL'], ['step4.STL'], ['step5.STL'], ['step6.STL'], ['step7.STL'], ['step8.STL']]

def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    total_start_time = time.time()

    # (Temp) Isaac
    if not os.path.exists(os.path.join(os.getcwd(), 'model')):
        print('copying model folder from .. to .')
        if platform.system() == 'Windows':
            os.system('Xcopy /E /I /y ..\model .\model')
        else:
            os.system('cp -r ' + os.getcwd() + '/../model ' + os.getcwd() + '/model')
    print('----------------')
    print('Loading Weights')
    print('----------------')
    IKEA = Assembly(opt)
    print('--------------')
    print('gpu : ', opt.gpu)
    print('num_steps : ', IKEA.num_steps)
    print('--------------')

    list_prev_obj = []
    list_prev_stl = []

    if not opt.step_num:
        start_step = 1
    else:
        start_step = opt.step_num

    def get_elapsed_time(toc, tic):
        sec = int((toc - tic) % 60)
        minute = int((toc - tic) // 60)
        return minute, sec

    save_cad_center(initial=True)

    for step in range(start_step, IKEA.num_steps + 1):
        print('\n\n(step {})\n'.format(step))
        step_start_time = time.time()
        # IKEA.detect_step_component(step) ?? # 이삭 to 민우 : 이거 실수 맞나?

        if opt.step_num:

            # Restore Information of prior step
            pickle_read_filepath =  opt.cad_path + '/' + str(step-1) + '/info_dict_'+ str(step-1) + '.pickle'
            with open(pickle_read_filepath,'rb') as f:
                pickle_data = pickle.load(f)

            print("{} Step Information Restored".format(step-1))
            IKEA.circles_loc, IKEA.circles_separated_loc, IKEA.rectangles_loc, IKEA.connectors_serial_imgs, \
            IKEA.connectors_serial_loc, IKEA.connectors_mult_imgs, IKEA.connectors_mult_loc, IKEA.connectors_loc, \
            IKEA.parts_loc, IKEA.tools_loc, IKEA.is_merged, IKEA.is_tool, IKEA.connectors_serial_OCR, IKEA.connectors_mult_OCR, \
            IKEA.parts, IKEA.parts_info, IKEA.cad_models, IKEA.candidate_classes, IKEA.actions, IKEA.step_action, IKEA.unused_parts, IKEA.used_parts \
                = pickle_data_loader(mode="download", pickle_data=pickle_data)

            # if opt.add_cad:
            #     print("Need to add Mid from the prior step")
            #     list_prev_obj = sorted(glob.glob(os.path.join(IKEA.opt.cad_path, '*.obj')))
            #     list_prev_obj = [os.path.basename(x) for x in list_prev_obj]
            #     list_prev_stl = sorted(glob.glob(os.path.join(IKEA.opt.cad_path, '*.STL')))
            #     list_prev_stl = [os.path.basename(x) for x in list_prev_stl]

        opt.step_num = False
        WAIT_SIGNAL = False  # 챌린지 환경에서는 True
        SIGNAL = False
        list_added_obj = []
        list_added_stl = []

        IKEA.detect_step_component(step)

        if WAIT_SIGNAL:
            while not SIGNAL:
                print('Waiting Signal ...', end='\r')
                time.sleep(0.5)
                list_update_obj = sorted(glob.glob(os.path.join(IKEA.opt.cad_path, '*.obj')))
                list_update_obj = [os.path.basename(x) for x in list_update_obj]
                if len(list_prev_obj) < len(list_update_obj):
                    list_added_obj = [x for x in list_update_obj if x not in list_prev_obj]
                    list_prev_obj = list_update_obj.copy()
                    print('list_added_obj :', list_added_obj)
                    SIGNAL = True
                list_update_stl = sorted(glob.glob(os.path.join(IKEA.opt.cad_path, '*.STL')))
                list_update_stl = [os.path.basename(x) for x in list_update_stl]
                if len(list_prev_stl) < len(list_update_stl):
                    list_added_stl = [x for x in list_update_stl if x not in list_prev_stl]
                    list_prev_stl = list_update_stl.copy()
                    print('list_added_stl :', list_added_stl)
                    SIGNAL = True
        if WAIT_SIGNAL:
#            IKEA.rendering(step, list_added_obj, list_added_stl)
            IKEA.predict_pose(step)
            list_prev_obj = sorted(glob.glob(os.path.join(IKEA.opt.cad_path, '*.obj')))
            list_prev_obj = [os.path.basename(x) for x in list_prev_obj]
        else:
            IKEA.predict_pose(step)

        
        if step > 2 and opt.mid_RT_on:
            IKEA.group_RT_mid(step)
            if opt.hole_detection_on:
                try:
                    IKEA.msn2_hole_detector(step)
                except IndexError:
                    pass

        IKEA.group_as_action(step)

        print(IKEA.step_action)
        # IKEA.write_csv_mission(step, option=0)

        # dictionary Info Backup per step
        backup_data = [IKEA.circles_loc, IKEA.circles_separated_loc, IKEA.rectangles_loc, IKEA.connectors_serial_imgs, \
            IKEA.connectors_serial_loc, IKEA.connectors_mult_imgs, IKEA.connectors_mult_loc, IKEA.connectors_loc, \
            IKEA.parts_loc, IKEA.tools_loc, IKEA.is_merged, IKEA.is_tool, IKEA.connectors_serial_OCR, IKEA.connectors_mult_OCR, \
            IKEA.parts, IKEA.parts_info, IKEA.cad_models, IKEA.candidate_classes, IKEA.actions, IKEA.step_action, IKEA.unused_parts, IKEA.used_parts]
        info_dict = pickle_data_loader(mode="upload", backup_data=backup_data)

        pickle_filepath = IKEA.part_dir+'/info_dict_'+str(step)+'.pickle'
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(info_dict, f)

        step_end_time = time.time()
        step_min, step_sec = get_elapsed_time(step_end_time, step_start_time)
        total_min, total_sec = get_elapsed_time(step_end_time, total_start_time)
        if opt.print_time:
            print('step time : {} min {} sec'.format(step_min, step_sec))
            print('total time : {} min {} sec'.format(total_min, total_sec))
        if WAIT_SIGNAL and AUTO and step<9:
            for p in new_cad_list[step-1]:
                os.system('mv '+os.path.join(opt.assembly_path, p)+' '+os.path.join(opt.cad_path, p))
        


def pickle_data_loader(mode, pickle_data=None, backup_data=None):
    if mode == "download":
        circles_loc = pickle_data['circles_loc']
        circles_separated_loc = pickle_data['circles_separated_loc']
        rectangles_loc = pickle_data['rectangles_loc']
        connectors_serial_imgs = pickle_data['connectors_serial_imgs']
        connectors_serial_loc = pickle_data['connectors_serial_loc']
        connectors_mult_imgs = pickle_data['connectors_mult_imgs']
        connectors_mult_loc = pickle_data['connectors_mult_loc']
        connectors_loc = pickle_data['connectors_loc']
        parts_loc = pickle_data['parts_loc']
        tools_loc = pickle_data['tools_loc']
        is_merged = pickle_data['is_merged']
        is_tool = pickle_data['is_tool']
        connectors_serial_OCR = pickle_data['connectors_serial_OCR']
        connectors_mult_OCR = pickle_data['connectors_mult_OCR']
        parts = pickle_data['parts']
        parts_info = pickle_data['parts_info']
        cad_models = pickle_data['cad_models']
        candidate_classes = pickle_data['candidate_classes']
        actions = pickle_data['actions']
        step_action = pickle_data['step_action']
        unused_parts = pickle_data['unused_parts']
        used_parts = pickle_data['used_parts']
        return circles_loc, circles_separated_loc, rectangles_loc, connectors_serial_imgs, connectors_serial_loc,\
            connectors_mult_imgs, connectors_mult_loc, connectors_loc, parts_loc, tools_loc, is_merged, is_tool,\
            connectors_serial_OCR, connectors_mult_OCR, parts, parts_info, cad_models, candidate_classes, actions, step_action, unused_parts, used_parts

    if mode == "upload":
        info_dict = {}
        info_dict['circles_loc'] = backup_data[0]
        info_dict['circles_separated_loc'] = backup_data[1]
        info_dict['rectangles_loc'] = backup_data[2]
        info_dict['connectors_serial_imgs'] = backup_data[3]
        info_dict['connectors_serial_loc'] = backup_data[4]
        info_dict['connectors_mult_imgs'] = backup_data[5]
        info_dict['connectors_mult_loc'] = backup_data[6]
        info_dict['connectors_loc'] = backup_data[7]
        info_dict['parts_loc'] = backup_data[8]
        info_dict['tools_loc'] = backup_data[9]
        info_dict['is_merged'] = backup_data[10]
        info_dict['is_tool'] = backup_data[11]
        info_dict['connectors_serial_OCR'] = backup_data[12]
        info_dict['connectors_mult_OCR'] = backup_data[13]
        info_dict['parts'] = backup_data[14]
        info_dict['parts_info'] = backup_data[15]
        info_dict['cad_models'] = backup_data[16]
        info_dict['candidate_classes'] = backup_data[17]
        info_dict['actions'] = backup_data[18]
        info_dict['step_action'] = backup_data[19]
        info_dict['unused_parts'] = backup_data[20]
        info_dict['used_parts'] = backup_data[21]
        return info_dict

def save_cad_center(initial=True):
    mute = True
    flag = ' -cad_path ' + opt.cad_path + ' -json_path ' + opt.cad_path + ' -initial ' + str(initial)
    if mute:
        os.system(opt.blender + ' -b -P ./function/utilities/save_cad_center.py > ./function/utilities/stdout.txt -- ' + flag + ' 2>&1')
    else:
        os.system(opt.blender + ' -b -P ./function/utilities/save_cad_center.py -- ' + flag)

if __name__ == '__main__':
    main()
