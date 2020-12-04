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

    for step in range(start_step, IKEA.num_steps + 1):
        print('\n\n(step {})\n'.format(step))
        step_start_time = time.time()
        IKEA.detect_step_component(step)

        if opt.step_num:

            # Restore Information of prior step
            pickle_read_filepath =  opt.cad_path + '/' + str(step-1) + '/info_dict_'+ str(step-1) + '.pickle'
            with open(pickle_read_filepath,'rb') as f:
                pickle_data = pickle.load(f)

            print("{} Step Information Restored".format(step-1))
            IKEA.circles_loc = pickle_data['circles_loc']
            IKEA.circles_separated_loc = pickle_data['circles_separated_loc']
            IKEA.rectangles_loc = pickle_data['rectangles_loc']
            IKEA.connectors_serial_imgs = pickle_data['connectors_serial_imgs']
            IKEA.connectors_serial_loc = pickle_data['connectors_serial_loc']
            IKEA.connectors_mult_imgs = pickle_data['connectors_mult_imgs']
            IKEA.connectors_mult_loc = pickle_data['connectors_mult_loc']
            IKEA.connectors_loc = pickle_data['connectors_loc']
            IKEA.parts_loc = pickle_data['parts_loc']
            IKEA.tools_loc = pickle_data['tools_loc']
            IKEA.is_merged = pickle_data['is_merged']
            IKEA.is_tool = pickle_data['is_tool']
            IKEA.connectors_serial_OCR = pickle_data['connectors_serial_OCR']
            IKEA.connectors_mult_OCR = pickle_data['connectors_mult_OCR']
            IKEA.parts = pickle_data['parts']
            IKEA.parts_info = pickle_data['parts_info']
            IKEA.cad_models = pickle_data['cad_models']
            IKEA.candidate_classes = pickle_data['candidate_classes']
            IKEA.actions = pickle_data['actions']
            IKEA.step_action = pickle_data['step_action']

            if opt.add_cad:
                print("Need to add Mid from the prior step")
                list_prev_obj = sorted(glob.glob(os.path.join(IKEA.opt.cad_path, '*.obj')))
                list_prev_obj = [os.path.basename(x) for x in list_prev_obj]
                list_prev_stl = sorted(glob.glob(os.path.join(IKEA.opt.cad_path, '*.STL')))
                list_prev_stl = [os.path.basename(x) for x in list_prev_stl]

        opt.step_num = False
        WAIT_SIGNAL = True  # 챌린지 환경에서는 True
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
            IKEA.rendering(step, list_added_obj, list_added_stl)
            IKEA.retrieve_part(step, list_added_obj, list_added_stl)
            list_prev_obj = sorted(glob.glob(os.path.join(IKEA.opt.cad_path, '*.obj')))
            list_prev_obj = [os.path.basename(x) for x in list_prev_obj]
        else:
            IKEA.rendering(step, list_added_obj, list_added_stl)
            IKEA.retrieve_part(step)
        IKEA.group_as_action(step)
        print(IKEA.actions[step])
        IKEA.write_mission(step)

        # dictionary Info Backup per step
        info_dict = {}
        info_dict['circles_loc'] = IKEA.circles_loc
        info_dict['circles_separated_loc'] = IKEA.circles_separated_loc
        info_dict['rectangles_loc'] = IKEA.rectangles_loc
        info_dict['connectors_serial_imgs'] = IKEA.connectors_serial_imgs #
        info_dict['connectors_serial_loc'] = IKEA.connectors_serial_loc
        info_dict['connectors_mult_imgs'] = IKEA.connectors_mult_imgs #
        info_dict['connectors_mult_loc'] = IKEA.connectors_mult_loc
        info_dict['connectors_loc'] = IKEA.connectors_loc
        info_dict['parts_loc'] = IKEA.parts_loc
        info_dict['tools_loc'] = IKEA.tools_loc
        info_dict['is_merged'] = IKEA.is_merged
        info_dict['is_tool'] = IKEA.is_tool
        info_dict['connectors_serial_OCR'] = IKEA.connectors_serial_OCR
        info_dict['connectors_mult_OCR'] = IKEA.connectors_mult_OCR
        info_dict['parts'] = IKEA.parts
        info_dict['parts_info'] = IKEA.parts_info
        info_dict['cad_models'] = IKEA.cad_models
        info_dict['candidate_classes'] = IKEA.candidate_classes
        info_dict['actions'] = IKEA.actions
        info_dict['step_action'] = IKEA.step_action

        pickle_filepath = IKEA.retrieved_cad_dir+'/info_dict_'+str(step)+'.pickle'
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

if __name__ == '__main__':
    main()
