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
    print('assembly_name', opt.assembly_name)
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

        if step > 1:
            save_cad_center(initial=False, cad_adrs=list_added_obj + list_added_stl)

        if WAIT_SIGNAL:
            IKEA.predict_pose(step)
            list_prev_obj = sorted(glob.glob(os.path.join(IKEA.opt.cad_path, '*.obj')))
            list_prev_obj = [os.path.basename(x) for x in list_prev_obj]
        else:
            IKEA.predict_pose(step)

        if not opt.no_hole:
            IKEA.fastener_detector(step)

            if step > 1:
                IKEA.make_parts_info_indexed(step)

            if step > 2 and opt.mid_RT_on:
                IKEA.group_RT_mid(step)
                if opt.hole_detection_on:
                    IKEA.msn2_hole_detector(step)
            else:
                if opt.hole_detection_on:
                    IKEA.msn2_hole_detector(step)
            IKEA.group_as_action(step)

            print(IKEA.step_action)

        step_end_time = time.time()
        step_min, step_sec = get_elapsed_time(step_end_time, step_start_time)
        total_min, total_sec = get_elapsed_time(step_end_time, total_start_time)
        if opt.print_time:
            print('step time : {} min {} sec'.format(step_min, step_sec))
            print('total time : {} min {} sec'.format(total_min, total_sec))
        if WAIT_SIGNAL and AUTO and step<IKEA.num_steps:
            for p in new_cad_list[step-1]:
                os.system('mv '+os.path.join(opt.assembly_path, p)+' '+os.path.join(opt.cad_path, p))

def save_cad_center(initial=True, cad_adrs=['']):
    mute = True
    if cad_adrs != ['']:
        cad_adrs_flag = (',').join(cad_adrs)
        flag = ' -cad_path ' + opt.cad_path + ' -json_path ' + opt.cad_path + ' -initial ' + str(initial) + ' -cad_adrs ' + cad_adrs_flag
    else:
        flag = ' -cad_path ' + opt.cad_path + ' -json_path ' + opt.cad_path + ' -initial ' + str(initial)
    if mute:
        os.system(opt.blender + ' -b -P ./function/utilities/save_cad_center.py > ./function/utilities/stdout.txt -- ' + flag + ' 2>&1')
    else:
        os.system(opt.blender + ' -b -P ./function/utilities/save_cad_center.py -- ' + flag)

if __name__ == '__main__':
    main()
