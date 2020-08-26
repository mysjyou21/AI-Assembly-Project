from Robot import Assembly
import glob
import os
import cv2
import time
import platform

from config import *
opt = init_args()


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

    def get_elapsed_time(toc, tic):
        sec = int((toc - tic) % 60)
        minute = int((toc - tic) // 60)
        return minute, sec

    for step in range(1, IKEA.num_steps + 1):
        print('\n\n(step {})\n'.format(step))
        step_start_time = time.time()
        IKEA.detect_step_component(step)
        WAIT_SIGNAL = True  # 챌린지 환경에서는 True
        SIGNAL = False
        list_added_obj = []
        list_added_stl = []
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
            IKEA.retrieve_part(step, list_added_obj, list_added_stl)
            list_prev_obj = sorted(glob.glob(os.path.join(IKEA.opt.cad_path, '*.obj')))
            list_prev_obj = [os.path.basename(x) for x in list_prev_obj]
        else:
            IKEA.retrieve_part(step)
        IKEA.group_as_action(step)
        print(IKEA.actions[step])
        IKEA.write_csv_mission(step, option=0)

        step_end_time = time.time()
        step_min, step_sec = get_elapsed_time(step_end_time, step_start_time)
        total_min, total_sec = get_elapsed_time(step_end_time, total_start_time)
        if opt.print_time:
            print('step time : {} min {} sec'.format(step_min, step_sec))
            print('total time : {} min {} sec'.format(total_min, total_sec))

if __name__ == '__main__':
    main()
