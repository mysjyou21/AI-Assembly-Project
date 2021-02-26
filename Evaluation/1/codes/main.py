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
    print('--------------\n')


    def get_elapsed_time(toc, tic):
        sec = int((toc - tic) % 60)
        minute = int((toc - tic) // 60)
        return minute, sec
    
    print('detection start\n')
    print('--------------')
    for iter in range(opt.iters):
        for n, cut in enumerate(IKEA.cuts):
            # print('(cut {})\n'.format(n))
            cut_start_time = time.time()

            # detect_cut_component: 조립도 이미지 input으로
            IKEA.detect_cut_component(cut, n)

            cut_end_time = time.time()
            cut_min, cut_sec = get_elapsed_time(cut_end_time, cut_start_time)
            total_min, total_sec = get_elapsed_time(cut_end_time, total_start_time)
            if opt.print_time:
                print('cut time: {} min {} sec'.format(cut_min, cut_sec))
                print('total time: {} min {} sec'.format(total_min, total_sec))

        det_end_time = time.time()
        
        print('\ndetection ended\n')
        if opt.print_time:
            det_min, det_sec = get_elapsed_time(det_end_time, total_start_time)
            print('detection time: {} min {} sec'.format(det_min, det_sec))
        print('---------------')
        print('evaluation start')
        print('---------------')

        eval_start_time = time.time()

        # calculate editdistance and mAP
        IKEA.evaluate()

        eval_end_time = time.time()
        eval_min, eval_sec = get_elapsed_time(eval_end_time, eval_start_time)
        total_min, total_sec = get_elapsed_time(eval_end_time, total_start_time)
        if opt.print_time:
            print('evaluation time : {} min {} sec'.format(eval_min, eval_sec))
            print('total time : {} min {} sec'.format(total_min, total_sec))
    
    print('')


if __name__ == '__main__':
    main()
