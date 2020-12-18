from Robot import Assembly
import glob, os
import fnmatch
from os.path import exists
import shutil
import cv2
import time
import platform
import pickle
import json
from socket import *
from sys import exit
from data_file import SocketInfo, bcolors
from function.Grouping_mid.hole_loader import mid_loader

from config import *
opt = init_args()

# 파일 사이즈
def getFileSize(path, filename):
        fileSize = os.path.getsize(os.path.join(path, filename))
        return str(fileSize)

# 송신할 파일
def getFileData(path, filename):
        with open(os.path.join(path, filename), 'r', encoding="UTF-8") as f:
            data = ""
            for line in f:
                data += line
        return data

csock = socket(AF_INET, SOCK_STREAM)
csock.connect(SocketInfo.ADDR1)

# auto-run (Eunji)
AUTO=False
new_cad_list = [['step1_a.STL', 'step1_b.STL'], ['step2.STL'], ['step3.STL'], ['step4.STL'], ['step5.STL'], ['step6.STL'], ['step7.STL'], ['step8.STL']]

def main():
    while True:
        try:
            commend = csock.recv(SocketInfo.BUFSIZE)
            if commend.decode('utf-8') == "Program_init":
                print(bcolors.CGREY+bcolors.CBOLD+"[main program] Program init"+bcolors.CEND)
                # Initialize
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

                print(bcolors.CBLUE2+'Loading Weights'+bcolors.CEND)
                IKEA = Assembly(opt)
                print('gpu : ', opt.gpu)
                print('num_steps : ', IKEA.num_steps)

                list_prev_obj = []
                list_prev_stl = []

                step = 1
                is_start = 1

                list_added_obj = []
                list_added_stl = []

                def get_elapsed_time(toc, tic):
                    sec = int((toc - tic) % 60)
                    minute = int((toc - tic) // 60)
                    return minute, sec

                save_cad_center(initial=True)

                csock.send(("msg_success").encode())
                print(bcolors.CBLUE2+bcolors.CBOLD+"\n[SNU] Wait main program request"+bcolors.CEND)

            elif commend.decode('utf-8') == "request_step_num":
                print(bcolors.CGREEN2+bcolors.CBOLD+"[main program] require step num"+bcolors.CEND)
                message = "msg_success#"+str(IKEA.num_steps)
                csock.send((message).encode())
                print(bcolors.CBLUE2+bcolors.CBOLD+"\n[SNU] Wait main program request"+bcolors.CEND)

            elif commend.decode('utf-8')=="request_recognize_info":
                # kitech planner가 서울대에게 인식 정보를 요청하면서 서울대의 ./input/stefan/ 폴더에 중간산출물을 ./input/stefan/cad_info와 ./input/stefan/cad_info2에 이전 스탭에서 생성한 중간산출물 정보를 넣어줌.
                print(bcolors.CGREEN2+bcolors.CBOLD+"[main program] Request recognize info"+bcolors.CEND)
                commend_decode = commend.decode('utf-8')
                print(bcolors.CBLUE2+"\n(Step %d)\n"%(step)+bcolors.CEND)

                if step == 1: # 스탭 1은 중간산출물이 없기 때문에 파일을 읽어가지 않음
                    SIGNAL = True
                else:
#                    for p in new_cad_list[step-2]:
#                        os.system('mv '+os.path.join(opt.assembly_path, p)+' '+os.path.join(opt.cad_path, p))
                    SIGNAL = False
                    while not SIGNAL:
                        print('Waiting Signal ...', end='\r')
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

                    print(list_added_stl, list_added_obj)
                    save_cad_center(initial=False, cad_adrs=list_added_obj + list_added_stl)

                csock.send(("msg_success").encode()) # receive the request

                print(bcolors.CBLUE2+bcolors.CBOLD+"[SNU] Process recognize"+bcolors.CEND)

                step_start_time = time.time()
                print(bcolors.CBLUE2+"Start Recognizing"+bcolors.CEND)
                IKEA.detect_step_component(step)

                IKEA.predict_pose(step)
                
                IKEA.fastener_detector(step)

                if step > 2 and opt.mid_RT_on:
                    IKEA.group_RT_mid(step)
                    if opt.hole_detection_on:
                        IKEA.msn2_hole_detector(step)
                else:
                    if opt.hole_detection_on:
                        IKEA.msn2_hole_detector(step)

                IKEA.group_as_action(step)
                print(IKEA.actions[step])

                print(bcolors.CBLUE2+bcolors.CBOLD+"[SNU] Send recognize info"+bcolors.CEND)
                csock.send(("send_recognize_info").encode())

                step_end_time = time.time()
                step_min, step_sec = get_elapsed_time(step_end_time, step_start_time)
                total_min, total_sec = get_elapsed_time(step_end_time, total_start_time)
                if opt.print_time:
                    print(bcolors.CBLUE2+'step time : {} min {} sec'.format(step_min, step_sec)+bcolors.CEND)
                    print(bcolors.CBLUE2+'total time : {} min {} sec'.format(total_min, total_sec)+bcolors.CEND)
                if AUTO and step<IKEA.num_steps:
                    for p in new_cad_list[step-1]:
                        os.system('mv '+os.path.join(opt.assembly_path, p)+' '+os.path.join(opt.cad_path, p))
                step += 1
                if step > IKEA.num_steps:
                    print(bcolors.CBLUE2+'Last Step'+bcolors.CEND)

                # 서울대 인식 정보를 Kitech planner로 송신하는 부분
                print(bcolors.CGREEN2+bcolors.CBOLD+"[main program] Request SNU recognition results"+bcolors.CEND)
                while True:
                    commend=csock.recv(SocketInfo.BUFSIZE)
                    if commend.decode('utf-8')=="msg_success":
                        break;

                flist = sorted(os.listdir(IKEA.opt.output_dir)) # 서울대의 json(서울대 인식 정보) 파일 리스트를 생성함
                msg = pickle.dumps(flist)
                csock.send(msg) # 서울대가 json(서울대 인식 정보) 파일 리스트를 kitech planner로 송신

                filename = csock.recv(SocketInfo.BUFSIZE) # Kitech planner가 json 파일 리스트를 참고하여 스탭에 해당하는 json(서울대 인식 정보) 요청
                filename = filename.decode()

                # 서울대의 아웃풋 경로에 파일이 없으면 Kitech planner로 에러 메세지를 보냄
                if not exists(os.path.join(IKEA.opt.output_dir, filename)):
                    msg = "error"
                    csock.sendall(msg.encode())
                    print(bcolors.CYELLOW2+"File Not Found%s"%(filename)+bcolors.CEND) # ...???
                    return

                csock.sendall(getFileSize(IKEA.opt.output_dir, filename).encode()) # 서울대가 Kitech planner에게 요청받은 json의 파일 크기를 알려줌

                reReady = csock.recv(SocketInfo.BUFSIZE)
                if reReady.decode('utf-8') == "ready": # 파일 크기를 확인한 Kitech planner가 파일 내용을 받을 준비가 됐다고 서울대에 알림
                    csock.sendall(getFileData(IKEA.opt.output_dir, filename).encode()) # 서울대가 Kitech planner에게 json 파일을 보냄(파일 크기만큼)
                    print(bcolors.CBLUE2+bcolors.CBOLD+"[SNU] Send output file %s "%(filename)+"of Size"+getFileSize(IKEA.opt.output_dir, filename)+bcolors.CEND)

                print(bcolors.CBLUE2+bcolors.CBOLD+"\n[SNU] Wait main program request"+bcolors.CEND)

            elif commend.decode('utf-8') == "program_end":
                print(bcolors.CBLUE2+bcolors.CBOLD+"[SNU] Program Finish"+bcolors.CEND)
                csock.close()
                exit()
        except Exception as e:
            print(bcolors.CRED2+bcolors.CBOLD+"%s:%s" % (e, SocketInfo.ADDR1)+bcolors.CEND)
            csock.close()
            exit()

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
