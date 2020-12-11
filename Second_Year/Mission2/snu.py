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

            elif "request_recognize_info" in commend.decode('utf-8'):
                # kitech planner가 서울대에게 인식 정보를 요청하면서 서울대의 ./input/stefan/ 폴더에 중간산출물을 ./input/stefan/cad_info와 ./input/stefan/cad_info2에 이전 스탭에서 생성한 중간산출물 정보를 넣어줌.
                print(bcolors.CGREEN2+bcolors.CBOLD+"[main program] Request recognize info"+bcolors.CEND)
                commend_decode = commend.decode('utf-8')
                if len(commend_decode.split('#'))==0:
                    start_step_num_temp = 2
                    restoration = 0
                    add_cad = 0
                    step = start_step_num_temp
                else:
                    start_step_num_temp = int(commend_decode.split('#')[-3])
                    restoration = int(commend_decode.split('#')[-2])
                    add_cad = int(commend_decode.split('#')[-1])
                    step = start_step_num_temp
                    if is_start:
                        print_message = 'start step %d' % start_step_num_temp
                    else:
                        print_message = 'step %d' % start_step_num_temp
                ########## 이전 정보 불러오기
                if start_step_num_temp != 1 and restoration:
                    # Restore Information of prior step
                    pickle_read_filepath =  opt.cad_path + '/' + str(step-1) + '/info_dict_'+ str(step-1) + '.pickle'
                    with open(pickle_read_filepath,'rb') as f:
                        pickle_data = pickle.load(f)

                    # 하나의 함수로 정리
                    print("{} Step Information Restored".format(step-1))
                    IKEA.circles_loc, IKEA.circles_separated_loc, IKEA.rectangles_loc, IKEA.connectors_serial_imgs, \
                    IKEA.connectors_serial_loc, IKEA.connectors_mult_imgs, IKEA.connectors_mult_loc, IKEA.connectors_loc, \
                    IKEA.parts_loc, IKEA.tools_loc, IKEA.is_merged, IKEA.is_tool, IKEA.connectors_serial_OCR, IKEA.connectors_mult_OCR, \
                    IKEA.parts, IKEA.parts_info, IKEA.cad_models, IKEA.candidate_classes, IKEA.actions, IKEA.step_action, IKEA.unused_parts, IKEA.used_parts \
                        = pickle_data_loader(mode="download", pickle_data=pickle_data)

                    if add_cad:
                        print("Need to add Mid from the prior step")
                        list_prev_obj = sorted(glob.glob(os.path.join(IKEA.opt.cad_path, '*.obj')))
                        list_prev_obj = [os.path.basename(x) for x in list_prev_obj]
                        list_prev_stl = sorted(glob.glob(os.path.join(IKEA.opt.cad_path, '*.STL')))
                        list_prev_stl = [os.path.basename(x) for x in list_prev_stl]
                        add_cad = 0

                print(bcolors.CBLUE2+"\n(Step %d)\n"%(step)+bcolors.CEND)

                if step == 1: # 스탭 1은 중간산출물이 없기 때문에 파일을 읽어가지 않음
                    SIGNAL = True
                else:
                    for p in new_cad_list[step-2]:
                        os.system('mv '+os.path.join(opt.assembly_path, p)+' '+os.path.join(opt.cad_path, p))
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

                    save_cad_center(initial=False, cad_adrs=list_added_obj + list_added_stl, load_previous=True if (start_step_num_temp!=1 and is_start) else False)

                is_start = 0

                csock.send(("msg_success").encode()) # receive the request

                print(bcolors.CBLUE2+bcolors.CBOLD+"[SNU] Process recognize"+bcolors.CEND)

                step_start_time = time.time()
                print(bcolors.CBLUE2+"Start Recognizing"+bcolors.CEND)
                IKEA.detect_step_component(step)

                IKEA.predict_pose(step)
                if step > 2 and opt.mid_RT_on:
                    IKEA.group_RT_mid(step)
                    if opt.hole_detection_on:
                        IKEA.msn2_hole_detector(step)
                else:
                    if opt.hole_detection_on:
                        IKEA.msn2_hole_detector(step)

                IKEA.group_as_action(step)
                print(IKEA.actions[step])

                # dictionary Info Backup per step
                backup_data = [IKEA.circles_loc, IKEA.circles_separated_loc, IKEA.rectangles_loc, IKEA.connectors_serial_imgs, \
                    IKEA.connectors_serial_loc, IKEA.connectors_mult_imgs, IKEA.connectors_mult_loc, IKEA.connectors_loc, \
                    IKEA.parts_loc, IKEA.tools_loc, IKEA.is_merged, IKEA.is_tool, IKEA.connectors_serial_OCR, IKEA.connectors_mult_OCR, \
                    IKEA.parts, IKEA.parts_info, IKEA.cad_models, IKEA.candidate_classes, IKEA.actions, IKEA.step_action, IKEA.unused_parts, IKEA.used_parts]
                info_dict = pickle_data_loader(mode="upload", backup_data=backup_data)

                pickle_filepath = IKEA.part_dir+'/info_dict_'+str(step)+'.pickle'

                with open(pickle_filepath, 'wb') as f:
                    pickle.dump(info_dict, f)

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

def save_cad_center(initial=True, cad_adrs=[''], load_previous=False):
    mute = True
    if load_previous:
        shutil.copy2(os.path.join(opt.cad_path, 'center_backup.json'), os.path.join(opt.cad_path, 'center.json'))
    if cad_adrs != ['']:
        cad_adrs_flag = (',').join(cad_adrs)
        flag = ' -cad_path ' + opt.cad_path + ' -json_path ' + opt.cad_path + ' -initial ' + str(initial) + ' -cad_adrs ' + cad_adrs_flag
    else:
        flag = ' -cad_path ' + opt.cad_path + ' -json_path ' + opt.cad_path + ' -initial ' + str(initial)
    if mute:
        os.system(opt.blender + ' -b -P ./function/utilities/save_cad_center.py > ./function/utilities/stdout.txt -- ' + flag + ' 2>&1')
    else:
        os.system(opt.blender + ' -b -P ./function/utilities/save_cad_center.py -- ' + flag)

    if initial:
        if not os.path.exists(os.path.join(opt.cad_path, 'center.json')):
            print('Error: Initial step, center.json not created')
        with open(os.path.join(opt.cad_path, 'center.json'), 'r') as f:
            backup = json.load(f)
        with open(os.path.join(opt.cad_path, 'center_backup.json'), 'w') as f:
            json.dump(backup, f, indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
