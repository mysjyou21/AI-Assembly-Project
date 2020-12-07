from Robot import Assembly
import glob
import os
import cv2
import time
import platform
import pickle
from socket import *
from sys import exit
from data_file import SocketInfo, bcolors
from function.Grouping_mid.hole_loader import mid_loader

from config import *
opt = init_args()

class SocketInfo(SocketInfo):
    HOST='xx.xx.xx.xx'

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

                csock.send(("msg_success").encode())
                print(bcolors.CBLUE2+bcolors.CBOLD+"\n[SNU] Wait main program request"+bcolors.CEND)

            elif commend.decode('utf-8') == "extract_CAD_info":
                print(bcolors.CGREEN2+bcolors.CBOLD+"[main program] extract CAD info"+bcolors.CEND)
                # check whether part stl or part obj files in cad folder, json files
                cad_files = glob.glob(os.path.join(opt.cad_path, '*.obj'))
                cad_files += glob.glob(os.path.join(opt.cad_path, '*.STL'))
                cad_files = [os.path.basename(x).replace('.obj','').replace('.stl','') for x in cad_files]
                check_cad = [x for x in range(1,7) if 'part%d'%(x) in cad_files]

                cadinfo_files = glob.glob(os.path.join(opt.hole_path, '*.json'))
                cadinfo_files = [os.path.basename(x).replace('.json','') for x in cadinfo_files]
                check_cadinfo = [x for x in range(1,7) if 'part%d'%(x) in cadinfo_files]

                if len(check_cad)==6 and len(check_cadinfo)==6:
                    print(bcolors.CBLUE2+"Check all 6 base parts cad and cad info files are in right location"+bcolors.CEND)
                    csock.send(("msg_success").encode())
                else:
                    csock.send(("msg_failed").encode()) ### ??

                print(bcolors.CBLUE2+bcolors.CBOLD+"\n[SNU] Wait main program request"+bcolors.CEND)

            elif commend.decode('utf-8') == "request_step_num":
                print(bcolors.CGREEN2+bcolors.CBOLD+"[main program] require step num"+bcolors.CEND)
                message = "msg_success#"+str(IKEA.num_steps)
                csock.send((message).encode())
                print(bcolors.CBLUE2+bcolors.CBOLD+"\n[SNU] Wait main program request"+bcolors.CEND)

            elif "check_cad_file" in commend.decode('utf-8'):
                start_step_num_temp = int(commend.decode('utf-8').split('#')[-3])
                restoration = int(commend.decode('utf-8').split('#')[-2])
                add_cad = int(commend.decode('utf-8').split('#')[-1])
                step = start_step_num_temp
                if is_start:
                    print_message = 'start step %d' % start_step_num_temp
                else:
                    print_message = 'step %d' % start_step_num_temp
                print(bcolors.CGREEN2+bcolors.CBOLD+"[main program] SNU status checks-CAD_file,"+print_message+bcolors.CEND)
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

                print(bcolors.CBLUE2+'\n\n(step {}) CAD Rendering\n'.format(step)+bcolors.CEND)
                # Rendering
                SIGNAL = True
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

                if step > 1:
                    message = cadinfo_check(step, IKEA)
                    csock.send(message.encode())
                else:
                    csock.send(("msg_success").encode())

                print(bcolors.CBLUE2+bcolors.CBOLD+"\n[SNU] Wait main program request"+bcolors.CEND)
                is_start = 0

            elif commend.decode('utf-8') == "request_recognize_info":
                print(bcolors.CGREEN2+bcolors.CBOLD+"[main program] Request recognize info"+bcolors.CEND)
                time.sleep(1)
                csock.send(("msg_success").encode()) # receive the request

                print(bcolors.CBLUE2+bcolors.CBOLD+"[SNU] Process recognize"+bcolors.CEND)

                step_start_time = time.time()
                print(bcolors.CBLUE2+"Start Recognizing"+bcolors.CEND)
                IKEA.detect_step_component(step)

                IKEA.predict_pose(step)
                if step > 2 and opt.mid_RT_on:
                    IKEA.group_RT_mid(step)
                    if opt.hole_detection_on:
                        try:
                            IKEA.msn2_hole_detector(step)
                        except IndexError:
                            pass

                IKEA.group_as_action(step)
                print(IKEA.actions[step])
                IKEA.write_csv_mission(step, option=0)

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

def cadinfo_check(step, IKEA):
    cadinfo_files = glob.glob(os.path.join(opt.hole_path, '*.json'))
    cadinfo_files = [x for x in cadinfo_files if ('part' not in os.path.basename(x)) and ('C' not in os.path.basename(x))]
    cadinfo_files = [os.path.basename(x).replace('.json','') for x in cadinfo_files]
    temp_check=0
    message = ''
    for cadinfo_name in cadinfo_files:
        if int(cadinfo_name.split('_')[0].replace('step','')) == step-1: #just previous step
            mid_hole_XYZ = mid_loader(cadinfo_name)
            temp_check=1
            mid_id_list = [int(x.replace('part','')) for x in mid_hole_XYZ.keys()]
            check_cadinfo_mid = [x for x in mid_id_list if x in IKEA.unused_parts[step]]
            if len(check_cadinfo_mid)==0:
                message = "msg_success"
                print(bcolors.CBLUE2+"Correct cad info files"+bcolors.CEND)
            else:
                message = "msg_success" #failed
                print(bcolors.CYELLOW2+"Wrong cad info files"+bcolors.CEND)
                for_print = ['in json, '] + [str(x) for x in mid_id_list] + ['detected, '] + \
                            [str(x) for x in range(1,9) if x not in IKEA.unused_parts[step]]
                for_print = [bcolors.CYELLOW2+x+bcolors.CEND for x in for_print]
                print(*for_print)
    if temp_check==0:
        message = "msg_success" #failed
        print(bcolors.CYELLOW2+"failed to find cadinfo in step %d"%(step)+bcolors.CEND)

    return message

if __name__ == '__main__':
    main()
