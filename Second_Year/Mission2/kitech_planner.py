#!/usr/bin/env python3

import shutil
#import rospy
#import rospkg
import sys
import os
from os.path import exists
import argparse
import time

from socket import *

import json
import glob
import pickle

#from utils import json_parser
#from utils import PDDL_Creator
#from utils import query
#from utils import execute_plan
#from utils.socket_buffer import getFileData, getFileSize
from data_file import SocketInfo, bcolors

#from std_msgs.msg import String
#from std_srvs.srv import Empty, EmptyResponse
#from rosplan_knowledge_msgs.srv import *
#from rosplan_dispatch_msgs.srv import DispatchService, DispatchServiceResponse, PlanningService, PlanningServiceResponse
#from diagnostic_msgs.msg import KeyValue

def generate_problem_and_plan():
    rospy.loginfo("kitech_planner: (%s) Calling planner" % rospy.get_name())
    pi = rospy.ServiceProxy('/rosplan_planner_interface/planning_server_params', PlanningService)
    pi_response = pi(domain_path, problem_path, data_path, planner_command, False)

    if not pi_response:
        rospy.logerr("kitech_planner: (%s) No response from the planning server." % rospy.get_name())
        return False
    if not pi_response.plan_found:
        rospy.loginfo("kitech_planner: (%s) No plan could be found." % rospy.get_name())
        return False
    else:
        rospy.loginfo("kitech_planner: (%s) Plan was found." % rospy.get_name())
        return True

# 프로그램 initialization
def program_init(csock_S):
    print(bcolors.CBLUE2+bcolors.CBOLD+"1. Start Program Init"+bcolors.CEND)
    send_msg = "Program_init"
    csock_S.send(send_msg.encode())
    while True:
        commend=csock_S.recv(SocketInfo.BUFSIZE)

        if commend.decode('utf-8')=="msg_success":
            print((bcolors.CBLUE2+bcolors.CBOLD+"2. {}: Connection from {} has been established."+bcolors.CEND).format((bcolors.CGREEN2+bcolors.CBOLD+"[AI-Assembly Project]"+bcolors.CBLUE2+bcolors.CBOLD), SocketInfo.ADDR1))
            break;

    send_msg = "request_step_num"
    csock_S.send(send_msg.encode())

    while True:
        commend=csock_S.recv(SocketInfo.BUFSIZE)
        if "msg_success" in commend.decode('utf-8'):
            step_num = int(commend.decode('utf-8').rstrip("msg_success").split('#')[-1])
            print((bcolors.CBLUE2+bcolors.CBOLD+"3. Entire step number is %d." % step_num)+bcolors.CEND)
            break;

    print(bcolors.CBLUE2+bcolors.CBOLD+"4. Init end"+bcolors.CEND)
    return step_num # 스탭 수 반환

# Kitech Planner와 서울대 인식기 실행
def program_run(csock_S, loop, step=1, add_cad=0):
    test_ans_path="./temp/"
    if os.path.exists(test_ans_path):
        shutil.rmtree(test_ans_path)
    if not os.path.exists(test_ans_path):
        os.mkdir(test_ans_path)
    for i in range(step-1, loop):
#        while True:
#            key = input("Press Enter if the next step process is ready")
#            if key == "":
#                break
        print(bcolors.CBLUE2+"\n\n(step {})\n".format(i+1)+bcolors.CEND)
        print(bcolors.CBLUE2+bcolors.CBOLD+"1. Program run"+bcolors.CEND)
#        rospy.sleep(1)

#        print(bcolors.CBLUE2+bcolors.CBOLD+"2. SNU status checks"+bcolors.CEND)
#        if i == step-1:
#            send_msg = "check_cad_file#{}#1#{}".format(i+1,add_cad)
#        else:
#            send_msg = "check_cad_file#{}#0#{}".format(i+1,add_cad)
#
#        csock_S.send(send_msg.encode())
#
#        while True:
#            commend=csock_S.recv(SocketInfo.BUFSIZE)
#
#            if commend.decode('utf-8')=="msg_success": # msg_fail : CASE1. 공유 폴더 연결 끊어짐 CASE2. VREP에서 중간산출물 CAD 생성 실패
#                print(bcolors.CBLUE2+bcolors.CBOLD+"3. Request recognize info"+bcolors.CEND)
#                break

        if i==step-1:
            send_msg = "request_recognize_info#%d#1#%d"%(i+1, add_cad) # 서울대에 인식 정보 요청
        else:
            send_msg = "request_recognize_info#%d#0#%d"%(i+1, add_cad)
        csock_S.send(send_msg.encode())

        while True:
            commend=csock_S.recv(SocketInfo.BUFSIZE)

            if commend.decode('utf-8')=="msg_success":
                print(bcolors.CBLUE2+bcolors.CBOLD+"4. msg sending success"+bcolors.CEND)
                break

        # KITECH Planner가 서울대 인식 정보 수신
        while True:
            commend=csock_S.recv(SocketInfo.BUFSIZE)

            if commend.decode('utf-8')=="send_recognize_info":
                print(bcolors.CBLUE2+bcolors.CBOLD+"5. Receiving recogniize info"+bcolors.CEND)
#                rospy.sleep(1)
                break

        msg = "msg_success"
        csock_S.send(msg.encode())

        msg = csock_S.recv(SocketInfo.BUFSIZE)
        flist = pickle.loads(msg)
        filename = str(flist[i]) # KITECH Planner가 수신할 인식 정보 파일 이름 서울대로부터 수신
        print("Request info of :", flist[i])

        csock_S.send(filename.encode()) # 해당 이름을 가진 인식 정보 파일 요청

        # reSize = 수신할 파일 사이즈
        reSize = csock_S.recv(SocketInfo.BUFSIZE)
        reSize = reSize.decode()

        # 서울대 디렉토리에서 파일을 못찾아 error를 보냈을 경우
        if reSize == "error":
            print("There is no file.")
            return csock_S.send(filename.encode())

        # KITECH Planner에서 파일 크기를 받았고, 파일 내용을 받을 준비가 되었다는 것을 서울대에 알림
        msg = "ready"
        csock_S.send(msg.encode())
        time.sleep(0.5)

        # 수신할 파일 사이즈만큼 recv
        with open(os.path.join(test_ans_path + filename), 'w', encoding = 'UTF-8') as jsonf:
            data = csock_S.recv(int(reSize))
            jsonf.write(data.decode())
        print("Receive :", filename, "Size : " + reSize) # 받은 파일, 사이즈

        print("step - {}".format(i))
#        snu_mission, snu_info = json_parser.json_parser(ans_path + "mission_{}.json".format(i+1)) # 서울대 인식 결과 호출 함수
#        PDDL = PDDL_Creator.PDDL_Creator(snu_mission, AAO, plan_path) # PDDL 생성 함수
#        plan_found = generate_problem_and_plan() # 작업계획 생성 함수
#
#        if plan_found:
#            execute_plan.execute_plan(plan_path, i+1) # 중간산출물과 중간산출물 데이터 생성 함수

        print(bcolors.CBLUE2+bcolors.CBOLD+"6. Augment info"+bcolors.CEND)
#        rospy.sleep(1)
        print(bcolors.CBLUE2+bcolors.CBOLD+"7. Create PDDL"+bcolors.CEND)
#        rospy.sleep(1)
        print(bcolors.CBLUE2+bcolors.CBOLD+"8. Make Plan"+bcolors.CEND)
#        rospy.sleep(2)

if __name__ == "__main__":
    # get path of pkg
#    rospy.init_node("kitech_planner")
#    owl_file_name = rospy.get_param("owl_file_name")
#    ans_path = rospy.get_param("ans_file_path")
#    test_ans_path = rospy.get_param("test_ans_file_path")
#    plan_path = rospy.get_param("plan_path")
#
#    # load parameters
#    planner_command = rospy.get_param('~planner_command', "")
#    domain_path = rospy.get_param('~domain_path', "")
#    problem_path = rospy.get_param('~problem_path', "")
#    data_path = rospy.get_param('~data_path', "")
#    initial_state = rospy.get_param('~initial_state', "")

    print("-------------------------------------------")

    # wait for services
#    rospy.wait_for_service('/rosplan_planner_interface/planning_server_params')

    print("-------------------------------------------")

#    AAO = query.AAODatabaseIndividualQuery(owl_file_name) # AAO DB 관계정보 쿼리 함수

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_step', default=1, type=int)
    parser.add_argument('--add_cad', default=0, type=int)
    opt, unknown = parser.parse_known_args()

    print("-------------------------------------------")
    print("Program Start - Main Program\n")

    ssock1=socket(AF_INET, SOCK_STREAM)
    ssock1.bind(SocketInfo.ADDR1)
    ssock1.listen(5)
    # KITECH Planner 소켓이 서울대 클라이언트의 접속 대기
    csock_S=None

    while True:
        if csock_S is None:
            # 서울대 클라이언트 접속 대기
            print("waiting for SNU connection...")
            csock_S, addr_info = ssock1.accept()
        else:
            break;

    print("\n\n-------------------------------------------")
#    start = rospy.get_rostime().to_sec() # 시간 측정 시작
    loop = program_init(csock_S) # 초기화 실행 함수

    print("\n\n-------------------------------------------")
    program_run(csock_S, loop, step=opt.start_step, add_cad=opt.add_cad) # 작업계획 프로그램 실행 함수
#    end = rospy.get_rostime().to_sec() # 시간 측정 종료

    print()
#    print("Total elasped time: ", format(end - start, ".2f"), "[s]")

    send_msg = "program_end"
    csock_S.send(send_msg.encode())
    csock_S.close()
    print("Program end")
