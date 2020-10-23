# !/usr/bin/env python3

#import rospy
#import rospkg
import sys
import os
import argparse
import time
from socket import *
from data_file import *

parser = argparse.ArgumentParser()
parser.add_argument('--start_step', default=1, type=int)
parser.add_argument('--add_cad', default=0, type=int)
opt = parser.parse_args()

# 프로그램 init (미션2에서 사용)
def program_init(csock_S, csock_W):
    print("1. Start Program Init")
    send_msg = "Program_init"
    csock_S.send(send_msg.encode())
    while True:
        commend=csock_S.recv(SocketInfo.BUFSIZE)

        if commend.decode('utf-8')=="msg_success":
            print("2. [SNU] msg sending success")
            break;

    csock_W.send(send_msg.encode())
    while True:
        commend=csock_W.recv(SocketInfo.BUFSIZE)

        if commend.decode('utf-8')=="msg_success":
            print("3. [Windows] msg sending success")
            break;

    time.sleep(1)
    print("4. Request Cad Info")

    send_msg = "request_cadinfo"
    csock_W.send(send_msg.encode())

    while True:
        commend=csock_W.recv(SocketInfo.BUFSIZE)

        if commend.decode('utf-8')=="msg_success":
            print("5. msg sending success")
            break;

    print("6. Wait Windows Data")

    while True:
        commend=csock_W.recv(SocketInfo.BUFSIZE)

        if commend.decode('utf-8')=="send_candinfo":
            print("7. Cad information receive")
            break;

    time.sleep(1)
    print("8. Cad info filtering")

    time.sleep(1)
    print("9. Cad extraction info send")

    send_msg = "extract_CAD_info"
    csock_S.send(send_msg.encode())

    while True:
        commend=csock_S.recv(SocketInfo.BUFSIZE)

        if commend.decode('utf-8')=="msg_success":
            print("10. msg sending success")
            break;

    send_msg = "request_step_num"
    csock_S.send(send_msg.encode())

    while True:
        commend=csock_S.recv(SocketInfo.BUFSIZE)
        if "msg_success" in commend.decode('utf-8'):
            step_num = int(commend.decode('utf-8').rstrip("msg_success").split('#')[-1])
            print("11. Entire step number is %d" % step_num)
            break;

    print("12. Init end")
    time.sleep(2)

    return step_num

# Assembly Planner와 서울대 인식기 실행
def program_run(csock_S, csock_W, loop, step=1, add_cad=0):
    for i in range(step-1, loop):
        print("-------------------------------------------")
        while True:
            key = input("Press Enter if the next step process is ready")
            if key == "":
                break
        print("1. Program run")
        time.sleep(1)

        print("2. SNU status checks")
        if i == step-1:
            send_msg = "check_cad_file#{}#1#{}".format(i+1,add_cad)
        else:
            send_msg = "check_cad_file#{}#0#{}".format(i+1,add_cad)

        csock_S.send(send_msg.encode())
        while True:
            commend=csock_S.recv(SocketInfo.BUFSIZE)

            if commend.decode('utf-8')=="msg_success":
                print("3. Request recognize info")
                break;

        send_msg = "request_recognize_info"
        csock_S.send(send_msg.encode())
        while True:
            commend=csock_S.recv(SocketInfo.BUFSIZE)

            if commend.decode('utf-8')=="msg_success":
                print("4. msg sending success")
                break;

        while True:
            commend=csock_S.recv(SocketInfo.BUFSIZE)

            if commend.decode('utf-8')=="send_recognize_info":
                print("5. Receiving recognize info")
                time.sleep(1)
                break;

        print("6. Augment info")
        time.sleep(1)
        print("7. Create PDDL")
        time.sleep(1)
        print("8. Make Plan")
        time.sleep(2)


if __name__ == "__main__":
    print("-------------------------------------------")
    print("Program Start - Main Program\n")
    # loop = 9
    ssock1=socket(AF_INET, SOCK_STREAM)
    ssock1.bind(SocketInfo.ADDR1)
    ssock1.listen(5)
    # 서버 소켓이 클라이언트의 접속 대기
    csock_S=None

    while True:
        if csock_S is None:
            # 서울대 클라이언트 접속 대기
            print("waiting for SNU connection...")
            csock_S, addr_info = ssock1.accept()
        else:
            break;

    ssock2=socket(AF_INET, SOCK_STREAM)
    ssock2.bind(SocketInfo.ADDR2)
    ssock2.listen(5)
    csock_W=None

    while True:
        if csock_W is None:
            # 윈도우 클라이언트 접속 대기
            print("waiting for Windows connection...")
            csock_W, addr_info = ssock2.accept()
        else:
            break;


    print("\n\n-------------------------------------------")
    input("Wait program init...\n\n")
    loop = program_init(csock_S, csock_W)

    print("\n\n-------------------------------------------")
    input("Wait program run...\n\n")
    program_run(csock_S, csock_W, loop, step=opt.start_step, add_cad=opt.add_cad)

    send_msg = "program_end"
    csock_S.send(send_msg.encode())
    csock_W.send(send_msg.encode())
    csock_S.close()
    csock_W.close()
