# !/usr/bin/env python3

from socket import *
from sys import exit
from data_file import SocketInfo
import time

class SocketInfo(SocketInfo):
    HOST='xx.xx.xx.xx'

csock=socket(AF_INET, SOCK_STREAM)
csock.connect(SocketInfo.ADDR2)

print("Program Conneted - Windows")

while True:
    try:
        commend=csock.recv(SocketInfo.BUFSIZE)
        if commend.decode('utf-8') == "Program_init":
            print("[main program] Program init")
            time.sleep(1)
            csock.send(("msg_success").encode())
            print("\n[Window] Wait main program request")

        elif commend.decode('utf-8') == "request_cadinfo":
            print("[main program] request cadinfo")
            time.sleep(1)
            csock.send(("msg_success").encode())

            time.sleep(1)
            print("[Window] extract CAD info")

            time.sleep(1)
            print("[Window] Send CAD info")
            csock.send(("send_candinfo").encode())
            print("\n[Window] Wait main program request")

        elif commend.decode('utf-8') == "program_end":
            print("[Window] Program Finsh")
            csock.close()
            exit()
    except Exception as e :
        print("%s:%s" %(e, SocketInfo.ADDR))
        csock.close()
        exit()
