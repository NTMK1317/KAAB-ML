# client.py

import socket
import sys
import os

def send_string(sock,string):
    sock.sendall(string.encode() + b'\n')

def send_int(sock,integer):
    sock.sendall(str(integer).encode() + b'\n')

def transmit(sock,folder):
    print(f'Sending folder: {folder}')
    send_string(sock,folder)
    files = os.listdir(folder)
    send_int(sock,len(files))
    for file in files:
        path = os.path.join(folder,file)
        filesize = os.path.getsize(path)
        print(f'Sending file: {file} ({filesize} bytes)')
        send_string(sock,file)
        send_int(sock,filesize)
        with open(path,'rb') as f:
            sock.sendall(f.read())

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost',8000))
with s:
    transmit(s,sys.argv[1])