# client.py

import socket
import sys
import os

def send_string(sock,string):
    sock.sendall(string.encode() + b'\n')

def send_int(sock,integer):
    sock.sendall(str(integer).encode() + b'\n')

def transmit(sock,folder):
    print(folder)
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

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost',8000))

with s,s.makefile('rb') as serverfile:
    while True:
        folder = serverfile.readline()
        if not folder: # When client closes connection folder == b''
            break
        folder = folder.strip().decode()
        no_files = int(serverfile.readline())
        print(f'Receiving folder: {folder} ({no_files} files)')
        folderpath = 'ML-output'
        #os.path.join('Downloads',folder)  # put in different directory in case server/client on same system
        os.makedirs(folderpath,exist_ok=True)
        for i in range(no_files):
            filename = serverfile.readline().strip().decode()
            filesize = int(serverfile.readline())
            data = serverfile.read(filesize)
            print(f'Receiving file: {filename} ({filesize} bytes)')
            with open(os.path.join(folderpath,filename),'wb') as f:
                f.write(data)