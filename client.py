#run this code seccond to send the image to the listening server
import socket #get the socket function libraries

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #AF_INET = IP
client.connect(('localhost', 1002))

file = open('tzeentch.jpg', "rb") #read the file

image_data = file.read(2048)

while image_data: 
    client.send(image_data)
    image_data = file.read(2048)


file.close()
client.close()
