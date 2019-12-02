import socket
import base64

host = '192.168.0.177'
port = 5050
address = (host, port)

socket01 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

socket01.bind(address)
socket01.listen(1)
print('Socket Startup')

conn, addr = socket01.accept() 
print('Connected by', addr)

##################################################
print('begin write image file')
i = 0
while True:
    imgName = "socketImage%d.jpg" %i
    print(imgName)
    imgFile = open(imgName, 'wb')
    Break = False
    while True:
        imgData = conn.recv(1024)
        print(type(imgData))
        imgFile.write(imgData)
        if not imgData:
            print("enter here")
            Break = True
            break
        if imgData[-4:] == "over":
            break
        

    if Break:
        break

    imgFile.close()
    i += 1
    print('image save')
##################################################

conn.close()
socket01.close()
print('server close')