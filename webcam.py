import cv2, imutils
from ultralytics import YOLO
import socket
import sys
import numpy as np
import pickle
import time
import base64
import struct

BUFF_SIZE = 65536
server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
host_name = socket.gethostname()
host_ip = "100.70.110.141" #socket.gethostbyname(host_name)
port = 9999
socket_address = (host_ip, port)
server_socket.bind(socket_address)
print("Listening at:", socket_address)
model = YOLO('yolo11n.pt')
webcamera = cv2.VideoCapture(0)
fps,st,frames_to_count,cnt = (0,0,20,0)

while True:
	msg,client_addr = server_socket.recvfrom(BUFF_SIZE)
	print('GOT connection from ',client_addr)
	WIDTH=480
	success, frame = webcamera.read()
	frame = cv2.flip(frame, 1)
	while(webcamera.isOpened()):
		_, frame = webcamera.read()
		frame = imutils.resize(frame, width=WIDTH)
		encoded, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
		message = base64.b64encode(buffer)
		server_socket.sendto(message, client_addr)
		frame = cv2.putText(frame, 'FPS: ' + str(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		key = cv2.waitKey(1) & 0xFF
		results = model.predict(frame)
		cv2.putText(frame, f"Total: {len(results[0].boxes)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
		cv2.imshow("Live Camera", results[0].plot())
		if key == ord('q'):
			server_socket.close()
			break
		if cnt == frames_to_count:
			try:
				fps = round(frames_to_count/(time.time()-st))
				st = time.time()
				cnt = 0
			except:
				pass
		cnt += 1
   
    
    
   
   # results = model.predict("/home/bigboss/Pictures/monitor.png", save=True, imgsz=320, conf=0.5)

webcamera.release()
cv2.destroyAllWindows()
