
# This is client code to receive video frames over UDP
import cv2, imutils, socket
from ultralytics import YOLO
import numpy as np
import time
import base64

BUFF_SIZE = 95536
client_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
host_name = socket.gethostname()
host_ip = "100.70.110.191"
print(host_ip)
port = 9999
message = b'Hello'
model = YOLO('yolo11n.pt')

client_socket.sendto(message,(host_ip,port))
fps,st,frames_to_count,cnt = (0,0,50,0)
while True:
	packet,_ = client_socket.recvfrom(BUFF_SIZE)
	data = base64.b64decode(packet,' /')
	npdata = np.fromstring(data,dtype=np.uint8)
	frame = cv2.imdecode(npdata,1)
	frame = cv2.putText(frame,'FPS: '+str(fps),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
	results = model.predict(frame)
	cv2.imshow("RECEIVING VIDEO",results[0].plot())
	cv2.putText(frame, f"Total: {len(results[0].boxes)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		client_socket.close()
		break
	if cnt == frames_to_count:
		try:
			fps = round(frames_to_count/(time.time()-st))
			st=time.time()
			cnt=0
		except:
			pass
	cnt+=1

	for result in results:
		boxes = result.boxes 
		for box in boxes:
			c = box.cls
			if model.names[int(c)]:
				print("hello person!") 

