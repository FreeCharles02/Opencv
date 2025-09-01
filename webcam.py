import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
webcamera = cv2.VideoCapture(0)


while True:
    success, frame = webcamera.read()
    frame = cv2.flip(frame, 1)
    
    results = model.predict(frame)
   # results = model.predict("/home/bigboss/Pictures/monitor.png", save=True, imgsz=320, conf=0.5)
    
    cv2.putText(frame, f"Total: {len(results[0].boxes)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Live Camera", results[0].plot())
   

    if cv2.waitKey(1) == ord('q'):
         break

webcamera.release()
cv2.destroyAllWindows()
