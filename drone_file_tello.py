import torch
import cv2
from djitellopy import Tello
import time

drone = Tello()
drone.connect()
print(f"Connected to drone: {drone.get_battery()}% battery")
drone.takeoff()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='your_model.pt')

TWO_FINGERS_LABEL = "two_fingers"
PALM_LABEL = "palm"

cap = cv2.VideoCapture(0)

def follow_user():
    drone.move_forward(30)
    drone.move_up(10)

def stop_drone():
    drone.send_rc_control(0, 0, 0, 0)
    drone.land()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    results = model(frame)

    gesture_detected = None
    for *xyxy, conf, cls in results.xywh[0]:
        label = model.names[int(cls)]

        if label == TWO_FINGERS_LABEL:
            gesture_detected = TWO_FINGERS_LABEL
            break
        elif label == PALM_LABEL:
            gesture_detected = PALM_LABEL
            break
    
    if gesture_detected == TWO_FINGERS_LABEL:
        follow_user()
    elif gesture_detected == PALM_LABEL:
        stop_drone()

    results.show()

    time.sleep(0.1)

cap.release()
cv2.destroyAllWindows()
