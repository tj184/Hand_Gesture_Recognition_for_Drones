import cv2
import torch

model_path = 'runs/train/exp/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    results.render()

    cv2.imshow('Palm Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
