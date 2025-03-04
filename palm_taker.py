import cv2
import mediapipe as mp
import numpy as np
import os

output_folder = "palm_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

image_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

            padding = 50
            x_min = max(x_min - padding, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + padding, frame.shape[1])
            y_max = min(y_max + padding, frame.shape[0])

            palm_image = frame[y_min:y_max, x_min:x_max]

            if palm_image.size > 0:
                image_counter += 1
                palm_image_filename = os.path.join(output_folder, f"palm_{image_counter}.jpg")
                cv2.imwrite(palm_image_filename, palm_image)
                print(f"Saved {palm_image_filename}")

    cv2.imshow("Palm Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
