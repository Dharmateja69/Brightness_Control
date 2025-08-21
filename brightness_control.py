import cv2
import mediapipe as mp
import numpy as np
from math import hypot
import screen_brightness_control as sbc   # ✅ added

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    max_num_hands=2
)
Draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Process = hands.process(frameRGB)

    landmarkList = []

    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                h, w, _ = frame.shape
                x, y = int(landmarks.x * w), int(landmarks.y * h)
                landmarkList.append([_id, x, y])

            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    if len(landmarkList) != 0:
        x_1, y_1 = landmarkList[4][1], landmarkList[4][2]
        x_2, y_2 = landmarkList[8][1], landmarkList[8][2]

        cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED)
        cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3)

        L = hypot(x_2 - x_1, y_2 - y_1)
        brightness_value = int(np.interp(L, [15, 220], [0, 100]))

        # ✅ Set system brightness
        try:
            sbc.set_brightness(brightness_value)
            print(f"Setting Brightness: {brightness_value}%")
        except Exception as e:
            print("Error setting brightness:", e)

        cv2.putText(frame, f"Brightness: {brightness_value}%", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
