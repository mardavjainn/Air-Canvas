#Air-Canvas to draw using index finger

import cv2
import numpy as np
import mediapipe as mp
vid = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
colors = [(0, 0, 214), (0, 255, 127), (126, 18, 0)]
color = None
canvas = np.zeros((471, 636, 4), dtype=np.uint8) + 150
canvas = cv2.resize(canvas, None, None, fx=1, fy=1)

if not (vid.isOpened()):
    print("Could not open") 

while vid.isOpened():
    ret, frame = vid.read()
    frame = cv2.resize(frame, None, None, fx=1, fy=1)
    frame = cv2.flip(frame, 1)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    cv2.rectangle(frame, (20,1), (120,65), (122,122,122), -1)
    cv2.rectangle(frame, (140,1), (240,65), colors[0], -1)
    cv2.rectangle(frame, (260,1), (360,65), colors[1], -1)
    cv2.rectangle(frame, (380,1), (480,65), colors[2], -1)

    cv2.putText(frame, "CLEAR ALL", (30, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (175, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (285, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (410, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    if results.multi_hand_landmarks:
        normalizedLandmark = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        if normalizedLandmark:
            h, w, c = frame.shape
            cx, cy = int(normalizedLandmark.x * w), int(normalizedLandmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), 2)
            
            if cy < 65 and cx <= 490:
                if cx >= 20 and cx <= 120:
                    canvas = np.zeros((471, 636, 4), dtype=np.uint8) + 150
                    canvas = cv2.resize(canvas, None, None, fx=1.5, fy=1.5)
                    color = None
                
                if cx >= 140 and cx <= 240:
                    color = colors[0]
                    previous_point = (cx, cy)
                
                if cx >= 260 and cx <= 360:
                    color = colors[1]
                    previous_point = (cx, cy)
                
                if cx >= 380 and cx <= 480:
                    color = colors[2]
                    previous_point = (cx, cy)
                    
            else:
                if color:
                    cv2.line(canvas, previous_point, (cx, cy), color, 5)
                    previous_point = (cx, cy)   
    cv2.imshow('Canvas', canvas)
    # cv2.imshow('Mask', mask)
    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
