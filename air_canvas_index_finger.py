 # Air-Canvas: Draw Using Index Finger

import cv2
import numpy as np
import mediapipe as mp

# Initialize webcam
vid = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Define drawing colors
colors = [(0, 0, 214), (0, 255, 127), (126, 18, 0)]  # RED, GREEN, BLUE
color = None  # Current selected color

# Create a blank canvas to draw on
canvas = np.zeros((471, 636, 4), dtype=np.uint8) + 150
canvas = cv2.resize(canvas, None, None, fx=1, fy=1)

# Check if webcam is opened
if not vid.isOpened():
    print("Could not open webcam")

# Main loop to capture frames
while vid.isOpened():
    ret, frame = vid.read()
    frame = cv2.resize(frame, None, None, fx=1, fy=1)
    frame = cv2.flip(frame, 1)  # Flip horizontally for mirror effect
    
    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Draw color and clear buttons
    cv2.rectangle(frame, (20, 1), (120, 65), (122, 122, 122), -1)  # Clear All
    cv2.rectangle(frame, (140, 1), (240, 65), colors[0], -1)       # Red
    cv2.rectangle(frame, (260, 1), (360, 65), colors[1], -1)       # Green
    cv2.rectangle(frame, (380, 1), (480, 65), colors[2], -1)       # Blue

    # Labels for buttons
    cv2.putText(frame, "CLEAR ALL", (30, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    cv2.putText(frame, "RED", (175, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    cv2.putText(frame, "GREEN", (285, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    cv2.putText(frame, "BLUE", (410, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # Process hand landmarks
    if results.multi_hand_landmarks:
        # Get index finger tip coordinates
        normalizedLandmark = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        h, w, c = frame.shape
        cx, cy = int(normalizedLandmark.x * w), int(normalizedLandmark.y * h)

        # Draw a small circle on index finger tip
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), 2)

        # If finger is in top area, check for button selection
        if cy < 65 and cx <= 490:
            if 20 <= cx <= 120:
                # Clear canvas
                canvas = np.zeros((471, 636, 4), dtype=np.uint8) + 150
                canvas = cv2.resize(canvas, None, None, fx=1.5, fy=1.5)
                color = None

            elif 140 <= cx <= 240:
                color = colors[0]  # RED
                previous_point = (cx, cy)

            elif 260 <= cx <= 360:
                color = colors[1]  # GREEN
                previous_point = (cx, cy)

            elif 380 <= cx <= 480:
                color = colors[2]  # BLUE
                previous_point = (cx, cy)

        # If finger is not on buttons, draw on canvas
        else:
            if color:
                cv2.line(canvas, previous_point, (cx, cy), color, 5)
                previous_point = (cx, cy)

    # Show both the canvas and live camera feed
    cv2.imshow('Canvas', canvas)
    cv2.imshow('Camera', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
vid.release()
cv2.destroyAllWindows()
