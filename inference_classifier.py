#! C:\Users\Asus\Python\mvenv\Scripts\python.exe

import pickle
import cv2
import mediapipe as mp
import numpy as np
import serial # <-- NEW: Import serial library
import time   # <-- NEW: To add a small delay

# --- SERIAL COMMUNICATION SETUP ---
# CRITICAL: Find your Arduino's port.
# - Windows: It's 'COM3', 'COM4', etc. (Check Arduino IDE > Tools > Port)
# - Mac/Linux: It's '/dev/tty.usbmodem...' or '/dev/ttyUSB0', etc.
SERIAL_PORT = 'COM4'  # <-- CHANGE THIS TO YOUR ARDUINO'S PORT
BAUD_RATE = 9600

try:
    arduino = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=0.1)
    print(f"Connected to Arduino on {SERIAL_PORT}")
    time.sleep(2) # Wait 2 seconds for the Arduino to reset
except serial.SerialException as e:
    print(f"Error: Could not connect to Arduino on {SERIAL_PORT}. {e}")
    print("Please check the port and ensure the Arduino is plugged in.")
    arduino = None
# ----------------------------------

# Load the model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: model.p file not found.")
    print("Please run create_dataset.py and train_classifier.py first.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

last_sent_char = '' # To avoid flooding the serial port

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    current_char_to_send = 'N' # Default: 'N' for "No gesture"

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # --- Data Normalization (MUST MATCH create_dataset.py) ---
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            min_x = min(x_)
            min_y = min(y_)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min_x)
                data_aux.append(hand_landmarks.landmark[i].y - min_y)
            
            if len(data_aux) != 42:
                continue

            # --- Prediction ---
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0] # e.g., 'A', 'B', 'L'
            
            current_char_to_send = predicted_character # Set char to send

            # --- Draw Bounding Box and Text ---
            x1 = int(min_x * W) - 10
            y1 = int(min_y * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # --- NEW: Send to Arduino ---
    # Only send if the gesture has changed (or if it's the first time)
    if arduino and current_char_to_send != last_sent_char:
        # Send the single character, encoded as bytes
        arduino.write(current_char_to_send.encode()) 
        print(f"Sent to Arduino: {current_char_to_send}")
        last_sent_char = current_char_to_send
    # ----------------------------

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Cleanup
cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.write('N'.encode()) # Tell Arduino to clear the screen
    arduino.close()
    print("Arduino connection closed.")