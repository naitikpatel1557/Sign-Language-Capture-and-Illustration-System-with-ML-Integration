#! C:\Users\Asus\Python\mvenv\Scripts\python.exe

import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Iterate over each label directory (e.g., 'A', 'B', 'L')
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    print(f"Processing directory: {dir_}")
    
    # Iterate over each image in the label directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img_full_path = os.path.join(dir_path, img_path)
        
        data_aux = [] # Features for this one image
        x_ = []       # To store all x-coordinates
        y_ = []       # To store all y-coordinates

        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Warning: Could not read image {img_full_path}. Skipping.")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            # Assuming only one hand per image for this dataset
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # --- NEW CHECK ---
            # Check if MediaPipe returned the expected 21 landmarks
            if len(hand_landmarks.landmark) != 21:
                print(f"Warning: Incomplete landmarks in {img_full_path}. Expected 21, got {len(hand_landmarks.landmark)}. Skipping.")
                continue
            
            # First pass: Collect all coordinates to find min values
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Find the minimum x and y
            min_x = min(x_)
            min_y = min(y_)

            # Second pass: Normalize coordinates by subtracting the min
            # This makes the features relative to the hand's top-left corner
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min_x)
                data_aux.append(y - min_y)
            
            # --- NEW ROBUST CHECK ---
            # Final check to ensure the feature vector is *exactly* 42 elements
            if len(data_aux) == 42:
                # Add the normalized features and their label
                data.append(data_aux)
                labels.append(dir_)
            else:
                # This case should be rare but catches bugs
                print(f"Warning: Incorrect feature length ({len(data_aux)}) for {img_full_path}. Skipping.")
        
        else:
            print(f"Warning: No hand detected in {img_full_path}. Skipping.")

print(f"Dataset creation complete. Found {len(data)} valid samples.")

# Save the normalized data and labels to a pickle file
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print("data.pickle file saved.")