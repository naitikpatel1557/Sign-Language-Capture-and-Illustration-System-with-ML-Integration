#! C:\Users\Asus\Python\mvenv\Scripts\python.exe
import os
import cv2

# --- Configuration ---
DATA_DIR = './data'
NUM_IMAGES_TO_COLLECT = 50 # You can change this
# ---------------------

# Get gesture name from user
gesture_name = input("Enter gesture name (e.g., 'A', 'B', 'L'): ")

# Create the directory for the new gesture if it doesn't exist
gesture_path = os.path.join(DATA_DIR, gesture_name)
if not os.path.exists(gesture_path):
    os.makedirs(gesture_path)
    print(f"Created directory: {gesture_path}")

# Find the starting number for images
start_index = 0
existing_files = os.listdir(gesture_path)
if existing_files:
    # Find the highest number in existing filenames like 'image_XXX.jpg'
    indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith('image_')]
    if indices:
        start_index = max(indices) + 1

print(f"Starting image save index at: {start_index}")

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

img_counter = 0

print("\n--- Starting Image Collection ---")
print(f"Saving to: {gesture_path}")
print("Position your hand and press 's' to save an image.")
print("Move your hand slightly between each capture.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Show the frame
    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Saved: {img_counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press 's' to save, 'q' to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Collect Images', frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break
    
    # Save
    if key == ord('s'):
        img_name = f"image_{start_index + img_counter}.jpg"
        img_save_path = os.path.join(gesture_path, img_name)
        
        cv2.imwrite(img_save_path, frame)
        
        print(f"Saved: {img_save_path}")
        img_counter += 1

print(f"\nCollection complete. Saved {img_counter} images.")
cap.release()
cv2.destroyAllWindows()