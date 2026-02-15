#! C:\Users\Asus\Python\mvenv\Scripts\python.exe


#! C:\Users\Asus\Python\mvenv\Scripts\python.exe

import pickle
import cv2
import mediapipe as mp
import time
from flask import Flask, render_template_string, jsonify
import threading
import serial
import numpy as np

# --- SERIAL COMMUNICATION SETUP ---
SERIAL_PORT = 'COM4'  
BAUD_RATE = 9600
arduino = None  
try:
    arduino = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=0.1)
    print(f"Connected to Arduino on {SERIAL_PORT}")
    time.sleep(2)
except serial.SerialException as e:
    print(f"Error: Could not connect to Arduino on {SERIAL_PORT}. {e}")
    print("Please check the port and ensure the Arduino is plugged in.")
# ----------------------------------

# --- FLASK WEB SERVER SETUP ---
app = Flask(__name__)
global_predicted_char = 'N'
char_lock = threading.Lock()
# -----------------------------

# Load the model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: model.p file not found. Please run create_dataset.py and train_classifier.py first.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# --- Function to handle video processing ---
def process_video():
    global global_predicted_char, char_lock 

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    last_sent_char = ''

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        current_char_to_send = 'N'

        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # --- Data Normalization ---
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
                
                if len(data_aux) == 42:
                    # --- Prediction ---
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = prediction[0]
                    current_char_to_send = predicted_character

                    # --- Draw Bounding Box and Text ---
                    x1 = int(min_x * W) - 10
                    y1 = int(min_y * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # --- Send to Arduino ---
        if arduino and current_char_to_send != last_sent_char:
            arduino.write(current_char_to_send.encode()) 
            print(f"Sent to Arduino: {current_char_to_send}")
            last_sent_char = current_char_to_send

        # --- Update the global character for the web server ---
        with char_lock:
            global_predicted_char = current_char_to_send
        
        # We still show the local window on the PC
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.write('N'.encode())
        arduino.close()
        print("Arduino connection closed.")
# ----------------------------------------------------


# --- Flask web server functions ---

@app.route('/')
def index():
    # Renders an HTML page with JavaScript to fetch and display the gesture
    # --- NEW: HTML AND JAVASCRIPT ARE UPDATED ---
    html_template = """
    <html>
    <head>
        <title>Sign Gesture Speaker</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
        <style>
            body {
                background-color: #1a1a1a;
                color: #e0e0e0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                font-family: 'Arial', sans-serif;
                overflow: hidden;
            }
            #gesture-display {
                font-size: 20vh; /* Made text smaller to fit 'Tap to Start' */
                font-weight: bold;
                text-align: center;
                color: #ffffff;
            }
        </style>
    </head>
    <body>
        <h1 id="gesture-display">Tap screen to start</h1>
        <script>
            // NEW: Track the last spoken gesture to avoid repeats
            let lastSpokenGesture = '';

            // NEW: Function to speak the text
            function speakText(text) {
                // Check if it's a new gesture and not 'N'
                if (text !== lastSpokenGesture && text !== 'N') {
                    // Create a speech request
                    let utterance = new SpeechSynthesisUtterance(text);
                    utterance.lang = 'en-US'; // Set language
                    
                    // Speak the text
                    window.speechSynthesis.speak(utterance);
                    
                    // Update the last spoken gesture
                    lastSpokenGesture = text;

                } else if (text === 'N') {
                    // Reset when no gesture is detected
                    lastSpokenGesture = 'N';
                }
            }

            function fetchGesture() {
                // This function asks the server for the current gesture
                fetch('/get_gesture')
                    .then(response => response.json())
                    .then(data => {
                        let gesture = data.gesture;
                        let displayElement = document.getElementById('gesture-display');
                        
                        if (gesture === 'N') {
                            displayElement.innerText = '-';
                        } else {
                            displayElement.innerText = gesture;
                        }
                        
                        // NEW: Speak the gesture
                        speakText(gesture);
                    })
                    .catch(err => {
                        console.error(err);
                        document.getElementById('gesture-display').innerText = 'X';
                    });
            }
            
            // NEW: Add a 'click' listener to start the system
            // This is required by mobile browsers to allow audio
            document.body.addEventListener('click', function startApp() {
                console.log("Audio enabled. Starting system.");
                
                // Speak a welcome message
                let welcome = new SpeechSynthesisUtterance('System activated.');
                welcome.lang = 'en-US';
                window.speechSynthesis.speak(welcome);

                // Start polling for gestures
                setInterval(fetchGesture, 250); // Poll 4 times/sec
                
                // Run it once immediately
                fetchGesture();

            }, { once: true }); // {once: true} makes the listener run only once
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)
# -----------------------------------------------

@app.route('/get_gesture')
def get_gesture():
    # This endpoint is unchanged. It just provides the data.
    global global_predicted_char, char_lock
    with char_lock:
        char_to_send = global_predicted_char
    
    return jsonify(gesture=char_to_send)
# --------------------------------------


# --- Main entry point ---
if __name__ == '__main__':
    # Start the video processing in a separate thread
    video_thread = threading.Thread(target=process_video)
    video_thread.daemon = True
    video_thread.start()
    
    # Start the Flask web server
    print("\n--- Web Server Starting ---")
    print(f"Open this URL in your mobile browser:")
    print(f"http://<YOUR_PC_IP>:5000")
    print("Replace <YOUR_PC_IP> with your PC's local IP address (find with 'ipconfig').")
    print("---------------------------\n")
    app.run(host='0.0.0.0', port=5000, debug=False)







# To show the Gesture in writing in my mobile

# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import serial
# import time
# from flask import Flask, render_template_string, jsonify  # <-- NEW: Added jsonify
# import threading

# # --- SERIAL COMMUNICATION SETUP ---
# SERIAL_PORT = 'COM4'  
# BAUD_RATE = 9600
# arduino = None  
# try:
#     arduino = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=0.1)
#     print(f"Connected to Arduino on {SERIAL_PORT}")
#     time.sleep(2)
# except serial.SerialException as e:
#     print(f"Error: Could not connect to Arduino on {SERIAL_PORT}. {e}")
#     print("Please check the port and ensure the Arduino is plugged in.")
# # ----------------------------------

# # --- FLASK WEB SERVER SETUP ---
# app = Flask(__name__)
# # NEW: Global variable to hold the latest predicted character
# global_predicted_char = 'N'
# # NEW: Lock for thread-safe access to the character
# char_lock = threading.Lock()i 
# # -----------------------------

# # Load the model
# try:
#     model_dict = pickle.load(open('./model.p', 'rb'))
#     model = model_dict['model']
# except FileNotFoundError:
#     print("Error: model.p file not found. Please run create_dataset.py and train_classifier.py first.")
#     exit()
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit()

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# # --- Function to handle video processing ---
# def process_video():
#     # NEW: Need to declare globals to write to them
#     global global_predicted_char, char_lock 

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         exit()

#     last_sent_char = ''

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to grab frame.")
#             break

#         H, W, _ = frame.shape
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         current_char_to_send = 'N'

#         results = hands.process(frame_rgb)
        
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style())

#                 # --- Data Normalization ---
#                 data_aux = []
#                 x_ = []
#                 y_ = []

#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     x_.append(x)
#                     y_.append(y)

#                 min_x = min(x_)
#                 min_y = min(y_)

#                 for i in range(len(hand_landmarks.landmark)):
#                     data_aux.append(hand_landmarks.landmark[i].x - min_x)
#                     data_aux.append(hand_landmarks.landmark[i].y - min_y)
                
#                 if len(data_aux) == 42:
#                     # --- Prediction ---
#                     prediction = model.predict([np.asarray(data_aux)])
#                     predicted_character = prediction[0]
#                     current_char_to_send = predicted_character

#                     # --- Draw Bounding Box and Text ---
#                     x1 = int(min_x * W) - 10
#                     y1 = int(min_y * H) - 10
#                     x2 = int(max(x_) * W) + 10
#                     y2 = int(max(y_) * H) + 10
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#                     cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

#         # --- Send to Arduino ---
#         if arduino and current_char_to_send != last_sent_char:
#             arduino.write(current_char_to_send.encode()) 
#             print(f"Sent to Arduino: {current_char_to_send}")
#             last_sent_char = current_char_to_send

#         # --- NEW: Update the global character for the web server ---
#         with char_lock:
#             global_predicted_char = current_char_to_send
        
#         # We still show the local window on the PC
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Cleanup
#     cap.release()
#     cv2.destroyAllWindows()
#     if arduino:
#         arduino.write('N'.encode())
#         arduino.close()
#         print("Arduino connection closed.")
# # ----------------------------------------------------


# # --- NEW: Flask web server functions ---

# @app.route('/')
# def index():
#     # Renders an HTML page with JavaScript to fetch and display the gesture
#     html_template = """
#     <html>
#     <head>
#         <title>Sign Gesture Display</title>
#         <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
#         <style>
#             body {
#                 background-color: #1a1a1a; /* Dark background */
#                 color: #e0e0e0; /* Light text */
#                 display: flex;
#                 justify-content: center;
#                 align-items: center;
#                 height: 100vh;
#                 margin: 0;
#                 font-family: 'Arial', sans-serif;
#                 overflow: hidden; /* Disable scrolling */
#             }
#             #gesture-display {
#                 font-size: 40vh; /* Make text as large as possible */
#                 font-weight: bold;
#                 text-align: center;
#                 color: #ffffff; /* Bright white text */
#             }
#         </style>
#     </head>
#     <body>
#         <h1 id="gesture-display">-</h1> <!-- Start with a placeholder -->
#         <script>
#             function fetchGesture() {
#                 // This function asks the server for the current gesture
#                 fetch('/get_gesture')
#                     .then(response => response.json())
#                     .then(data => {
#                         let gesture = data.gesture;
#                         let displayElement = document.getElementById('gesture-display');
                        
#                         if (gesture === 'N') {
#                             displayElement.innerText = '-'; // Show a dash for 'N'
#                         } else {
#                             displayElement.innerText = gesture; // Show the letter
#                         }
#                     })
#                     .catch(err => {
#                         console.error(err);
#                         document.getElementById('gesture-display').innerText = 'X'; // Show 'X' on error
#                     });
#             }
            
#             // Ask for the gesture 4 times per second (every 250ms)
#             setInterval(fetchGesture, 250);
            
#             // Run it once immediately on load
#             fetchGesture();
#         </script>
#     </body>
#     </html>
#     """
#     return render_template_string(html_template)


# @app.route('/get_gesture')
# def get_gesture():
#     # This is the new endpoint that the JavaScript calls
#     global global_predicted_char, char_lock
#     with char_lock:
#         char_to_send = global_predicted_char
    
#     # Return the character as a JSON object
#     return jsonify(gesture=char_to_send)
# # --------------------------------------


# # --- Main entry point ---
# if __name__ == '__main__':
#     # Start the video processing in a separate thread
#     video_thread = threading.Thread(target=process_video)
#     video_thread.daemon = True
#     video_thread.start()
    
#     # Start the Flask web server
#     # host='0.0.0.0' makes it accessible on your local network
#     print("\n--- Web Server Starting ---")
#     print(f"Open this URL in your mobile browser:")
#     print(f"http://<YOUR_PC_IP>:5000")
#     print("Replace <YOUR_PC_IP> with your PC's local IP address (find with 'ipconfig').")
#     print("---------------------------\n")
#     app.run(host='0.0.0.0', port=5000, debug=False)











# To open the webcam in my mobile

# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import serial
# import time
# from flask import Flask, Response, render_template_string  # <-- NEW: Import Flask
# import threading  # <-- NEW: To run Flask and OpenCV together

# # --- SERIAL COMMUNICATION SETUP ---
# SERIAL_PORT = 'COM4'  
# BAUD_RATE = 9600
# arduino = None  # <-- NEW: Initialize as None
# try:
#     arduino = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=0.1)
#     print(f"Connected to Arduino on {SERIAL_PORT}")
#     time.sleep(2)
# except serial.SerialException as e:
#     print(f"Error: Could not connect to Arduino on {SERIAL_PORT}. {e}")
#     print("Please check the port and ensure the Arduino is plugged in.")
# # ----------------------------------

# # --- FLASK WEB SERVER SETUP ---
# app = Flask(__name__)
# # Global variable to hold the latest processed frame
# output_frame = None
# # Lock to ensure thread-safe access to output_frame
# frame_lock = threading.Lock()
# # -----------------------------

# # Load the model
# try:
#     model_dict = pickle.load(open('./model.p', 'rb'))
#     model = model_dict['model']
# except FileNotFoundError:
#     print("Error: model.p file not found. Please run create_dataset.py and train_classifier.py first.")
#     exit()
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit()

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# # --- NEW: Function to handle video processing ---
# def process_video():
#     global output_frame, frame_lock

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         exit()

#     last_sent_char = ''

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to grab frame.")
#             break

#         H, W, _ = frame.shape
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         current_char_to_send = 'N'

#         results = hands.process(frame_rgb)
        
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style())

#                 # --- Data Normalization ---
#                 data_aux = []
#                 x_ = []
#                 y_ = []

#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     x_.append(x)
#                     y_.append(y)

#                 min_x = min(x_)
#                 min_y = min(y_)

#                 for i in range(len(hand_landmarks.landmark)):
#                     data_aux.append(hand_landmarks.landmark[i].x - min_x)
#                     data_aux.append(hand_landmarks.landmark[i].y - min_y)
                
#                 if len(data_aux) == 42:
#                     # --- Prediction ---
#                     prediction = model.predict([np.asarray(data_aux)])
#                     predicted_character = prediction[0]
#                     current_char_to_send = predicted_character

#                     # --- Draw Bounding Box and Text ---
#                     x1 = int(min_x * W) - 10
#                     y1 = int(min_y * H) - 10
#                     x2 = int(max(x_) * W) + 10
#                     y2 = int(max(y_) * H) + 10
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#                     cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

#         # --- Send to Arduino ---
#         if arduino and current_char_to_send != last_sent_char:
#             arduino.write(current_char_to_send.encode()) 
#             print(f"Sent to Arduino: {current_char_to_send}")
#             last_sent_char = current_char_to_send

#         # --- NEW: Update the global frame for streaming ---
#         with frame_lock:
#             # Encode the frame as JPEG
#             (flag, encoded_image) = cv2.imencode(".jpg", frame)
#             if flag:
#                 output_frame = encoded_image.tobytes()
        
#         # We can still show the local window if we want
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Cleanup
#     cap.release()
#     cv2.destroyAllWindows()
#     if arduino:
#         arduino.write('N'.encode())
#         arduino.close()
#         print("Arduino connection closed.")
# # ----------------------------------------------------


# # --- NEW: Flask web server functions ---

# @app.route('/')
# def index():
#     # Renders a simple HTML page that displays the video stream
#     return render_template_string(
#         '<html><head><title>Sign Language Stream</title></head>'
#         '<body style="background-color:black; margin:0; padding:0;">'
#         '<img src="{{ url_for(\'video_feed\') }}" style="width:100vw; height:100vh; object-fit:contain;" />'
#         '</body></html>'
#     )

# def generate_stream():
#     global output_frame, frame_lock
#     while True:
#         with frame_lock:
#             if output_frame is None:
#                 continue
            
#             # Yield the frame in the multipart format
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
        
#         # Control the stream speed slightly
#         time.sleep(0.03)

# @app.route('/video_feed')
# def video_feed():
#     # Returns the video streaming response
#     return Response(generate_stream(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
# # --------------------------------------


# # --- NEW: Main entry point ---
# if __name__ == '__main__':
#     # Start the video processing in a separate thread
#     video_thread = threading.Thread(target=process_video)
#     video_thread.daemon = True
#     video_thread.start()
    
#     # Start the Flask web server
#     # host='0.0.0.0' makes it accessible on your local network
#     print("\n--- Web Server Starting ---")
#     print(f"Open this URL in your mobile browser:")
#     print(f"http://<YOUR_PC_IP>:5000")
#     print("Replace <YOUR_PC_IP> with your PC's local IP address.")
#     print("---------------------------\n")
#     app.run(host='0.0.0.0', port=5000, debug=False)