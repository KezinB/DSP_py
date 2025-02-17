import cv2
from deepface import DeepFace
import time
import serial
import matplotlib.pyplot as plt
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()
    
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  # Ensure this matches your Arduino's port

print("Press 'q' to quit the application.")

# Initialize plot
plt.ion()
fig, ax = plt.subplots()

# Emotions in the order from happy to sad
emotions = ['happy', 'surprise', 'neutral', 'fear', 'sad', 'angry', 'disgust']
emotion_mapping = {emotion: idx for idx, emotion in enumerate(emotions)}

# Lists to store time and emotions
time_stamps = []
emotion_values = []

while True:
    ret, frame = cap.read()  # Read frame from the webcam
    if not ret:
        print("Error: Failed to capture frame.")
        break

    try:
        # Perform face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Perform emotion analysis using DeepFace
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            # Get the dominant emotion
            dominant_emotion = result[0]['dominant_emotion']
            print(f"Detected Emotion: {dominant_emotion}")
            arduino.write(f"{dominant_emotion}\n".encode())
            
            # Determine box color based on the detected emotion
            if dominant_emotion in ['happy', 'neutral']:
                box_color = (0, 255, 0)  # Green
            else:
                box_color = (0, 0, 255)  # Red

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Append current time and detected emotion to lists
            current_time = time.time()
            time_stamps.append(current_time)
            emotion_values.append(emotion_mapping[dominant_emotion])

            # Filter data to show only the latest 2 minutes
            two_minutes_ago = current_time - 120  # 2 minutes in seconds
            filtered_time_stamps = [t for t in time_stamps if t >= two_minutes_ago]
            filtered_emotion_values = emotion_values[-len(filtered_time_stamps):]
            
            # Update plot
            ax.clear()
            ax.plot(filtered_time_stamps, filtered_emotion_values, marker='o', linestyle='-')
            ax.set_yticks(np.arange(len(emotions)))
            ax.set_yticklabels(emotions)
            ax.set_xlabel('Time')
            ax.set_ylabel('Emotion')
            ax.set_title('Live Emotion Detection Over Time')
            plt.pause(0.1)  # Pause to update plot
        else:
            print("No face detected.")
            arduino.write(f"no face\n".encode())
            
        # Display the frame with detected emotion
        cv2.imshow('Emotion Detection', frame)

    except Exception as e:
        print(f"Error analyzing emotion: {e}")

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
