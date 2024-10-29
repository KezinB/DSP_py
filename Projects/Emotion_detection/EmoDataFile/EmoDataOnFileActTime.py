import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
from collections import deque
import time
import os
from datetime import datetime

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit the application.")

# Initialize plot elements
plt.ion()  # Interactive mode for live plotting
fig, ax = plt.subplots()
time_data = deque(maxlen=100)  # Store up to 100 timestamps for smoother plotting
emotion_data = deque(maxlen=100)  # Store up to 100 emotions

# Reversed Emotion categories (y-axis values)
emotions_list = ['neutral', 'surprise', 'sad', 'happy', 'fear', 'disgust', 'angry']
y_emotion_map = {e: i for i, e in enumerate(emotions_list)}

line, = ax.plot([], [], 'bo-', label='Emotion')  # Blue circles and line

# Plot settings
ax.set_ylim(-1, len(emotions_list))  # Map emotions to y-values
ax.set_yticks(range(len(emotions_list)))
ax.set_yticklabels(emotions_list)  # Y-tick labels in reversed order
ax.set_xlabel('Time (s)')
ax.set_ylabel('Emotion')
ax.set_title('Live Emotion Detection')
ax.legend()

# Create a unique folder for storing data based on start time in the same directory as the script
start_time = datetime.now()
script_directory = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
folder_name = f"EmoDataFile_{start_time.strftime('%Y%m%d_%H%M%S')}"
folder_path = os.path.join(script_directory, folder_name)  # Ensure folder is created in the script's directory
os.makedirs(folder_path, exist_ok=True)

# Create a text file to store emotion and time data
file_path = os.path.join(folder_path, "emotion_data.txt")

with open(file_path, 'w') as f:
    f.write("Elapsed Time (s), Actual Time, Emotion\n")  # Header for the file

start_time_seconds = time.time()  # Store the actual start time in seconds

# Set the data collection frequency
data_collection_rate = 0.2  # 200 ms (5 times per second)

while True:
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    try:
        # Perform emotion analysis
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Get the emotion with the highest score
        emotion = result[0]['dominant_emotion']
        print(f"Detected Emotion: {emotion}")

        # Get the current actual time
        actual_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Update time and emotion data for plotting
        elapsed_time = time.time() - start_time_seconds
        time_data.append(elapsed_time)
        emotion_data.append(y_emotion_map[emotion])

        # Write to the text file
        with open(file_path, 'a') as f:
            f.write(f"{elapsed_time:.2f}, {actual_time}, {emotion}\n")  # Log elapsed time, actual time, and emotion

        # Update the graph
        line.set_xdata(time_data)
        line.set_ydata(emotion_data)
        ax.relim()  # Recompute the data limits
        ax.autoscale_view()  # Automatically adjust view to fit data
        plt.pause(data_collection_rate)  # Small delay for smoother updating

    except Exception as e:
        print(f"Error detecting emotion: {e}")

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    # Show the webcam feed with detected emotion
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final graph
