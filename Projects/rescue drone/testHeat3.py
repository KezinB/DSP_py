import cv2
import numpy as np

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to grayscale for detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Create a copy of the frame for heatmap
    heatmap_frame = frame.copy()

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for the detected face
        roi = gray_frame[y:y+h, x:x+w]

        # Apply a heatmap to the ROI
        heatmap_roi = cv2.applyColorMap(cv2.resize(roi, (w, h)), cv2.COLORMAP_JET)

        # Replace the original region in the frame with the heatmap
        heatmap_frame[y:y+h, x:x+w] = heatmap_roi

        # Optionally, draw a rectangle around the face (for debugging)
        cv2.rectangle(heatmap_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Display the resulting frame with simulated heat signatures
    cv2.imshow('Simulated Heat Signature', heatmap_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
