import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a color map (e.g., COLORMAP_JET for heatmap effect)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Display the resulting frame
    cv2.imshow('Heat Signature Simulation', heatmap)

    # Press 'q' to exit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
