import cv2
import numpy as np

# Initialize HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (640, 480))

    # Detect people in the frame
    boxes, weights = hog.detectMultiScale(frame_resized, winStride=(8, 8))

    # Create a copy of the frame for applying heat signatures
    heatmap_frame = frame_resized.copy()

    for (x, y, w, h) in boxes:
        # Extract the region of interest (ROI) for the detected human
        roi = heatmap_frame[y:y+h, x:x+w]

        # Convert ROI to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply a heatmap to the ROI
        heatmap_roi = cv2.applyColorMap(gray_roi, cv2.COLORMAP_JET)

        # Replace the ROI in the original frame with the heatmap
        heatmap_frame[y:y+h, x:x+w] = heatmap_roi

    # Display the resulting frame with simulated heat signatures
    cv2.imshow('Simulated Heat Signature', heatmap_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
