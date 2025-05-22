import cv2

# Load pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture (0 for webcam or file path for video)
# cap = cv2.VideoCapture(0)  # For ESP32-CAM stream, use: cap = cv2.VideoCapture('http://IP_ADDRESS/stream')
# cap = cv2.VideoCapture('http://192.168.1.100/stream')
cap = cv2.VideoCapture('http://192.168.4.1/')
# cap = cv2.VideoCapture('http://192.168.1.200')

while True:
    # Read frame from video source
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display result
    cv2.imshow('Visual Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()