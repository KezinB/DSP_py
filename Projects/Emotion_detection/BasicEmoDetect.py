import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit the application.")

while True:
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        print("Failed to grab frame.")
        break

    try:
        # Perform emotion analysis
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Get the emotion with the highest score
        emotion = result[0]['dominant_emotion']

        # Display emotion in terminal
        print(f"Detected Emotion: {emotion}")

        # Display the emotion on the frame
        #cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    except Exception as e:
        print(f"Error detecting emotion: {e}")

    # Show the webcam feed with detected emotion
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
