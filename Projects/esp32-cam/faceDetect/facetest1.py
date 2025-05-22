import requests
import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

stream_url = "http://192.168.1.6/sustain?stream=0"
username = "ESPCAM"
password = "1234"

response = requests.get(stream_url, auth=(username, password), stream=True)
bytes_data = bytes()

try:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg_frame = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                
                # Decode frame
                img = cv2.imdecode(np.frombuffer(jpg_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30))
                    
                    # Draw rectangles around detected faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display the resulting frame
                    cv2.imshow('Face Detection', img)
                
                if cv2.waitKey(1) == ord('q'):
                    break
except Exception as e:
    print(f"Error: {e}")
finally:
    cv2.destroyAllWindows()