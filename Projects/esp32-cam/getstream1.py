import requests
import cv2
import numpy as np

# stream_url = "http://192.168.1.6/sustain?stream=0"
stream_url = "http://192.168.1.6/sustain?stream=0"  # Replace with your ESP32-CAM stream URL
username = "ESPCAM"
password = "1234"

response = requests.get(stream_url, auth=(username, password), stream=True)
bytes_data = bytes()

try:
    # for chunk in response.iter_content(chunk_size=4096):
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:  # Check if chunk is not empty
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')  # JPEG start
            b = bytes_data.find(b'\xff\xd9')  # JPEG end
            if a != -1 and b != -1:
                jpg_frame = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                # Debug: Check frame size
                print(f"JPEG Frame Size: {len(jpg_frame)} bytes")
                if len(jpg_frame) == 0:
                    print("Empty JPEG frame!")
                    continue
                # Decode and display
                img = cv2.imdecode(np.frombuffer(jpg_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imshow('Stream', img)
                else:
                    print("Failed to decode JPEG frame!")
                if cv2.waitKey(1) == ord('q'):
                    break
        else:
            print("Empty chunk received!")
except Exception as e:
    print(f"Error: {e}")
finally:
    cv2.destroyAllWindows()