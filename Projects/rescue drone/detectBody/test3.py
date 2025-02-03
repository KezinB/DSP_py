import cv2

# Load the pre-trained body part detection model
protoFile = r"C:\Users\user\OneDrive\Documents\Codes\python\Projects\rescue drone\detectBody\pose_deploy_linevec.prototxt"
weightsFile = r"C:\Users\user\OneDrive\Documents\Codes\python\Projects\rescue drone\detectBody\pose_iter_440000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Define a function to detect and label body parts
def detect_body_parts(frame):
    inWidth = 368
    inHeight = 368
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

    # Points to be detected
    points = []
    for i in range(15):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        if prob > 0.1:
            points.append((int(x), int(y)))
        else:
            points.append(None)

    return points

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply inferno colormap
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    
    # Detect and label body parts
    points = detect_body_parts(frame)
    for pair in [ [0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [14,11], [8,9], [9,10], [11,12], [12,13] ]:
        partA = pair[0]
        partB = pair[1]
        
        if points[partA] and points[partB]:
            cv2.line(thermal, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(thermal, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(thermal, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    
    # Display the result
    cv2.imshow('Thermal Feed with Body Parts', thermal)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
