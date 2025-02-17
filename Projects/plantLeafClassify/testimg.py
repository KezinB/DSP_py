import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model(r'C:\Users\user\OneDrive\Documents\Codes\python\Projects\plantLeafClassify\models-trained\plant_leaf_classification_model1.h5')

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Load a new image
image_path = r'C:\Users\user\Downloads\image122.jpeg'
new_image = load_and_preprocess_image(image_path)

# Make predictions
predictions = model.predict(new_image)
predicted_class = np.argmax(predictions, axis=1)

# Map predicted class to label
class_labels = {0: 'Apple', 1: 'Black Pepper'}
predicted_label = class_labels[predicted_class[0]]

print(f'Predicted Class: {predicted_label}')
