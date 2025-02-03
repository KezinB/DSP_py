import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os

# Define directories for apple and black pepper leaf images
apple_dir = r'C:\Users\user\OneDrive\Documents\Codes\python\Projects\plantLeafClassify\dataset\PlantVillage-Dataset\Apple___healthy'
pepper_dir = r'C:\Users\user\OneDrive\Documents\Codes\python\Projects\plantLeafClassify\dataset\PlantVillage-Dataset\Pepper,_bell___healthy'

# Function to load images
def load_images(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

# Load apple and pepper images
apple_images, apple_labels = load_images(apple_dir, 0)
pepper_images, pepper_labels = load_images(pepper_dir, 1)

# Combine the images and labels
X = np.concatenate((apple_images, pepper_images), axis=0)
y = np.concatenate((apple_labels, pepper_labels), axis=0)

# Split the dataset into training, validation, and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Load the saved model
model = load_model(r'C:\Users\user\OneDrive\Documents\Codes\python\Projects\plantLeafClassify\models-trained\plant_leaf_classification_model1.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
