import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import cv2

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

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # Assuming 2 classes: apple and black pepper
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))

# Save the model
model.save(r'C:\Users\user\OneDrive\Documents\Codes\python\Projects\plantLeafClassify\models-trained\plant_leaf_classification_model1.h5')
