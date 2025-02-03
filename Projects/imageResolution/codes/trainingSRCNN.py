import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, ReLU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from skimage.io import imread


# Define paths for the dataset and save locations for the models
hr_train_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\CIFAR10\hrtrain"
hr_test_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\CIFAR10\hrtest"
lr_train_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\preprocessed\lrTrain16"
lr_test_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\preprocessed\lrTest16"

# Directories for saving models
cnn_model_save_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\models\CNN\srcnn_model.h5"
svm_model_save_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\models\SVM\svm_model.pkl"

# Function to load images from a folder
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            img = imread(img_path)
            images.append(img)
    return np.array(images)

print("loding images from folder......")
# Load HR and LR images
hr_train_images = load_images_from_folder(hr_train_folder)
print("successfull loaded hr_train_images") 
hr_test_images = load_images_from_folder(hr_test_folder)
print("successfull loaded hr_test_images")
lr_train_images = load_images_from_folder(lr_train_folder)
print("successfull loaded lr_train_images")
lr_test_images = load_images_from_folder(lr_test_folder)
print("successfull loaded lr_test_images")
print("Data loaded successfully.")

# Split the dataset into training and validation sets
lr_train_split, lr_val_split, hr_train_split, hr_val_split = train_test_split(
    lr_train_images, hr_train_images, test_size=0.2, random_state=42
    
)
print("Data split successfully.")

# 1. Train SRCNN Model (Super-Resolution Convolutional Neural Network)

def create_srcnn_model(input_shape):
    print("Creating SRCNN model...")
    model = Sequential()
    model.add(Input(shape=input_shape))  # Define input shape explicitly
    model.add(Conv2D(64, (9, 9), activation='relu', padding='same'))  # First convolution layer
    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))  # Second convolution layer
    model.add(Conv2D(3, (5, 5), activation='linear', padding='same'))  # Output layer (3 channels for RGB)
    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
    print("Model compiled successfully.")
    return model


# Define the SRCNN model
srcnn_model = create_srcnn_model((16, 16, 3))

# Train the SRCNN model
print("Training SRCNN model...")
srcnn_model.fit(
    lr_train_split, hr_train_split,
    validation_data=(lr_val_split, hr_val_split),
    epochs=10,
    batch_size=32
)
print("Training completed.")

# Save the trained SRCNN model
srcnn_model.save(cnn_model_save_path)
print(f"SRCNN model saved at {cnn_model_save_path}")

# 2. Train SVM Model for Image Super-Resolution
# Flatten the images for SVM input (SVM does not take image arrays directly)
def flatten_images(images):
    return images.reshape(images.shape[0], -1)

# Flatten the train and test images
lr_train_flat = flatten_images(lr_train_split)
hr_train_flat = flatten_images(hr_train_split)

# Create and train the SVM model (SVR - Support Vector Regression)
print("Training SVM model...")
svm_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
svm_model.fit(lr_train_flat, hr_train_flat)

# Save the trained SVM model
import joblib
joblib.dump(svm_model, svm_model_save_path)
print(f"SVM model saved at {svm_model_save_path}")
