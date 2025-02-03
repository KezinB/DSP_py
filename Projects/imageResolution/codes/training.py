import os
import numpy as np
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, ReLU
from tensorflow.keras.optimizers import Adam
from skimage.io import imread
from skimage.transform import resize

# Define paths for the dataset and save locations for the models
hr_train_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\CIFAR10\train_images"
hr_test_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\CIFAR10\test_images"
lr_train_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\preprocessed\lrTrain16"
lr_test_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\preprocessed\lrTest16"

# Directories for saving models
cnn_model_save_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\models\cnn_model.h5"
svm_model_save_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\models\svm_model.pkl"

# Function to downsample images to low resolution
def downsample_and_save(input_folder, output_folder, target_size):
    image_paths = []
    images = []
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path):
            # Read the high-resolution image
            hr_image = imread(file_path)
            # Downsample to low resolution
            lr_image = resize(hr_image, target_size, mode='reflect', anti_aliasing=True)
            # Save the low-resolution image
            lr_image = (lr_image * 255).astype("uint8")
            images.append(lr_image)
            image_paths.append(file_path)
    
    return np.array(images)

# Downsample and load the HR and LR images
target_size = (16, 16)  # Define the target low resolution size (16x16 for CIFAR-10)

lr_train_images = downsample_and_save(lr_train_folder, lr_train_folder, target_size)
hr_train_images = downsample_and_save(hr_train_folder, hr_train_folder, target_size)
lr_test_images = downsample_and_save(lr_test_folder, lr_test_folder, target_size)
hr_test_images = downsample_and_save(hr_test_folder, hr_test_folder, target_size)

# Split the dataset into training and validation sets
lr_train_split, lr_val_split, hr_train_split, hr_val_split = train_test_split(
    lr_train_images, hr_train_images, test_size=0.2, random_state=42
)

# 1. Train CNN Model
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, (3, 3), padding='same'))  # 3 channels for RGB output
    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')
    return model

# Define the CNN model
cnn_model = create_cnn_model((16, 16, 3))

# Train the CNN model
cnn_model.fit(
    lr_train_split, hr_train_split,
    validation_data=(lr_val_split, hr_val_split),
    epochs=10,
    batch_size=32
)

# Save the trained CNN model
cnn_model.save(cnn_model_save_path)
print(f"CNN model saved at {cnn_model_save_path}")

# 2. Train SVM Model for Image Super-Resolution
# Flatten the images for SVM input (SVM does not take image arrays directly)
def flatten_images(images):
    return images.reshape(images.shape[0], -1)

# Flatten the train and test images
lr_train_flat = flatten_images(lr_train_split)
hr_train_flat = flatten_images(hr_train_split)

# Create and train the SVM model (SVR - Support Vector Regression)
svm_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
svm_model.fit(lr_train_flat, hr_train_flat)

# Save the trained SVM model
import joblib
joblib.dump(svm_model, svm_model_save_path)
print(f"SVM model saved at {svm_model_save_path}")
