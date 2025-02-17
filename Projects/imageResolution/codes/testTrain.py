import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.optimizers import Adam
from skimage.io import imread
from skimage.transform import resize

# Define paths for the dataset and save locations for the models
hr_train_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\CIFAR10\hrtrain"
hr_test_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\CIFAR10\hrtest"
lr_train_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\preprocessed\lrTrain16"
lr_test_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\preprocessed\lrTest16"

# Directories for saving models
cnn_model_save_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\models\CNN\srcnn_model.h5"
svm_model_save_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\models\SVM\svm_model.pkl"

# Define the desired image shape
image_shape = (16, 16, 3)

# Function to load and resize images from a folder
def load_and_resize_images(folder_path, image_shape):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            img = imread(img_path)
            img_resized = resize(img, image_shape)
            images.append(img_resized)
    return np.array(images)

# Load and resize HR and LR images
hr_train_images = load_and_resize_images(hr_train_folder, image_shape)
hr_test_images = load_and_resize_images(hr_test_folder, image_shape)
lr_train_images = load_and_resize_images(lr_train_folder, image_shape)
lr_test_images = load_and_resize_images(lr_test_folder, image_shape)

# Split the dataset into training and validation sets
lr_train_split, lr_val_split, hr_train_split, hr_val_split = train_test_split(
    lr_train_images, hr_train_images, test_size=0.2, random_state=42
)

# Train SRCNN Model (Super-Resolution Convolutional Neural Network)
def create_srcnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(64, (9, 9), activation='relu', padding='same'))
    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(3, (5, 5), activation='linear', padding='same'))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
    return model

# Define and train the SRCNN model
srcnn_model = create_srcnn_model(image_shape)
srcnn_model.fit(
    lr_train_split, hr_train_split,
    validation_data=(lr_val_split, hr_val_split),
    epochs=10,
    batch_size=32
)

# Save the trained SRCNN model
srcnn_model.save(cnn_model_save_path)

# Flatten the images for SVM input
def flatten_images(images):
    return images.reshape(images.shape[0], -1)

lr_train_flat = flatten_images(lr_train_split)
hr_train_flat = flatten_images(hr_train_split)

# Create and train the SVM model (SVR - Support Vector Regression)
svm_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
svm_model.fit(lr_train_flat, hr_train_flat)

# Save the trained SVM model
import joblib
joblib.dump(svm_model, svm_model_save_path)
