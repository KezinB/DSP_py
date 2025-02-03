import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Define paths for the low-resolution and high-resolution image folders
lr_train_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\preprocessed\lrTrain16"
hr_train_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\CIFAR10\train_images"
lr_test_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\preprocessed\lrTest16"
hr_test_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\CIFAR10\test_images"

# Function to load images from a directory and preprocess them
def load_and_preprocess_images(lr_folder, hr_folder, target_size=(16, 16)):
    lr_images = []
    hr_images = []
    filenames = os.listdir(lr_folder)
    
    for filename in filenames:
        lr_image_path = os.path.join(lr_folder, filename)
        hr_image_path = os.path.join(hr_folder, filename)

        if os.path.isfile(lr_image_path) and os.path.isfile(hr_image_path):
            # Load the low-resolution and high-resolution images
            lr_image = imread(lr_image_path)
            hr_image = imread(hr_image_path)
            
            # Resize to target size if needed (make sure images are of same size)
            lr_image = resize(lr_image, target_size, mode='reflect', anti_aliasing=True)
            hr_image = resize(hr_image, target_size, mode='reflect', anti_aliasing=True)
            
            # Normalize and convert to array
            lr_images.append(lr_image)
            hr_images.append(hr_image)
    
    return np.array(lr_images), np.array(hr_images)

# Load and preprocess the images
lr_train_images, hr_train_images = load_and_preprocess_images(lr_train_folder, hr_train_folder, target_size=(16, 16))
lr_test_images, hr_test_images = load_and_preprocess_images(lr_test_folder, hr_test_folder, target_size=(16, 16))

# Normalize the images to the range [0, 1]
lr_train_images = lr_train_images.astype('float32') / 255.0
hr_train_images = hr_train_images.astype('float32') / 255.0
lr_test_images = lr_test_images.astype('float32') / 255.0
hr_test_images = hr_test_images.astype('float32') / 255.0

print("Data loaded and preprocessed successfully!")
