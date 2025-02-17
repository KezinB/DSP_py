import tensorflow as tf
import os

# Define the directory where you want to save the dataset
dataset_dir = r'C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\CIFAR10'

# Create the directory if it doesn't exist
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Download and prepare the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Save the dataset to the specified directory
train_images_dir = os.path.join(dataset_dir, 'train_images')
test_images_dir = os.path.join(dataset_dir, 'test_images')

# Create directories for training and testing images
if not os.path.exists(train_images_dir):
    os.makedirs(train_images_dir)
if not os.path.exists(test_images_dir):
    os.makedirs(test_images_dir)

# Save training images
for i in range(train_images.shape[0]):
    img_path = os.path.join(train_images_dir, f'train_img_{i}.png')
    tf.keras.preprocessing.image.save_img(img_path, train_images[i])

# Save testing images
for i in range(test_images.shape[0]):
    img_path = os.path.join(test_images_dir, f'test_img_{i}.png')
    tf.keras.preprocessing.image.save_img(img_path, test_images[i])

print("CIFAR-10 dataset downloaded and saved in the specified folder.")
