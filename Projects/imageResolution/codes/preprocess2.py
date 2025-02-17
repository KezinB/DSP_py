import os
from skimage.io import imread, imsave
from skimage.transform import resize

# Define paths for the high-resolution image folders
hr_train_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\CIFAR10\hrtrain"  # Replace with your HR train folder path
hr_test_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\CIFAR10\hrtest"    # Replace with your HR test folder path

# Directories for saving low-resolution images
lr_train_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\preprocessed\lrTrain16"  # Replace with your desired LR train folder path
lr_test_folder = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\dataset\preprocessed\lrTest16"    # Replace with your desired LR test folder path
os.makedirs(lr_train_folder, exist_ok=True)
os.makedirs(lr_test_folder, exist_ok=True)

# Function to downsample images to lower resolution
def downsample_and_save(input_folder, output_folder, target_size, paths_file):
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder not found: {input_folder}")
        return  # Exit the function if folder does not exist

    image_paths = []
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path):  # Ensure it's a file
            try:
                # Read the high-resolution image
                hr_image = imread(file_path)
                
                # Downsample to low resolution
                lr_image = resize(hr_image, target_size, mode='reflect', anti_aliasing=True)
                
                # Save the low-resolution image
                lr_path = os.path.join(output_folder, file_name)
                imsave(lr_path, (lr_image * 255).astype("uint8"))  # Save as uint8 (pixel range: 0-255)
                image_paths.append(lr_path)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    
    # Save all the file paths to a text file
    with open(paths_file, "w") as f:
        for path in image_paths:
            f.write(path + "\n")

# Target size for low-resolution images (e.g., 16x16 for CIFAR-10)
target_size = (16, 16)

# Downsample and save for train and test datasets
downsample_and_save(hr_train_folder, lr_train_folder, target_size, "lr_train_paths.txt")
downsample_and_save(hr_test_folder, lr_test_folder, target_size, "lr_test_paths.txt")

print("Processing completed. Check for any missing folders or errors.")
