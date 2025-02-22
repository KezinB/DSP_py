import os
import random
import shutil

# Paths
source_folder = r"C:\Users\kezin\Downloads\archive - Copy\test"  # Change this to your source folder
destination_folder = r"C:\Users\kezin\Downloads\compData\Test"  # Change this to your destination folder

# Ensure destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Iterate over subfolders in the source directory
for subdir in os.listdir(source_folder):
    subdir_path = os.path.join(source_folder, subdir)
    
    # Check if it's a directory
    if os.path.isdir(subdir_path):
        images = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        # Select 100 random images (or all if less than 100 exist)
        selected_images = random.sample(images, min(20, len(images)))
        
        # Create corresponding subfolder in destination
        dest_subdir_path = os.path.join(destination_folder, subdir)
        os.makedirs(dest_subdir_path, exist_ok=True)
        
        # Copy selected images
        for img in selected_images:
            shutil.copy(os.path.join(subdir_path, img), os.path.join(dest_subdir_path, img))

print("Random selection complete!")
