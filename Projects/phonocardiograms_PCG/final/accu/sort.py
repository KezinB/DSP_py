import os
import shutil

# Define the source folder
source_folder = r"C:\Users\kezin\Downloads\dataset\SVM\augmented_data"

# Define the destination folders
aug_fetal_folder = r"C:\Users\kezin\Downloads\dataset\SVM\Aug_Fetal"
aug_noise_folder = r"C:\Users\kezin\Downloads\dataset\SVM\Aug_Noise"

# Create the destination folders if they don't exist
os.makedirs(aug_fetal_folder, exist_ok=True)
os.makedirs(aug_noise_folder, exist_ok=True)

# Loop through each file in the source folder
for filename in os.listdir(source_folder):
    if filename.startswith("aug_fetal"):
        shutil.move(os.path.join(source_folder, filename), aug_fetal_folder)
    elif filename.startswith("aug_noise"):
        shutil.move(os.path.join(source_folder, filename), aug_noise_folder)

print("Files have been moved successfully!")
