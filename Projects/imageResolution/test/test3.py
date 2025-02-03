import cv2
from PIL import Image
import numpy as np

def upscale_image(input_path, output_path, scale=6):
    # Load the input image
    image = Image.open(input_path).convert('RGB')

    # Convert the image to a NumPy array
    np_img = np.array(image)

    # Get the dimensions of the original image
    height, width, channels = np_img.shape

    # Calculate the new dimensions
    new_width = width * scale
    new_height = height * scale

    # Resize the image using OpenCV's INTER_CUBIC interpolation
    sr_image = cv2.resize(np_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Convert the result back to a PIL image and save
    sr_image = Image.fromarray(sr_image)
    sr_image.save(output_path)
    print(f"Upscaled image saved to: {output_path}")

if __name__ == "__main__":
    input_image_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\test\input.jpg"      # Path to your input image
    output_image_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\test\output1.jpg"   # Path to save the upscaled image

    upscale_image(input_image_path, output_image_path)
