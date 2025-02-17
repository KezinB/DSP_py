import torch
from PIL import Image
import numpy as np
from esrgan_pytorch import ESRGAN

def upscale_image(input_path, output_path, scale=4):
    # Load the ESRGAN model
    model = ESRGAN.from_pretrained('esrgan')

    # Load the input image
    image = Image.open(input_path).convert('RGB')

    # Convert image to numpy array
    np_img = np.array(image)

    # Perform super-resolution
    sr_image = model.predict(np_img)

    # Convert back to PIL image and save
    sr_image = Image.fromarray(sr_image)
    sr_image.save(output_path)
    print(f"Upscaled image saved to: {output_path}")

if __name__ == "__main__":
    input_image_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\test\input.jpg"      # Path to your input image
    output_image_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\test\output.jpg"   # Path to save the upscaled image

    upscale_image(input_image_path, output_image_path)
