import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained SRCNN model
def load_srcnn_model(model_path='srcnn_model.h5'):
    model = load_model(model_path)
    print("SRCNN model loaded successfully.")
    return model

# Preprocess image for SRCNN
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)  # Convert to YCrCb color space
    y, cr, cb = cv2.split(image)  # Extract Y (luminance) channel
    
    # Upscale the image (e.g., 2x using bicubic interpolation)
    height, width = y.shape
    y_upscaled = cv2.resize(y, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    cr_upscaled = cv2.resize(cr, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    cb_upscaled = cv2.resize(cb, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    
    # Normalize the Y channel for the model
    y_upscaled = y_upscaled.astype(np.float32) / 255.0
    y_upscaled = np.expand_dims(y_upscaled, axis=(0, -1))  # Add batch and channel dimensions
    
    return y_upscaled, cr_upscaled, cb_upscaled

# Postprocess SRCNN output
def postprocess_image(y_pred, cr, cb):
    # Convert Y channel back to the 0-255 range
    y_pred = (y_pred[0, :, :, 0] * 255.0).clip(0, 255).astype(np.uint8)
    
    # Merge Y, Cr, and Cb channels
    result_image = cv2.merge([y_pred, cr, cb])
    result_image = cv2.cvtColor(result_image, cv2.COLOR_YCrCb2BGR)  # Convert back to BGR color space
    
    return result_image

# Main function to apply SRCNN
def apply_srcnn(input_path, output_path, model_path='srcnn_model.h5'):
    # Load SRCNN model
    model = load_srcnn_model(model_path)
    
    # Preprocess input image
    y_upscaled, cr_upscaled, cb_upscaled = preprocess_image(input_path)
    
    # Predict high-resolution Y channel using SRCNN
    y_pred = model.predict(y_upscaled)
    
    # Postprocess and save the output image
    output_image = postprocess_image(y_pred, cr_upscaled, cb_upscaled)
    cv2.imwrite(output_path, output_image)
    print(f"Super-resolved image saved to: {output_path}")

# Input and output paths
input_image_path = r'C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\test\input.jpg'  # Replace with your image path
output_image_path = r'C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\test\output.jpg'  # Path to save the super-resolved image
model_path = r'C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\imageResolution\SRCNN-keras\3051crop_weight_200.h5'  # Path to your pre-trained SRCNN model

# Run the SRCNN process
apply_srcnn(input_image_path, output_image_path, model_path)
