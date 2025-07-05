import os
from PIL import Image

def image_to_hex(image_path):
    img = Image.open(image_path).convert("RGB")
    pixels = list(img.getdata())
    return [f'{r:02x}{g:02x}{b:02x}' for r, g, b in pixels]

def convert_folder_to_hex(folder_path, output_file):
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ])
    
    all_hex_data = []

    for filename in image_files:
        path = os.path.join(folder_path, filename)
        print(f"Processing {filename}...")
        hex_data = image_to_hex(path)
        all_hex_data.extend(hex_data)  # Combine all images' hex data

    # Save to a file
    with open(output_file, 'w') as f:
        for hex_pixel in all_hex_data:
            f.write(hex_pixel + '\n')  # One pixel per line (optional)

    print(f"✅ Hex data saved to {output_file}")

# Example usage
folder_path = "C:\Users\kezin\Downloads\ezgif-split"
output_file = "output_hex_data.txt"
convert_folder_to_hex(folder_path, output_file)
