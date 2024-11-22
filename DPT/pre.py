# import cv2
# import os

# # Define the folder containing the images
# input_folder = "input/gt"
# output_folder = os.path.join(input_folder, "preprocessed_images")
# os.makedirs(output_folder, exist_ok=True)

# # Define preprocessing parameters
# resize_dim = (128, 128)  # Resize to 128x128 pixels
# normalize_range = (0, 1)  # Normalize pixel values to [0, 1]

# for filename in os.listdir(input_folder):
#     if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#         img_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename)
        
#         # Read the image
#         img = cv2.imread(img_path)
        
#         # Resize the image
#         img_resized = cv2.resize(img, resize_dim)
        
#         # Normalize pixel values
#         img_normalized = img_resized / 255.0  # Convert to [0, 1]
        
#         # Convert back to 8-bit for saving
#         img_to_save = (img_normalized * 255).astype('uint8')
#         cv2.imwrite(output_path, img_to_save)
        
#         print(f"Processed and saved: {filename}")

# print(f"All images have been preprocessed and saved in {output_folder}.")


# import os
# from PIL import Image

# # Input and output directories
# input_dir = 'input/gt'  # Replace with the folder containing your images
# output_dir = 'input/gt/rotate'
# os.makedirs(output_dir, exist_ok=True)

# # Rotate all images in the input directory
# for filename in os.listdir(input_dir):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Supported image formats
#         file_path = os.path.join(input_dir, filename)
#         with Image.open(file_path) as img:
#             # Rotate the image by 90 degrees
#             rotated_img = img.rotate(-90, expand=True)
#             rotated_img.save(os.path.join(output_dir, filename))

# print(f"All images have been rotated and saved in the '{output_dir}' folder.")

import os
from PIL import Image

# Input and output directories
input_dir = 'input/gt'  # Replace with the folder containing your images
output_dir = 'input/gt'
os.makedirs(output_dir, exist_ok=True)

# Mirror all images in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Supported image formats
        file_path = os.path.join(input_dir, filename)
        with Image.open(file_path) as img:
            # Mirror the image horizontally (left-right flip)
            mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mirrored_img.save(os.path.join(output_dir, filename))

print(f"All images have been mirrored and saved in the '{output_dir}' folder.")
