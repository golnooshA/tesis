import os
import cv2
import torch
from torchvision.utils import save_image
from net.CycleGAN_generator import CycleGANGenerator

# Set device (CPU or GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
input_path = './data/UDD' 
output_path = './test/output/UDD'  # Enhanced output images
model_path = './saved_models/UDD/generator_a2b.pth'  # Trained generator model

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Load Generator Model
generator = CycleGANGenerator(3, 3).to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

# Get sorted list of input images
image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    print("No valid images found in the input directory.")
    exit()

print(f"Found {len(image_files)} images. Starting processing...")

# Process each image
for i, image_file in enumerate(image_files, start=1):
    try:
        # Read and preprocess the input image
        img_path = os.path.join(input_path, image_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Unable to read image {img_path}. Skipping...")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))  # Resize to match model input size

        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0

        # Generate enhanced output
        with torch.no_grad():
            enhanced_image = generator(img_tensor)

        # Save the output image
        output_file = os.path.join(output_path, os.path.splitext(image_file)[0] + "_enhanced.png")
        save_image(enhanced_image, output_file, nrow=1, normalize=True)
        print(f"Processed and saved: {output_file}")

    except Exception as e:
        print(f"Error processing {image_file}: {e}. Skipping...")

print("Processing complete.")
