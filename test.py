# import os
# import cv2
# import numpy as np
# import torch
# from torchvision.utils import save_image
# from net.CycleGAN_generator import ResnetGenerator  # Use ResnetGenerator for compatibility

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Paths
# input_path = './datasets/trainA'
# output_path = './test'
# model_path = './DPT/weights/latest_net_G_A.pth'  # Path to your latest_net_G_A.pth

# # Ensure output directory exists
# os.makedirs(output_path, exist_ok=True)

# # Fix state_dict keys to match the generator structure
# def fix_state_dict_keys(state_dict, add_prefix="model."):
#     """Add a prefix to state_dict keys to match the ResnetGenerator structure."""
#     fixed_state_dict = {}
#     for key, value in state_dict.items():
#         if not key.startswith(add_prefix):
#             new_key = add_prefix + key
#         else:
#             new_key = key
#         fixed_state_dict[new_key] = value
#     return fixed_state_dict

# # Load and fix Generator Model
# generator = ResnetGenerator(input_nc=3, output_nc=3, n_blocks=9).to(device)
# raw_state_dict = torch.load(model_path, map_location=device)  # Load raw state_dict
# fixed_state_dict = fix_state_dict_keys(raw_state_dict)  # Fix keys to match ResnetGenerator
# generator.load_state_dict(fixed_state_dict, strict=False)  # Load fixed state_dict
# generator.eval()

# # Post-processing function
# def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
#     """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
#     lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
#     cl = clahe.apply(l)
#     lab = cv2.merge((cl, a, b))
#     return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# # Get sorted list of input images
# image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# if not image_files:
#     print("No valid images found in the input directory.")
#     exit()

# print(f"Found {len(image_files)} images. Starting processing...")

# # Process each image
# for i, image_file in enumerate(image_files, start=1):
#     try:
#         # Load RGB image
#         img_path = os.path.join(input_path, image_file)
#         img = cv2.imread(img_path)

#         if img is None:
#             print(f"Warning: Unable to read image {img_path}. Skipping...")
#             continue

#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (256, 256))  # Resizing to match training dimensions

#         # Convert to PyTorch tensor
#         img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
#         img_tensor = (img_tensor - 0.5) / 0.5  # Normalize to [-1, 1] for generator input

#         # Generate enhanced output
#         with torch.no_grad():
#             enhanced_image_tensor = generator(img_tensor)

#         # Convert tensor to numpy image
#         enhanced_image = enhanced_image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
#         enhanced_image = (enhanced_image * 0.5 + 0.5) * 255.0  # De-normalize to [0, 255]
#         enhanced_image = enhanced_image.clip(0, 255).astype(np.uint8)

#         # Post-process the enhanced image
#         enhanced_image = apply_clahe(enhanced_image, clip_limit=2.0, tile_grid_size=(8, 8))

#         # Save the output image
#         output_file = os.path.join(output_path, os.path.splitext(image_file)[0] + "_enhanced.png")
#         cv2.imwrite(output_file, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
#         print(f"Processed and saved: {output_file}")
#     except Exception as e:
#         print(f"Error processing {image_file}: {e}")

# print("Processing complete.")


import os
import cv2
import numpy as np
import torch
from torchvision.utils import save_image
from net.CycleGAN_generator import ResnetGenerator

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
input_path = './datasets/trainA'
depth_path = './datasets/depth'
output_path = './test/output'
model_path = './DPT/weights/latest_net_G_A.pth'

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Fix state_dict keys and adjust for additional depth channel
def fix_state_dict_keys_and_input(state_dict, add_prefix="model."):
    """Adjust the first layer weights to accept 4-channel input (RGB + Depth)."""
    fixed_state_dict = {}
    for key, value in state_dict.items():
        new_key = add_prefix + key if not key.startswith(add_prefix) else key

        # Adjust the weights for the first convolutional layer
        if "model.1.weight" in new_key and value.shape[1] == 3:  # 3-channel input in checkpoint
            print("Adjusting the first layer weights for 4-channel input...")
            depth_channel = value[:, :1, :, :].clone()
            new_weights = torch.cat((value, depth_channel), dim=1)
            fixed_state_dict[new_key] = new_weights
        else:
            fixed_state_dict[new_key] = value

    return fixed_state_dict

# Load Generator Model
generator = ResnetGenerator(input_nc=4, output_nc=3, n_blocks=9).to(device)
raw_state_dict = torch.load(model_path, map_location=device)
fixed_state_dict = fix_state_dict_keys_and_input(raw_state_dict)
generator.load_state_dict(fixed_state_dict, strict=False)
generator.eval()

# Post-processing functions
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def denormalize_image(image_tensor):
    """Convert generator output tensor back to a valid RGB image."""
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (image * 0.5 + 0.5) * 255.0  # De-normalize to [0, 255]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

# Get sorted list of input images and depth maps
image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
depth_files = [f for f in os.listdir(depth_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files or not depth_files:
    print("No valid images or depth maps found in the input directories.")
    exit()

# Match images and depth maps by name
paired_files = []
for img_file in image_files:
    depth_file = os.path.splitext(img_file)[0] + "_depth.png"
    if depth_file in depth_files:
        paired_files.append((img_file, depth_file))

if not paired_files:
    print("No matching RGB images and depth maps found.")
    exit()

print(f"Found {len(paired_files)} matching RGB and depth map pairs. Starting processing...")

# Process each matched pair
for i, (image_file, depth_file) in enumerate(paired_files, start=1):
    try:
        # Load RGB image
        img_path = os.path.join(input_path, image_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Unable to read image {img_path}. Skipping...")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        # Load corresponding depth map
        depth_path_full = os.path.join(depth_path, depth_file)
        depth_img = cv2.imread(depth_path_full, cv2.IMREAD_GRAYSCALE)

        if depth_img is None:
            print(f"Warning: Unable to read depth map {depth_path_full}. Skipping...")
            continue

        depth_img = cv2.resize(depth_img, (256, 256))
        depth_img = np.expand_dims(depth_img, axis=-1)

        # Combine RGB and depth into 4-channel input
        img_with_depth = np.concatenate((img, depth_img), axis=-1)

        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(img_with_depth).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5

        # Generate enhanced output
        with torch.no_grad():
            enhanced_image_tensor = generator(img_tensor)

        # Convert tensor to RGB image
        enhanced_image = denormalize_image(enhanced_image_tensor)

        # Post-process the enhanced image
        enhanced_image = apply_clahe(enhanced_image)

        # Save the output image
        output_file = os.path.join(output_path, os.path.splitext(image_file)[0] + "_enhanced.png")
        cv2.imwrite(output_file, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
        print(f"Processed and saved: {output_file}")
    except Exception as e:
        print(f"Error processing {image_file}: {e}")

print("Processing complete.")
