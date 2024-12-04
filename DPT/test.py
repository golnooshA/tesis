import torch
from PIL import Image
import sys
import os
import torchvision.transforms as transforms

# Update the paths for your environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../net')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utility')))

from CycleGAN_generator import ResnetGenerator 

# Paths
weights_path = "weights/latest_net_G_A.pth"  # Path to the CycleGAN weights
input_image_path = r'C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\datasets\trainA\nm_578up.jpg'  # Path to an input image
output_image_path = r'C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\DPT\test_output_net.jpg'  # Path to save the output image

def fix_state_dict_keys(state_dict, add_prefix=None, remove_prefix=None):
   
    fixed_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if add_prefix and not key.startswith(add_prefix):
            new_key = add_prefix + key
        if remove_prefix and key.startswith(remove_prefix):
            new_key = key[len(remove_prefix):]
        fixed_state_dict[new_key] = value
    return fixed_state_dict

# Instantiate the generator
cyclegan_gen = ResnetGenerator(input_nc=3, output_nc=3, n_blocks=9)  # Match the training setup

# Load and fix the weights
raw_state_dict = torch.load(weights_path, map_location="cpu")
print("Original state_dict keys:", raw_state_dict.keys())

# Add or remove prefixes to match the model's key structure
fixed_state_dict = fix_state_dict_keys(raw_state_dict, add_prefix="model.")  # Add "model." prefix if necessary

# Load the fixed state_dict into the generator
cyclegan_gen.load_state_dict(fixed_state_dict, strict=False)  # Allow partial loading in case of minor key mismatches
cyclegan_gen.eval()  # Set the model to evaluation mode

# Preprocessing Transformations
transform = transforms.Compose([
    transforms.Resize(256),  # Resize to 256x256
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load and preprocess the input image
img = Image.open(input_image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Generate the output
with torch.no_grad():
    output_tensor = cyclegan_gen(img_tensor)

# Post-process the output
output_image = (output_tensor.squeeze().permute(1, 2, 0).cpu().numpy() + 1) * 127.5  # Convert to [0, 255]
output_image = output_image.clip(0, 255).astype("uint8")  # Ensure values are within valid range

# Save the output image
output_image_pil = Image.fromarray(output_image)
output_image_pil.save(output_image_path)
print(f"Saved output image to {output_image_path}")
