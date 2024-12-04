import os
import sys
import glob
import torch
import argparse
import warnings
from torchvision.transforms import Compose
from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
import util.io
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../net')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utility')))

from CycleGAN_generator import ResnetGenerator

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)


def fix_state_dict_keys(state_dict, add_prefix=None, remove_prefix=None):
    """Fix state_dict keys by adding or removing prefixes."""
    fixed_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if add_prefix and not key.startswith(add_prefix):
            new_key = add_prefix + key
        if remove_prefix and key.startswith(remove_prefix):
            new_key = key[len(remove_prefix):]
        fixed_state_dict[new_key] = value
    return fixed_state_dict


def load_models(model_path, model_type, use_cyclegan, device):
    """Load depth estimation and CycleGAN models."""
    print("Loading models...")

    # Load CycleGAN generator for preprocessing
    cyclegan_gen = None
    if use_cyclegan:
        print("Initializing CycleGAN Generator...")
        cyclegan_gen = ResnetGenerator(input_nc=3, output_nc=3, n_blocks=9)
        raw_state_dict = torch.load("weights/latest_net_G_A.pth", map_location="cpu")
        fixed_state_dict = fix_state_dict_keys(raw_state_dict, add_prefix="model.")
        cyclegan_gen.load_state_dict(fixed_state_dict, strict=False)
        cyclegan_gen.to(device).eval()

    # Load depth estimation model
    print(f"Loading Depth Estimation Model: {model_type}")
    if model_type == "dpt_large":
        net_w = net_h = 384
        model = DPTDepthModel(path=model_path, backbone="vitl16_384", non_negative=True)
    elif model_type == "dpt_hybrid":
        net_w = net_h = 384
        model = DPTDepthModel(path=model_path, backbone="vitb_rn50_384", non_negative=True)
    elif model_type == "midas_v21":
        net_w = net_h = 384
        model = MidasNet_large(model_path, non_negative=True)
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")

    model.to(device).eval()
    return cyclegan_gen, model, net_w, net_h


def preprocess_image(image, cyclegan_gen, transform, use_cyclegan, device, output_path, img_name):
    """Preprocess the input image with optional CycleGAN."""
    original_image = image.copy()

    if use_cyclegan:
        print("Applying CycleGAN preprocessing...")
        # Normalize input image to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        image = (image - 0.5) / 0.5  # Scale to [-1, 1]

        # Pass through CycleGAN
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            try:
                img_tensor = cyclegan_gen(img_tensor)
            except RuntimeError as e:
                print(f"Error in CycleGAN preprocessing for image {img_name}: {e}")
                torch.cuda.empty_cache()  # Clear GPU memory
                return None, None

        # Convert back to numpy
        image = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = (image * 0.5 + 0.5)  # Scale back to [0, 1]

        # Save preprocessed image for debugging
        preprocessed_path = os.path.join(output_path, f"{img_name}_cyclegan.png")
        cv2.imwrite(preprocessed_path, (image * 255).astype(np.uint8))
        print(f"Saved CycleGAN-preprocessed image to {preprocessed_path}")

    # Transform image for depth estimation
    img_input = transform({"image": image})["image"]
    return img_input, original_image


def infer_depth(model, img_input, device):
    """Perform depth estimation on the input image."""
    print("Predicting depth...")
    img_tensor = torch.from_numpy(img_input).to(device).unsqueeze(0)

    with torch.no_grad():
        prediction = model.forward(img_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_input.shape[1:],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    return prediction


def save_output(image_name, output_path, depth_map, original_image=None):
    """Save the depth map and optionally the original image to the output directory."""
    os.makedirs(output_path, exist_ok=True)

    # Save the depth map
    depth_file = os.path.join(output_path, os.path.splitext(os.path.basename(image_name))[0] + "_depth")
    util.io.write_depth(depth_file, depth_map, bits=2)
    print(f"Saved depth map to {depth_file}")

    # Save the original image if provided
    if original_image is not None:
        original_file = os.path.join(output_path, os.path.basename(image_name))
        cv2.imwrite(original_file, original_image)
        print(f"Saved original image to {original_file}")


def run(input_path, output_path, model_path, model_type="dpt_hybrid", use_cyclegan=False):
    """Main pipeline for underwater image enhancement and depth estimation."""
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    print(f"Using device: {device}")

    # Load models
    cyclegan_gen, model, net_w, net_h = load_models(model_path, model_type, use_cyclegan, device)

    # Image transformation pipeline
    transform = Compose([
        Resize(net_w, net_h, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
    ])

    # Collect input images
    img_files = glob.glob(os.path.join(input_path, "*"))
    os.makedirs(output_path, exist_ok=True)

    if not img_files:
        print(f"No images found in {input_path}. Exiting.")
        return

    print(f"Found {len(img_files)} images. Processing...")
    for img_file in img_files:
        if os.path.isdir(img_file):
            continue

        print(f"Processing image: {img_file}")
        img = util.io.read_image(img_file)

        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # Preprocess image
        img_name = os.path.splitext(os.path.basename(img_file))[0]
        img_input, original_image = preprocess_image(
            img, cyclegan_gen, transform, use_cyclegan, device, output_path, img_name
        )

        # Infer depth
        depth_map = infer_depth(model, img_input, device)

        # Save output
        save_output(img_file, output_path, depth_map, original_image)

    print("All images processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Depth Estimation with Optional CycleGAN Preprocessing.")

    parser.add_argument("-i", "--input_path", default=r'C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\datasets\trainA', help="Folder with input images")
    parser.add_argument("-o", "--output_path", default="output_withGAN", help="Folder for output images")
    parser.add_argument("-m", "--model_weights", required=True, help="Path to model weights.")
    parser.add_argument("-t", "--model_type", default="dpt_hybrid", help="Model type [dpt_large|dpt_hybrid|midas_v21].")
    parser.add_argument("--use_cyclegan", action="store_true", help="Enable CycleGAN preprocessing.")
    parser.add_argument("--device", default="cuda", help="Device to run the script on [cuda|cpu]. Default is cuda.")

    args = parser.parse_args()

    default_weights = {
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "midas_v21": "weights/midas_v21-f6b98070.pt",
    }

    # Handle default weights if no specific model weights are provided
    model_path = args.model_weights if args.model_weights else default_weights.get(args.model_type, None)
    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found for {args.model_type}. Expected path: {model_path}")

    run(
        args.input_path,
        args.output_path,
        model_path,
        args.model_type,
        args.use_cyclegan,
    )
