"""Compute depth maps for images in the input folder."""
import os
import glob
import torch
import cv2
import argparse
from torchvision.transforms import Compose
from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
import util.io

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="timm.models._factory")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


def run(input_path, output_path, model_path, model_type="dpt_hybrid_nyu", optimize=True, absolute_depth=False):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): Path to input folder
        output_path (str): Path to output folder
        model_path (str): Path to saved model
        model_type (str): Type of model to use (only NYU is supported here)
        optimize (bool): Optimize for GPU performance
        absolute_depth (bool): Scale depth predictions for absolute depth
    """
    print("Initializing...")

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the NYU model
    if model_type == "dpt_hybrid_nyu":
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        raise ValueError(
            f"Model type '{model_type}' not implemented. Use: [dpt_hybrid_nyu]"
        )

    # Define transformations
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    # Prepare the model
    model.eval()
    if optimize and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        model = model.half()
    model.to(device)

    # Get input images (.png and .jpg)
    img_names = glob.glob(os.path.join(input_path, "*.png"))
    if not img_names:
        print(f"No valid .png images found in {input_path}. Please check the folder path and file format.")
        return
    num_images = len(img_names)

    # Create output folder
    os.makedirs(output_path, exist_ok=True)

    print("Start processing...")
    for ind, img_name in enumerate(img_names):
        print(f"  Processing {img_name} ({ind + 1}/{num_images})")

        # Read and preprocess image
        img = util.io.read_image(img_name)
        if img is None:
            print(f"Warning: Unable to read {img_name}. Skipping...")
            continue

        img_input = transform({"image": img})["image"]

        # Compute depth map
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if optimize and device.type == "cuda":
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            # Scale predictions for NYU if absolute depth is enabled
            if absolute_depth:
                prediction *= 1000.0

        # Save depth map
        # Save depth map
        base_name, ext = os.path.splitext(os.path.basename(img_name))
        if ext.lower() != ".png":
            filename = os.path.join(output_path, f"{base_name}_depth.png")
        else:
            filename = os.path.join(output_path, f"{base_name}_depth")

        util.io.write_depth(filename, prediction, bits=2)


    print("Processing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="input/NYU", help="Folder with input images"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="output_fine_tune",
        help="Folder for output depth maps",
    )

    parser.add_argument(
        "-m", "--model_weights", default=None, help="Path to model weights"
    )

    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid_nyu",
        help="Model type [dpt_hybrid_nyu]",
    )

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.add_argument(
        "--absolute_depth", dest="absolute_depth", action="store_true",
        help="Scale depth predictions for absolute depth"
    )

    parser.set_defaults(optimize=True)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()

    # Define default model weights based on model type
    default_models = {
        "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # Set PyTorch options for performance
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Run the depth map computation
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.model_type,
        args.optimize,
        args.absolute_depth,
    )
