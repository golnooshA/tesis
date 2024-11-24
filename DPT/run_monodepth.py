"""Compute depth maps and train CycleGAN for underwater image enhancement."""
import os
import sys
import glob
import torch
import cv2
import argparse
import warnings
from torchvision.transforms import Compose
from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
import util.io

# Add the 'net' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../net')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utility')))

from CycleGAN_generator import CycleGANGenerator, CycleGANDiscriminator
from data import UnpairedDatasetWithDepth
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

warnings.filterwarnings("ignore", category=UserWarning, module="timm.models._factory")


# Define CycleGAN losses
def cycle_consistency_loss(real, reconstructed):
    return torch.mean(torch.abs(real - reconstructed))


def adversarial_loss(pred, target_is_real):
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return torch.nn.functional.mse_loss(pred, target)


def get_model_weights(model_type):
    """Return the default weights path for the specified model type."""
    weights = {
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_ade20k": "weights/dpt_hybrid-ade20k-53898607.pt",
        "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_large_ade20k": "weights/dpt_large-ade20k-b12dca68.pt",
        "midas_v21": "weights/midas_v21-f6b98070.pt",
    }
    if model_type not in weights:
        raise ValueError(f"Unsupported model type: {model_type}")
    return weights[model_type]


def run(
    input_path,
    output_path,
    model_path,
    model_type="dpt_hybrid",
    optimize=True,
    cycle_gan=False,
    epochs=10,
    batch_size=4,
    lr=0.0002,
):
    """Run MonoDepthNN or train CycleGAN for underwater enhancement."""
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # Depth estimation model setup
    if model_type in ["dpt_hybrid", "dpt_large"]:
        net_w = net_h = 384
        depth_model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384" if model_type == "dpt_hybrid" else "vitl16_384",
            non_negative=True,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":
        net_w = net_h = 384
        depth_model = MidasNet_large(model_path, non_negative=True)
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        raise ValueError("Unsupported model type")

    transform = Compose(
        [
            Resize(net_w, net_h, resize_target=None, keep_aspect_ratio=True),
            normalization,
            PrepareForNet(),
        ]
    )

    depth_model.eval()
    depth_model.to(device)

    # CycleGAN setup
    if cycle_gan:
        generator_a2b = CycleGANGenerator(3, 3).to(device)
        generator_b2a = CycleGANGenerator(3, 3).to(device)
        discriminator_a = CycleGANDiscriminator(3).to(device)
        discriminator_b = CycleGANDiscriminator(3).to(device)

        gen_optimizer = torch.optim.Adam(
            list(generator_a2b.parameters()) + list(generator_b2a.parameters()), lr=lr, betas=(0.5, 0.999)
        )
        disc_optimizer = torch.optim.Adam(
            list(discriminator_a.parameters()) + list(discriminator_b.parameters()), lr=lr, betas=(0.5, 0.999)
        )

        # Dataset and DataLoader
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        dataset = UnpairedDatasetWithDepth(input_path, output_path, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        print("Start CycleGAN training")
        for epoch in range(epochs):
            for i, (real_a, real_b) in enumerate(dataloader):
                real_a, real_b = real_a.to(device), real_b.to(device)

                # Forward pass
                fake_b = generator_a2b(real_a)
                reconstructed_a = generator_b2a(fake_b)
                fake_a = generator_b2a(real_b)
                reconstructed_b = generator_a2b(fake_a)

                # Discriminators
                disc_loss_a = adversarial_loss(discriminator_a(fake_a), False) + adversarial_loss(discriminator_a(real_a), True)
                disc_loss_b = adversarial_loss(discriminator_b(fake_b), False) + adversarial_loss(discriminator_b(real_b), True)
                disc_loss = (disc_loss_a + disc_loss_b) / 2

                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()

                # Generators
                gen_loss_a2b = adversarial_loss(discriminator_b(fake_b), True)
                gen_loss_b2a = adversarial_loss(discriminator_a(fake_a), True)
                cycle_loss = cycle_consistency_loss(real_a, reconstructed_a) + cycle_consistency_loss(real_b, reconstructed_b)
                gen_loss = gen_loss_a2b + gen_loss_b2a + cycle_loss

                gen_optimizer.zero_grad()
                gen_loss.backward()
                gen_optimizer.step()

                # Logging
                if i % 10 == 0:
                    print(f"Epoch {epoch}/{epochs}, Step {i}, Disc Loss: {disc_loss.item():.4f}, Gen Loss: {gen_loss.item():.4f}")

        print("CycleGAN training complete.")

    else:
        # Depth inference
        print("Start depth estimation")
        img_names = glob.glob(os.path.join(input_path, "*"))
        os.makedirs(output_path, exist_ok=True)

        for ind, img_name in enumerate(img_names):
            if os.path.isdir(img_name):
                continue

            print(f"Processing {img_name} ({ind + 1}/{len(img_names)})")
            img = cv2.imread(img_name)
            img_input = transform({"image": img})["image"]

            with torch.no_grad():
                sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
                prediction = depth_model.forward(sample)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()

            output_name = os.path.join(output_path, os.path.splitext(os.path.basename(img_name))[0])
            util.io.write_depth(output_name, prediction, bits=2)

        print("Depth estimation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_path", default="input/SUIM", help="folder with input images")
    parser.add_argument("-o", "--output_path", default="output_monodepth/SUIM", help="folder for output images")
    parser.add_argument("-m", "--model_weights", default=None, help="path to model weights")
    parser.add_argument("-t", "--model_type", default="dpt_hybrid", help="model type [dpt_hybrid|midas_v21|dpt_large]")
    parser.add_argument("--cycle_gan", action="store_true", help="Enable CycleGAN training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizers")
    parser.set_defaults(cycle_gan=False)

    args = parser.parse_args()

    model_path = args.model_weights or get_model_weights(args.model_type)

    run(
        args.input_path,
        args.output_path,
        model_path,
        args.model_type,
        cycle_gan=args.cycle_gan,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
