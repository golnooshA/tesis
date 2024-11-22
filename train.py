# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision.utils import save_image
# import numpy as np
# import cv2
# import random
# from net.Ushape_Trans import Generator, Discriminator
# from net.utils import batch_PSNR, VGG19_PercepLoss
# from loss.LAB import lab_Loss
# from loss.LCH import lch_Loss

# # Environment setup
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Hyperparameters
# IMAGE_SIZE = 256
# LR = 0.0005
# BATCH_SIZE = 1
# N_EPOCHS = 20
# CHECKPOINT_INTERVAL = 1
# SAMPLE_INTERVAL = 1000
# LAMBDA_PIXEL = 0.1
# LAMBDA_LAB = 0.001
# LAMBDA_LCH = 1
# LAMBDA_CON = 100
# LAMBDA_SSIM = 100

# # Paths
# INPUT_PATH = './data/input/'
# DEPTH_PATH = './data/depth/'
# GT_PATH = './data/GT/'
# TEST_INPUT_PATH = './test/input/'
# TEST_DEPTH_PATH = './test/depth/'
# TEST_GT_PATH = './test/GT/'

# # Loss functions
# criterion_GAN = nn.MSELoss().to(device)
# criterion_pixelwise = nn.MSELoss().to(device)
# L_vgg = VGG19_PercepLoss().to(device)
# L_lab = lab_Loss().to(device)
# L_lch = lch_Loss().to(device)

# # Data Loading Functions
# def load_dataset(input_path, depth_path, gt_path):
#     x_data = []
#     y_data = []
    
#     input_files = sorted([f for f in os.listdir(input_path) if f.endswith('.png')])
#     depth_files = sorted([f for f in os.listdir(depth_path) if f.endswith('.png')])
#     gt_files = sorted([f for f in os.listdir(gt_path) if f.endswith('.png')])

#     print(f"Input files: {len(input_files)}")
#     print(f"Depth files: {len(depth_files)}")
#     print(f"GT files: {len(gt_files)}")

#     # Find common files by base name
#     common_files = []
#     for f in input_files:
#         base_name = f.replace('.png', '')  # Remove suffix
#         if base_name + '.png' in depth_files and base_name + '.png' in gt_files:
#             common_files.append(f)

#     print(f"Matching files: {len(common_files)}")

#     if len(common_files) == 0:
#         print("Mismatched or missing files:")
#         print("Files in INPUT but not in DEPTH or GT:")
#         for f in input_files:
#             if f.replace('.png', '') not in [d.replace('.png', '') for d in depth_files] or f.replace('.png', '') not in [g.replace('.png', '') for g in gt_files]:
#                 print(f" - {f}")
#         print("Files in DEPTH but not in INPUT or GT:")
#         for f in depth_files:
#             if f.replace('.png', '') not in [i.replace('.png', '') for i in input_files] or f.replace('.png', '') not in [g.replace('.png', '') for g in gt_files]:
#                 print(f" - {f}")
#         print("Files in GT but not in INPUT or DEPTH:")
#         for f in gt_files:
#             if f.replace('.png', '') not in [i.replace('.png', '') for i in input_files] or f.replace('.png', '') not in [d.replace('.png', '') for d in depth_files]:
#                 print(f" - {f}")
#         raise ValueError("No valid image pairs found. Ensure files are correctly aligned in the dataset directories.")

#     # Process and load matching files
#     for filename in common_files:
#         base_name = filename.replace('.png', '')  # Remove the suffix to match all three
#         img_path = os.path.join(input_path, f"{base_name}.png")
#         depth_path_file = os.path.join(depth_path, f"{base_name}.png")
#         gt_path_file = os.path.join(gt_path, f"{base_name}.png")

#         # Load input, depth, and GT
#         img = cv2.imread(img_path)
#         depth = cv2.imread(depth_path_file, cv2.IMREAD_GRAYSCALE)
#         gt_img = cv2.imread(gt_path_file)

#         if img is None or depth is None or gt_img is None:
#             print(f"Skipping {filename}: One or more files not found or corrupted.")
#             continue

#         # Process input
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#         depth = cv2.resize(depth, (IMAGE_SIZE, IMAGE_SIZE)).reshape((IMAGE_SIZE, IMAGE_SIZE, 1))
#         depth = depth / 255.0  # Normalize depth map to 0-1

#         # Process GT
#         gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
#         gt_img = cv2.resize(gt_img, (IMAGE_SIZE, IMAGE_SIZE))

#         x_data.append(np.concatenate((img, depth), axis=2))
#         y_data.append(gt_img)

#     if len(x_data) == 0 or len(y_data) == 0:
#         raise ValueError("No valid data loaded. Ensure that the files are correctly aligned and processed.")

#     x_data = np.array(x_data).astype('float32') / 255.0
#     y_data = np.array(y_data).astype('float32') / 255.0
#     return x_data, y_data


# def preprocess_images(images):
#     if len(images.shape) != 4:
#         raise ValueError(f"Expected 4D tensor, but got {len(images.shape)}D tensor.")
#     return torch.from_numpy(images).permute(0, 3, 1, 2).to(device)


# def split_scales(img):
#     return [F.interpolate(img, scale_factor=sf) for sf in [0.125, 0.25, 0.5, 1]]

# # Load Datasets
# print("Loading training dataset...")
# x_train, y_train = load_dataset(INPUT_PATH, DEPTH_PATH, GT_PATH)
# x_train = preprocess_images(x_train)
# y_train = preprocess_images(y_train)
# print(f"x_train shape: {x_train.shape}")
# print(f"y_train shape: {y_train.shape}")

# print("Loading testing dataset...")
# x_test, y_test = load_dataset(TEST_INPUT_PATH, TEST_DEPTH_PATH, TEST_GT_PATH)
# x_test = preprocess_images(x_test)
# y_test = preprocess_images(y_test)
# print(f"x_test shape: {x_test.shape}")
# print(f"y_test shape: {y_test.shape}")

# # DataLoader
# dataset = TensorDataset(x_train, y_train)
# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# # Initialize models
# generator = Generator().to(device)
# discriminator = Discriminator().to(device)

# # Load pre-trained models if available
# if os.path.exists("saved_models/G/generator.pth"):
#     generator.load_state_dict(torch.load("saved_models/G/generator.pth"))
#     print("Loaded pre-trained generator.")
# else:
#     print("No pre-trained generator found. Starting from scratch.")

# # Optimizers
# optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
# optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# # Training Loop
# print("Starting training...")
# for epoch in range(N_EPOCHS):
#     for i, batch in enumerate(loader):
#         real_A, real_B = batch
#         real_A, real_B = real_A.to(device), real_B.to(device)

#         # Generate fake_B
#         fake_B = generator(real_A)

#         # Losses
#         loss_pixel = criterion_pixelwise(fake_B[-1], real_B)
#         loss_lab = L_lab(fake_B[-1], real_B)
#         loss_lch = L_lch(fake_B[-1], real_B)
#         loss_G = LAMBDA_PIXEL * loss_pixel + LAMBDA_LAB * loss_lab + LAMBDA_LCH * loss_lch

#         # Train Generator
#         optimizer_G.zero_grad()
#         loss_G.backward()
#         optimizer_G.step()

#         # Log progress
#         if i % SAMPLE_INTERVAL == 0:
#             print(f"Epoch [{epoch}/{N_EPOCHS}] Batch [{i}/{len(loader)}]: Loss G: {loss_G.item():.4f}")

#     # Save model checkpoints
#     if epoch % CHECKPOINT_INTERVAL == 0:
#         os.makedirs("saved_models/G", exist_ok=True)
#         torch.save(generator.state_dict(), f"saved_models/G/generator_{epoch}.pth")
#         print(f"Saved generator at epoch {epoch}")

# print("Training complete.")


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from net.CycleGAN_generator import CycleGANGenerator, CycleGANDiscriminator
from utility.data import UnpairedDataset
from loss.vgg_loss import VGGPerceptualLoss
from loss.ssim_loss import ssim
from torch.cuda.amp import GradScaler, autocast

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
IMAGE_SIZE = 128  # Reduced image size for memory efficiency
BATCH_SIZE = 1  # Reduced batch size
LR = 0.0002
N_EPOCHS = 20
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 5
LAMBDA_VGG = 0.01
LAMBDA_SSIM = 0.1

# Paths
DOMAIN_A_PATH = r'C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\data\EUVP\domianA' # Underwater images
DOMAIN_B_PATH = r'C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\data\EUVP\domianB'  # Enhanced images
SAVED_MODELS_PATH = './saved_models/'

# Dataset and DataLoader
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
])

dataset = UnpairedDataset(DOMAIN_A_PATH, DOMAIN_B_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Models
generator_a2b = CycleGANGenerator(3, 3).to(device)
generator_b2a = CycleGANGenerator(3, 3).to(device)
discriminator_a = CycleGANDiscriminator(3).to(device)
discriminator_b = CycleGANDiscriminator(3).to(device)

# Optimizers
optimizer_G = optim.Adam(
    list(generator_a2b.parameters()) + list(generator_b2a.parameters()), lr=LR, betas=(0.5, 0.999)
)
optimizer_D = optim.Adam(
    list(discriminator_a.parameters()) + list(discriminator_b.parameters()), lr=LR, betas=(0.5, 0.999)
)

# Loss functions
adversarial_loss = nn.MSELoss().to(device)
cycle_consistency_loss = nn.L1Loss().to(device)
identity_loss = nn.L1Loss().to(device)
vgg_loss = VGGPerceptualLoss().to(device)  # Added VGG perceptual loss

# Mixed Precision Training
scaler = GradScaler()

# Training Loop
print("Starting training...")
for epoch in range(N_EPOCHS):
    for i, (real_a, real_b) in enumerate(loader):
        real_a, real_b = real_a.to(device), real_b.to(device)

        # Train Generators
        with autocast():  # Mixed precision
            fake_b = generator_a2b(real_a)
            reconstructed_a = generator_b2a(fake_b)
            fake_a = generator_b2a(real_b)
            reconstructed_b = generator_a2b(fake_a)

            # Adversarial Loss
            loss_g_a2b = adversarial_loss(discriminator_b(fake_b), torch.ones_like(discriminator_b(fake_b)))
            loss_g_b2a = adversarial_loss(discriminator_a(fake_a), torch.ones_like(discriminator_a(fake_a)))

            # Cycle Consistency Loss
            loss_cycle_a = cycle_consistency_loss(reconstructed_a, real_a)
            loss_cycle_b = cycle_consistency_loss(reconstructed_b, real_b)

            # Identity Loss
            loss_identity_a = identity_loss(generator_b2a(real_a), real_a)
            loss_identity_b = identity_loss(generator_a2b(real_b), real_b)

            # Perceptual Loss (VGG)
            loss_vgg = vgg_loss(fake_b, real_b)

            # SSIM Loss
            loss_ssim = 1 - ssim(fake_b, real_b)

            # Total Generator Loss
            loss_g = (
                loss_g_a2b + loss_g_b2a +
                LAMBDA_CYCLE * (loss_cycle_a + loss_cycle_b) +
                LAMBDA_IDENTITY * (loss_identity_a + loss_identity_b) +
                LAMBDA_VGG * loss_vgg +
                LAMBDA_SSIM * loss_ssim
            )

        optimizer_G.zero_grad()
        scaler.scale(loss_g).backward()
        scaler.step(optimizer_G)
        scaler.update()

        # Train Discriminators
        with autocast():  # Mixed precision
            loss_d_a = adversarial_loss(discriminator_a(real_a), torch.ones_like(discriminator_a(real_a))) + \
                       adversarial_loss(discriminator_a(fake_a.detach()), torch.zeros_like(discriminator_a(fake_a)))

            loss_d_b = adversarial_loss(discriminator_b(real_b), torch.ones_like(discriminator_b(real_b))) + \
                       adversarial_loss(discriminator_b(fake_b.detach()), torch.zeros_like(discriminator_b(fake_b)))

            loss_d = (loss_d_a + loss_d_b) / 2

        optimizer_D.zero_grad()
        scaler.scale(loss_d).backward()
        scaler.step(optimizer_D)
        scaler.update()

        # Logging
        if i % 50 == 0:
            print(f"Epoch [{epoch}/{N_EPOCHS}], Batch [{i}/{len(loader)}], Loss G: {loss_g.item():.4f}, Loss D: {loss_d.item():.4f}")

    # Save models
    os.makedirs(SAVED_MODELS_PATH, exist_ok=True)
    torch.save(generator_a2b.state_dict(), os.path.join(SAVED_MODELS_PATH, "generator_a2b.pth"))
    torch.save(generator_b2a.state_dict(), os.path.join(SAVED_MODELS_PATH, "generator_b2a.pth"))
    print(f"Saved models at epoch {epoch}")

print("Training complete.")
