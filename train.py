import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from net.CycleGAN_generator import CycleGANGenerator, CycleGANDiscriminator
from utility.data import UnpairedDatasetWithDepth
from loss.vgg_loss import VGGPerceptualLoss
from loss.ssim_loss import ssim
from torch.cuda.amp import GradScaler, autocast

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
IMAGE_SIZE = 256  # Adjusted to match the dataset
# BATCH_SIZE = 4  # Adjusted for GPU memory availability
BATCH_SIZE = 1
LR = 0.0002
N_EPOCHS = 20
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 5
LAMBDA_VGG = 0.01
LAMBDA_SSIM = 0.1

# Paths
DOMAIN_A_PATH = r"C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\data\UDD"  # Underwater images
DOMAIN_B_PATH = None # Enhanced images
DOMAIN_DEPTH_PATH = r"C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\data\UDD-depth"  # Depth maps
SAVED_MODELS_PATH = './saved_models/UDD'

# Dataset and DataLoader
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
])

dataset = UnpairedDatasetWithDepth(
    domain_a_dir=DOMAIN_A_PATH,
    domain_b_dir=DOMAIN_B_PATH,
    depth_dir=DOMAIN_DEPTH_PATH,
    transform=transform
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Models
generator_a2b = CycleGANGenerator(3, 3).to(device)  # Generator: Domain A -> Domain B
generator_b2a = CycleGANGenerator(3, 3).to(device)  # Generator: Domain B -> Domain A
discriminator_a = CycleGANDiscriminator(3).to(device)  # Discriminator: Domain A
discriminator_b = CycleGANDiscriminator(3).to(device)  # Discriminator: Domain B

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
vgg_loss = VGGPerceptualLoss().to(device)  # VGG perceptual loss
scaler = GradScaler()  # Mixed Precision Training

# Training Loop
print("Starting training...")
for epoch in range(N_EPOCHS):
    for i, (real_a, depth_a, real_b) in enumerate(loader):
        # Load data and move to device
        real_a, depth_a, real_b = real_a.to(device), depth_a.to(device), real_b.to(device)

        # Concatenate depth with real_a for improved learning
        real_a_with_depth = torch.cat((real_a, depth_a), dim=1)  # Shape: [B, 4, H, W]

        # Train Generators
        with autocast():  # Mixed precision
            # Generate images
            fake_b = generator_a2b(real_a)  # A -> B
            reconstructed_a = generator_b2a(fake_b)  # B -> A
            fake_a = generator_b2a(real_b)  # B -> A
            reconstructed_b = generator_a2b(fake_a)  # A -> B

            # Generator losses
            loss_g_a2b = adversarial_loss(discriminator_b(fake_b), torch.ones_like(discriminator_b(fake_b)))
            loss_g_b2a = adversarial_loss(discriminator_a(fake_a), torch.ones_like(discriminator_a(fake_a)))
            loss_cycle_a = cycle_consistency_loss(reconstructed_a, real_a)
            loss_cycle_b = cycle_consistency_loss(reconstructed_b, real_b)
            loss_identity_a = identity_loss(generator_b2a(real_a), real_a)
            loss_identity_b = identity_loss(generator_a2b(real_b), real_b)
            loss_vgg = vgg_loss(fake_b, real_b)  # Perceptual Loss
            loss_ssim = 1 - ssim(fake_b, real_b)  # SSIM Loss

            # Total generator loss
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
            loss_d_a = (
                adversarial_loss(discriminator_a(real_a), torch.ones_like(discriminator_a(real_a))) +
                adversarial_loss(discriminator_a(fake_a.detach()), torch.zeros_like(discriminator_a(fake_a)))
            )
            loss_d_b = (
                adversarial_loss(discriminator_b(real_b), torch.ones_like(discriminator_b(real_b))) +
                adversarial_loss(discriminator_b(fake_b.detach()), torch.zeros_like(discriminator_b(fake_b)))
            )
            loss_d = (loss_d_a + loss_d_b) / 2

        optimizer_D.zero_grad()
        scaler.scale(loss_d).backward()
        scaler.step(optimizer_D)
        scaler.update()

        # Logging
        if i % 10 == 0:
            print(f"Epoch [{epoch}/{N_EPOCHS}], Batch [{i}/{len(loader)}], Loss G: {loss_g.item():.4f}, Loss D: {loss_d.item():.4f}")

    # Save models
    os.makedirs(SAVED_MODELS_PATH, exist_ok=True)
    torch.save(generator_a2b.state_dict(), os.path.join(SAVED_MODELS_PATH, "generator_a2b.pth"))
    torch.save(generator_b2a.state_dict(), os.path.join(SAVED_MODELS_PATH, "generator_b2a.pth"))
    print(f"Saved models at epoch {epoch}")

print("Training complete.")
