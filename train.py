import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from net.CycleGAN_generator import ResnetGenerator, CycleGANDiscriminator
from torch.cuda.amp import autocast, GradScaler

# Dataset definition
class UnderwaterDataset(Dataset):
    def __init__(self, input_folder, depth_folder, target_folder, transform=None):
        self.input_images = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jpg')])
        self.depth_maps = sorted([os.path.join(depth_folder, f) for f in os.listdir(depth_folder) if f.endswith('.png')])
        self.target_images = sorted([os.path.join(target_folder, f) for f in os.listdir(target_folder) if f.endswith('.jpg')])
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = self._read_image(self.input_images[idx])
        depth_map = self._read_image(self.depth_maps[idx], is_gray=True)
        target_image = self._read_image(self.target_images[idx])

        if self.transform:
            input_image = self.transform(input_image)
            depth_map = self.transform(depth_map)
            target_image = self.transform(target_image)

        combined_input = torch.cat([input_image, depth_map], dim=0)  # Combine RGB + Depth
        return combined_input, target_image

    def _read_image(self, path, is_gray=False):
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR)
        if not is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

# Paths
input_folder = './datasets/trainA'  # Original underwater images (domainA)
depth_folder = './DPT/output_withGAN'  # Final depth maps
target_folder = './datasets/trainB'  # Target enhanced images (domainB)
output_dir = './saved_models'  # Directory to save trained models

os.makedirs(output_dir, exist_ok=True)

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),
])
dataset = UnderwaterDataset(input_folder, depth_folder, target_folder, transform=transform)

if len(dataset) == 0:
    raise ValueError("No data found. Check your input, depth, and target folders.")

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Adjust batch size based on GPU memory

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_AB = ResnetGenerator(input_nc=4, output_nc=3).to(device)  # RGB + Depth -> RGB
G_BA = ResnetGenerator(input_nc=3, output_nc=4).to(device)  # RGB -> RGB + Depth
D_A = CycleGANDiscriminator(input_nc=3).to(device)  # Discriminator for domainA (RGB only)
D_B = CycleGANDiscriminator(input_nc=3).to(device)  # Discriminator for domainB (RGB only)

# Optimizers
optimizer_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss Functions
criterion_GAN = torch.nn.MSELoss()  # Adversarial loss
criterion_cycle = torch.nn.L1Loss()  # Cycle consistency loss
criterion_identity = torch.nn.L1Loss()  # Identity loss

# Mixed precision scaler
scaler = GradScaler()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, (input_data, target_data) in enumerate(dataloader):
        input_data, target_data = input_data.to(device), target_data.to(device)

        # Separate RGB and Depth from input
        real_a = input_data[:, :3, :, :]  # RGB
        depth_a = input_data[:, 3:, :, :]  # Depth

        # Update Generators
        optimizer_G.zero_grad()
        with autocast():  # Mixed precision without device_type
            fake_b = G_AB(input_data)  # A -> B
            fake_a = G_BA(target_data)[:, :3, :, :]  # B -> A (ensure RGB channels for loss)

            reconstructed_a = G_BA(fake_b)[:, :3, :, :]  # Reconstruct A (RGB only)
            input_for_reconstruct_b = torch.cat([fake_a, depth_a], dim=1) if fake_a.shape[1] == 3 else fake_a
            reconstructed_b = G_AB(input_for_reconstruct_b)  # Reconstruct B

            # Generator losses
            loss_GAN_AB = criterion_GAN(D_B(fake_b), torch.ones_like(D_B(fake_b)))
            loss_GAN_BA = criterion_GAN(D_A(fake_a), torch.ones_like(D_A(fake_a)))
            loss_cycle_A = criterion_cycle(reconstructed_a, real_a)  # Ensure cycle matches RGB
            loss_cycle_B = criterion_cycle(reconstructed_b, target_data)
            loss_identity_A = criterion_identity(G_AB(input_data), target_data)
            loss_identity_B = criterion_identity(G_BA(target_data)[:, :3, :, :], real_a)  # Identity only RGB
            loss_G = loss_GAN_AB + loss_GAN_BA + 10.0 * (loss_cycle_A + loss_cycle_B) + 5.0 * (loss_identity_A + loss_identity_B)

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        scaler.update()

        # Update Discriminators
        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad()
        with autocast():  # Mixed precision without device_type
            loss_D_A = criterion_GAN(D_A(real_a), torch.ones_like(D_A(real_a))) + \
                       criterion_GAN(D_A(fake_a.detach()), torch.zeros_like(D_A(fake_a)))
            loss_D_B = criterion_GAN(D_B(target_data), torch.ones_like(D_B(target_data))) + \
                       criterion_GAN(D_B(fake_b.detach()), torch.zeros_like(D_B(fake_b)))

        scaler.scale(loss_D_A).backward()
        scaler.scale(loss_D_B).backward()
        scaler.step(optimizer_D_A)
        scaler.step(optimizer_D_B)
        scaler.update()

        # Logging
        if i % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], "
                  f"G Loss: {loss_G.item():.4f}, D Loss A: {loss_D_A.item():.4f}, D Loss B: {loss_D_B.item():.4f}")

    # Save model checkpoints every epoch
    torch.save(G_AB.state_dict(), os.path.join(output_dir, f'G_AB_epoch_{epoch+1}.pth'))
    torch.save(G_BA.state_dict(), os.path.join(output_dir, f'G_BA_epoch_{epoch+1}.pth'))
    torch.save(D_A.state_dict(), os.path.join(output_dir, f'D_A_epoch_{epoch+1}.pth'))
    torch.save(D_B.state_dict(), os.path.join(output_dir, f'D_B_epoch_{epoch+1}.pth'))
    print(f"Models saved for epoch {epoch+1}")

print("Training completed!")
