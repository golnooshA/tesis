from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
import cv2
import numpy as np


class UnpairedDatasetWithDepth(Dataset):
    """
    Dataset class for unpaired data with depth maps.
    Supports loading images from two directories: one for Domain A (e.g., raw underwater images)
    and one for Domain B (e.g., enhanced images), as well as depth maps for Domain A.
    """
    def __init__(self, domain_a_dir, domain_b_dir, depth_dir, transform=None, filenames=False, preprocess_depth=False):
        """
        Initialize the dataset.
        Args:
            domain_a_dir (str): Path to the directory containing Domain A images.
            domain_b_dir (str): Path to the directory containing Domain B images.
            depth_dir (str): Path to the directory containing depth maps for Domain A.
            transform (torchvision.transforms.Compose, optional): Transformations to apply to the images.
            filenames (bool): If True, return the file names along with the images.
            preprocess_depth (bool): If True, apply preprocessing to depth maps.
        """
        self.domain_a_files = sorted([os.path.join(domain_a_dir, f) for f in os.listdir(domain_a_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        self.depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

        if len(self.domain_a_files) == 0 or len(self.depth_files) == 0:
            raise ValueError("Domain A or depth map directories are empty or invalid. Please check the paths.")

        # If Domain B is provided, load files; otherwise, set as None
        if domain_b_dir and os.path.isdir(domain_b_dir):
            self.domain_b_files = sorted([os.path.join(domain_b_dir, f) for f in os.listdir(domain_b_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        else:
            self.domain_b_files = None

        self.transform = transform if transform else transforms.Compose([transforms.ToTensor()])
        self.filenames = filenames
        self.preprocess_depth = preprocess_depth

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return max(len(self.domain_a_files), len(self.depth_files), len(self.domain_b_files or []))

    def preprocess_depth_map(self, depth):
        """
        Preprocess the depth map (e.g., normalize, smooth).
        Args:
            depth (PIL.Image): Depth map image.
        Returns:
            PIL.Image: Preprocessed depth map.
        """
        # Convert depth to numpy for preprocessing
        depth = np.array(depth)
        
        # Normalize to [0, 1]
        depth = depth / 255.0

        # Apply Gaussian smoothing to reduce noise
        depth = cv2.GaussianBlur(depth, (5, 5), 0)

        # Convert back to PIL image
        depth = Image.fromarray((depth * 255).astype(np.uint8))
        return depth

    def __getitem__(self, index):
        """
        Get an item from the dataset.
        Args:
            index (int): Index of the item to retrieve.
        Returns:
            tuple: Transformed images from Domain A and Domain B, depth map for Domain A, optionally with filenames.
        """
        # Cycle through the smaller datasets to match the largest
        a_path = self.domain_a_files[index % len(self.domain_a_files)]
        depth_path = self.depth_files[index % len(self.depth_files)]

        # Load Domain A and depth map
        a_image = Image.open(a_path).convert('RGB')
        depth_image = Image.open(depth_path).convert('L')  # Depth map as grayscale

        # Preprocess depth if required
        if self.preprocess_depth:
            depth_image = self.preprocess_depth_map(depth_image)

        if self.domain_b_files:
            b_path = self.domain_b_files[index % len(self.domain_b_files)]
            b_image = Image.open(b_path).convert('RGB')
        else:
            # Create a placeholder black image for Domain B if not provided
            b_image = Image.new('RGB', a_image.size, (0, 0, 0))

        # Apply transformations
        if self.transform:
            a_image = self.transform(a_image)
            depth_image = self.transform(depth_image)
            b_image = self.transform(b_image)

        # Return data with optional filenames
        if self.filenames:
            return a_image, depth_image, b_image, os.path.basename(a_path), os.path.basename(depth_path)
        else:
            return a_image, depth_image, b_image


# Test the dataset class
if __name__ == '__main__':
    domain_a_dir = r'C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\datasets\EUVP\Paired\domainA'
    domain_b_dir = r'C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\datasets\EUVP\Paired\domainB'
    depth_dir = r'C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\DPT\output_monodepth\Paired'

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = UnpairedDatasetWithDepth(
        domain_a_dir=domain_a_dir,
        domain_b_dir=domain_b_dir,
        depth_dir=depth_dir,
        transform=transform,
        preprocess_depth=True
    )

    for i in range(3):
        a_image, depth_image, b_image = dataset[i]
        print(f"Sample {i}: A image shape: {a_image.shape}, Depth image shape: {depth_image.shape}, B image shape: {b_image.shape}")
