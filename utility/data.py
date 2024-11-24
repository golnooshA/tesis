from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import torch


class UnpairedDatasetWithDepth(Dataset):
    """
    Dataset class for unpaired data with depth maps.
    Supports loading images from two directories: one for Domain A (e.g., raw underwater images)
    and one for Domain B (e.g., enhanced images), as well as depth maps for Domain A.
    Handles missing or empty Domain B by creating placeholder images.
    """
    def __init__(self, domain_a_dir, domain_b_dir, depth_dir, transform=None, filenames=False):
        """
        Initialize the dataset.
        Args:
            domain_a_dir (str): Path to the directory containing Domain A images.
            domain_b_dir (str): Path to the directory containing Domain B images.
            depth_dir (str): Path to the directory containing depth maps for Domain A.
            transform (torchvision.transforms.Compose, optional): Transformations to apply to the images.
            filenames (bool): If True, return the file names along with the images.
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

    def __len__(self):
        """
        Return the length of the dataset.
        The dataset length is determined by the larger domain to allow full unpaired coverage.
        """
        return max(len(self.domain_a_files), len(self.depth_files), len(self.domain_b_files or []))

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


def _test():
    """
    Test the UnpairedDatasetWithDepth with sample directories.
    """
    # Use the absolute paths
    domain_a_dir = r"C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\data\UDD"
    domain_b_dir = None  # Simulate missing Domain B
    depth_dir = r"C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\data\UDD-depth"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor()
    ])

    dataset = UnpairedDatasetWithDepth(domain_a_dir, domain_b_dir, depth_dir, transform=transform, filenames=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for i, (a_images, depth_images, b_images, a_names, depth_names) in enumerate(loader):
        print(f"Batch {i}:")
        print(f"Domain A images: {a_images.shape}, Names: {a_names}")
        print(f"Depth images: {depth_images.shape}, Names: {depth_names}")
        print(f"Domain B images: {b_images.shape}")
        break


if __name__ == '__main__':
    _test()
