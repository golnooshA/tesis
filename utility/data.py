from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import torch


class UnpairedDataset(Dataset):
    """
    Dataset class for unpaired data.
    Supports loading images from two directories: one for Domain A (e.g., raw underwater images)
    and one for Domain B (e.g., enhanced or clear images).
    """
    def __init__(self, domain_a_dir, domain_b_dir, transform=None, filenames=False):
        """
        Initialize the dataset.
        Args:
            domain_a_dir (str): Path to the directory containing Domain A images.
            domain_b_dir (str): Path to the directory containing Domain B images.
            transform (torchvision.transforms.Compose, optional): Transformations to apply to the images.
            filenames (bool): If True, return the file names along with the images.
        """
        self.domain_a_files = sorted([os.path.join(domain_a_dir, f) for f in os.listdir(domain_a_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
        self.domain_b_files = sorted([os.path.join(domain_b_dir, f) for f in os.listdir(domain_b_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
        self.transform = transform if transform else transforms.Compose([transforms.ToTensor()])
        self.filenames = filenames

    def __len__(self):
        """
        Return the length of the dataset.
        The dataset length is determined by the larger domain to allow full unpaired coverage.
        """
        return max(len(self.domain_a_files), len(self.domain_b_files))

    def __getitem__(self, index):
        """
        Get an item from the dataset.
        Args:
            index (int): Index of the item to retrieve.
        Returns:
            tuple: Transformed images from Domain A and Domain B, optionally with filenames.
        """
        # Cycle through the smaller dataset to match the larger one
        a_path = self.domain_a_files[index % len(self.domain_a_files)]
        b_path = self.domain_b_files[index % len(self.domain_b_files)]

        # Load images
        a_image = Image.open(a_path).convert('RGB')
        b_image = Image.open(b_path).convert('RGB')

        # Apply transformations
        if self.transform:
            a_image = self.transform(a_image)
            b_image = self.transform(b_image)

        if self.filenames:
            return a_image, b_image, os.path.basename(a_path), os.path.basename(b_path)
        else:
            return a_image, b_image


def _test():
    """
    Test the UnpairedDataset with sample directories.
    """
    # Use the absolute paths
    domain_a_dir = r"C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\data\EUVP\domianA"
    domain_b_dir = r"C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\data\EUVP\domianB"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor()
    ])

    dataset = UnpairedDataset(domain_a_dir, domain_b_dir, transform=transform, filenames=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for i, (a_images, b_images, a_names, b_names) in enumerate(loader):
        print(f"Batch {i}:")
        print(f"Domain A images: {a_images.shape}, Names: {a_names}")
        print(f"Domain B images: {b_images.shape}, Names: {b_names}")
        break

if __name__ == '__main__':
    _test()
