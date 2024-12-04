# import os
# import cv2
# import visdom
# import numpy as np

# # Initialize Visdom
# viz = visdom.Visdom()

# # Define paths
# input_folder = r'C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\datasets\trainA'  # Original images
# depth_with_cyclegan = r'output_withGAN'        # Depth maps with CycleGAN preprocessing
# depth_without_cyclegan = r'output_monodepth'  # Depth maps without CycleGAN preprocessing

# # Resize dimensions (e.g., 256x256)
# resize_dim = (256, 256)

# # List all files
# input_images = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg'))])
# depth_with = sorted([os.path.join(depth_with_cyclegan, f) for f in os.listdir(depth_with_cyclegan) if f.endswith(('_depth.png', '_depth.jpg'))])
# depth_without = sorted([os.path.join(depth_without_cyclegan, f) for f in os.listdir(depth_without_cyclegan) if f.endswith(('_depth.png', '_depth.jpg'))])

# # Visualization loop
# for idx, (input_img_path, depth_with_path, depth_without_path) in enumerate(zip(input_images, depth_with, depth_without)):
#     # Load and resize the original image
#     original = cv2.imread(input_img_path)
#     original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for visualization
#     original_resized = cv2.resize(original, resize_dim)

#     # Load and resize the depth map with CycleGAN preprocessing
#     depth_map_with = cv2.imread(depth_with_path, cv2.IMREAD_UNCHANGED)
#     if depth_map_with.ndim > 2:  # Convert to grayscale if needed
#         depth_map_with = cv2.cvtColor(depth_map_with, cv2.COLOR_BGR2GRAY)
#     depth_map_with_resized = cv2.resize(depth_map_with, resize_dim)
#     depth_map_with_normalized = cv2.normalize(depth_map_with_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

#     # Load and resize the depth map without CycleGAN preprocessing
#     depth_map_without = cv2.imread(depth_without_path, cv2.IMREAD_UNCHANGED)
#     if depth_map_without.ndim > 2:  # Convert to grayscale if needed
#         depth_map_without = cv2.cvtColor(depth_map_without, cv2.COLOR_BGR2GRAY)
#     depth_map_without_resized = cv2.resize(depth_map_without, resize_dim)
#     depth_map_without_normalized = cv2.normalize(depth_map_without_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

#     # Expand grayscale depth maps to 3 channels for alignment with RGB images
#     depth_map_with_colored = cv2.merge([depth_map_with_normalized] * 3)
#     depth_map_without_colored = cv2.merge([depth_map_without_normalized] * 3)

#     # Stack the three images side-by-side
#     combined = np.hstack([
#         original_resized,  # RGB image (H, W, 3)
#         depth_map_with_colored,  # Depth map with CycleGAN (H, W, 3)
#         depth_map_without_colored  # Depth map without CycleGAN (H, W, 3)
#     ])

#     # Display the stacked images in Visdom
#     viz.image(
#         combined.transpose(2, 0, 1),  # Convert to (C, H, W) format for Visdom
#         opts=dict(title=f"Comparison {idx + 1}: Original | With CycleGAN | Without CycleGAN")
#     )



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_depth_maps(folder1, folder2, output_folder):
    """
    Compare depth maps from two folders and save difference visualizations.
    
    Args:
        folder1: Path to the first folder (e.g., original depth maps).
        folder2: Path to the second folder (e.g., CycleGAN depth maps).
        output_folder: Path to save comparison visualizations.
    """
    os.makedirs(output_folder, exist_ok=True)
    folder1_files = sorted([f for f in os.listdir(folder1) if f.endswith("_depth.png")])
    folder2_files = sorted([f for f in os.listdir(folder2) if f.endswith("_depth.png")])

    if len(folder1_files) != len(folder2_files):
        print("Warning: The number of files in the two folders is not equal.")
    
    for file1, file2 in zip(folder1_files, folder2_files):
        path1 = os.path.join(folder1, file1)
        path2 = os.path.join(folder2, file2)

        # Load depth maps
        depth1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth2 = cv2.imread(path2, cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        # Calculate difference
        difference = np.abs(depth1 - depth2)
        
        # Normalize for visualization
        diff_vis = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Save visualization
        output_path = os.path.join(output_folder, f"comparison_{os.path.basename(file1)}")
        cv2.imwrite(output_path, diff_vis)

        # Display comparison
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Folder 1 Depth Map")
        plt.imshow(depth1, cmap='viridis')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title("Folder 2 Depth Map")
        plt.imshow(depth2, cmap='viridis')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title("Difference")
        plt.imshow(diff_vis, cmap='hot')
        plt.colorbar()

        plt.tight_layout()
        plt.show()
        print(f"Saved comparison visualization to {output_path}")

if __name__ == "__main__":
    folder1 = "output_monodepth"  # Path to folder with original depth maps
    folder2 = "output_withGAN"  # Path to folder with CycleGAN depth maps
    output_folder = "comparison_results"  # Folder to save visualizations

    compare_depth_maps(folder1, folder2, output_folder)