import os
import cv2
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Paths
input_path = './data/EUVP/domianA'
enhanced_path = './test/output'  # Directory with enhanced images
metrics_file = './test/metrics.csv'  # CSV file to save the metrics

# Ensure directories exist
if not os.path.exists(input_path):
    print(f"Input directory {input_path} does not exist.")
    exit()
if not os.path.exists(enhanced_path):
    print(f"Enhanced directory {enhanced_path} does not exist.")
    exit()

# Get sorted list of input and enhanced images
input_files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
enhanced_files = sorted([f for f in os.listdir(enhanced_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Check if files match
if len(input_files) != len(enhanced_files):
    print(f"Mismatch: {len(input_files)} input files and {len(enhanced_files)} enhanced files.")
    exit()

print(f"Found {len(input_files)} images. Starting evaluation...")

# Open CSV file to save metrics
os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
with open(metrics_file, mode='w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Image', 'SSIM', 'PSNR'])  # Write header

    # Process each pair of input and enhanced images
    for i, (input_file, enhanced_file) in enumerate(zip(input_files, enhanced_files), start=1):
        try:
            # Read input and enhanced images
            input_path_file = os.path.join(input_path, input_file)
            enhanced_path_file = os.path.join(enhanced_path, enhanced_file)

            input_img = cv2.imread(input_path_file)
            enhanced_img = cv2.imread(enhanced_path_file)

            if input_img is None or enhanced_img is None:
                print(f"Warning: Unable to read {input_file} or {enhanced_file}. Skipping...")
                continue

            # Resize enhanced image to match input size if necessary
            if input_img.shape != enhanced_img.shape:
                enhanced_img = cv2.resize(enhanced_img, (input_img.shape[1], input_img.shape[0]))

            # Convert to RGB if necessary
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

            # Determine a suitable win_size for SSIM
            min_dim = min(input_img.shape[0], input_img.shape[1])  # Smaller dimension of the input image
            win_size = max(3, min(7, min_dim))  # Ensure win_size is at least 3 and no larger than the smallest dimension

            # Compute SSIM and PSNR
            ssim_value = ssim(input_img, enhanced_img, multichannel=True, win_size=win_size, channel_axis=-1)
            psnr_value = psnr(input_img, enhanced_img)

            # Write metrics to CSV
            csvwriter.writerow([input_file, ssim_value, psnr_value])
            print(f"Processed {input_file} - SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.2f}")

        except Exception as e:
            print(f"Error processing {input_file} and {enhanced_file}: {e}. Skipping...")

print(f"Evaluation complete. Metrics saved to {metrics_file}.")
