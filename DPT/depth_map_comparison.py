import os
import numpy as np
import cv2
import csv


def load_image(file_path, single_channel=False):
    """Load an image as a numpy array."""
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found: {file_path}")
    if single_channel and len(image.shape) == 3:
        # Convert to single channel (grayscale) if requested
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def compute_rmse(predicted, ground_truth):
    """Compute Root Mean Squared Error (RMSE)."""
    return np.sqrt(((predicted - ground_truth) ** 2).mean())


def compute_mae(predicted, ground_truth):
    """Compute Mean Absolute Error (MAE)."""
    return np.abs(predicted - ground_truth).mean()


def extract_common_identifier(filename, prefix, suffix):
    """
    Extract the numerical identifier from a filename by removing the prefix and suffix.
    """
    filename = filename.replace(prefix, "").replace(suffix, "").strip("_")
    # Attempt to extract numerical part
    return ''.join(filter(str.isdigit, filename))


def evaluate_depth_metrics(predicted_folder, ground_truth_folder, output_csv_path):
    """
    Evaluate depth estimation results using RMSE and MAE metrics.
    
    Args:
        predicted_folder (str): Path to the folder containing predicted depth maps.
        ground_truth_folder (str): Path to the folder containing ground truth depth maps.
        output_csv_path (str): Path to save the evaluation results as a CSV file.
    """
    predicted_files = sorted([f for f in os.listdir(predicted_folder) if f.endswith("_depth.png")])
    ground_truth_files = sorted([f for f in os.listdir(ground_truth_folder) if f.endswith(".png")])

    # Match files based on their numerical identifier
    matching_files = []
    for pred_file in predicted_files:
        pred_id = extract_common_identifier(pred_file, "nm_", "_depth.png")
        for gt_file in ground_truth_files:
            gt_id = extract_common_identifier(gt_file, "", "_img_.png")
            if pred_id == gt_id:  # Match by identifier
                matching_files.append((pred_file, gt_file))
                break

    if not matching_files:
        raise ValueError("No matching predicted and ground truth depth maps found. Ensure filenames are correctly formatted.")

    print(f"Found {len(matching_files)} matching files for comparison.")

    metrics = {"RMSE": [], "MAE": []}
    results = []

    for pred_file, gt_file in matching_files:
        # Load predicted and ground truth images
        pred_path = os.path.join(predicted_folder, pred_file)
        gt_path = os.path.join(ground_truth_folder, gt_file)

        predicted = load_image(pred_path, single_channel=True).astype(np.float32)
        ground_truth = load_image(gt_path, single_channel=True).astype(np.float32)

        # Ensure the dimensions match
        if predicted.shape != ground_truth.shape:
            print(f"Resizing ground truth {gt_file} to match {pred_file}")
            ground_truth = cv2.resize(ground_truth, (predicted.shape[1], predicted.shape[0]))

        # Normalize ground truth and predictions if needed
        predicted /= 255.0  # Normalize to [0, 1]
        ground_truth /= 255.0  # Normalize to [0, 1]

        # Compute metrics
        rmse = compute_rmse(predicted, ground_truth)
        mae = compute_mae(predicted, ground_truth)

        metrics["RMSE"].append(rmse)
        metrics["MAE"].append(mae)

        results.append({"Image": pred_file, "RMSE": rmse, "MAE": mae})

        print(f"Image: {pred_file} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    # Aggregate overall metrics
    metrics["Overall RMSE"] = np.mean(metrics["RMSE"])
    metrics["Overall MAE"] = np.mean(metrics["MAE"])
    print(f"\nOverall RMSE: {metrics['Overall RMSE']:.4f}")
    print(f"Overall MAE: {metrics['Overall MAE']:.4f}")

    results.append({"Image": "Overall", "RMSE": metrics["Overall RMSE"], "MAE": metrics["Overall MAE"]})

    # Save results to a CSV file
    with open(output_csv_path, mode="w", newline="") as csv_file:
        fieldnames = ["Image", "RMSE", "MAE"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    print(f"Saved evaluation results to {output_csv_path}")


# Example Usage
if __name__ == "__main__":
    # Replace these paths with the actual paths to your predicted and ground truth folders
    predicted_depth_folder = 'output_withGAN'  # Folder containing predicted depth maps
    ground_truth_folder = r'C:\Users\golno\OneDrive\Desktop\uwgan\Depth-Aware-U-shape-Transformer\datasets\UIEB'  # Folder containing ground truth images (GTr)
    output_csv_path = "evaluation_results.csv"  # Path to save the evaluation results as a CSV file

    try:
        evaluate_depth_metrics(predicted_depth_folder, ground_truth_folder, output_csv_path)
    except ValueError as e:
        print(f"Error: {e}")
