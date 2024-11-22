import os
import argparse
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description='Depth Evaluation Script', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--pred_path', default='./output_monodepth/', type=str, help='Path to prediction results')
parser.add_argument('--gt_path', default='./output/gt/', type=str, help='Path to ground truth depth maps')
parser.add_argument('--dataset', type=str, default='nyu', help='Dataset: nyu or kitti')
parser.add_argument('--min_depth_eval', type=float, default=1e-3, help='Minimum depth for evaluation')
parser.add_argument('--max_depth_eval', type=float, default=10.0, help='Maximum depth for evaluation')

args = parser.parse_args()

def compute_errors(gt, pred):
    """Compute standard depth estimation errors."""
    thresh = np.maximum(gt / pred, pred / gt)
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    silog = np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2) - (np.mean(np.log(pred) - np.log(gt))) ** 2) * 100
    log10 = np.mean(np.abs(np.log10(pred) - np.log10(gt)))

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3

def load_predictions(pred_path):
    """Load prediction depth maps."""
    pred_files = sorted([f for f in os.listdir(pred_path) if f.endswith('.png')])
    pred_depths = []

    for pred_file in pred_files:
        pred_path_full = os.path.join(pred_path, pred_file)
        pred_depth = cv2.imread(pred_path_full, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # Convert from millimeters
        pred_depth = np.clip(pred_depth, args.min_depth_eval, args.max_depth_eval)
        pred_depths.append(pred_depth)

    print(f"Loaded {len(pred_depths)} predictions.")
    return pred_depths

def load_ground_truth(gt_path):
    """Load ground truth depth maps."""
    gt_files = sorted([f for f in os.listdir(gt_path) if f.endswith('.png')])
    gt_depths = []

    for gt_file in gt_files:
        gt_path_full = os.path.join(gt_path, gt_file)
        gt_depth = cv2.imread(gt_path_full, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # Convert from millimeters
        gt_depth = np.clip(gt_depth, args.min_depth_eval, args.max_depth_eval)
        gt_depths.append(gt_depth)

    print(f"Loaded {len(gt_depths)} ground truth depth maps.")
    return gt_depths

def evaluate(pred_depths, gt_depths):
    """Evaluate predictions against ground truth."""
    assert len(pred_depths) == len(gt_depths), "Mismatch between predictions and ground truth."
    num_samples = len(pred_depths)

    metrics = np.zeros((num_samples, 9))  # 9 evaluation metrics

    for i in range(num_samples):
        pred = pred_depths[i]
        gt = gt_depths[i]

        # Mask invalid regions
        valid_mask = (gt > args.min_depth_eval) & (gt < args.max_depth_eval)
        if np.any(valid_mask):
            metrics[i] = compute_errors(gt[valid_mask], pred[valid_mask])

    mean_metrics = metrics.mean(axis=0)
    print(f"Evaluation Results: {mean_metrics}")
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10'))
    print("{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(*mean_metrics))

def main():
    pred_depths = load_predictions(args.pred_path)
    gt_depths = load_ground_truth(args.gt_path)
    evaluate(pred_depths, gt_depths)

if __name__ == '__main__':
    main()
