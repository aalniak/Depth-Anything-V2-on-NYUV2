import kagglehub
import torch
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error
from depth_anything_v2.dpt import DepthAnythingV2
import argparse
import os
# KITTI Depth Scaling Factor (as per official documentation)
DEPTH_SCALE = 256.0  
MAX_DEPTH = 80  # Maximum depth for occluded/missing pixels


def compute_depth_metrics(pred_depth, gt_depth, max_depth=MAX_DEPTH):
    """Compute depth estimation error metrics, ignoring MAX_DEPTH pixels."""
    valid_mask = (gt_depth > 0) & (gt_depth < max_depth)  # Ignore zero and MAX_DEPTH pixels

    
    pred_depth = pred_depth[valid_mask]
    gt_depth = gt_depth[valid_mask]

    if len(gt_depth) == 0:
        print("Warning: No valid pixels found for metric computation!")
        return {"RMSE": np.nan, "Log RMSE": np.nan, "AbsRel": np.nan, "SqRel": np.nan, "SI Log Error": np.nan}

    # RMSE
    rmse = np.sqrt(mean_squared_error(gt_depth, pred_depth))

    # Log RMSE
    log_rmse = np.sqrt(mean_squared_error(np.log(gt_depth), np.log(pred_depth)))

    # Absolute Relative Difference
    absrel = np.mean(np.abs(gt_depth - pred_depth) / gt_depth)

    # Squared Relative Difference
    sqrel = np.mean(((gt_depth - pred_depth) ** 2) / gt_depth)

    # Scale Invariant Log Error
    log_diff = np.log(pred_depth) - np.log(gt_depth)
    silog = np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2))

    return {
        "RMSE": rmse,
        "Log RMSE": log_rmse,
        "AbsRel": absrel,
        "SqRel": sqrel,
        "SI Log Error": silog
    }




def run_kitti_test(dataset_path,model,args):
    total_metrics = {"RMSE": 0, "Log RMSE": 0, "AbsRel": 0, "SqRel": 0, "SI Log Error": 0}
    num_samples = 0
    
    dataset_path += "/"
    print("Path to dataset files:", dataset_path)
    
    # Read file containing image-depth pairs
    with open(dataset_path + "test_locations.txt", "r") as file:
        for line in file:
            img_path, gt_path = line.strip().split(" ")

            # Load depth map as uint16 (as per KITTI format)
            gt_depth = cv2.imread(dataset_path + gt_path, cv2.IMREAD_UNCHANGED)
            if gt_depth is None:
                print(f"Error loading depth map: {dataset_path + gt_path}")
                continue
            # Convert depth map to float32 and scale to meters
            gt_depth = gt_depth.astype(np.float32) / DEPTH_SCALE
            # Create a mask for invalid pixels (zero values = no valid depth data)
            zero_mask = (gt_depth == 0)
            # Set invalid depth pixels to MAX_DEPTH for visualization purposes
            gt_depth[zero_mask] = MAX_DEPTH
            rgb_image = cv2.imread(dataset_path + img_path, cv2.IMREAD_UNCHANGED)
            rgb_image = np.array(rgb_image)
            inferred_depth = model.infer_image(rgb_image, args.input_size)
            
        
            metrics = compute_depth_metrics(inferred_depth, gt_depth, MAX_DEPTH)

            # Accumulate metrics
            for key in total_metrics:
                total_metrics[key] += metrics[key]
        
            num_samples += 1
        
            print(f"Metrics: {metrics}")
            
    
    # Compute average metrics
        avg_metrics = {key: total_metrics[key] / num_samples for key in total_metrics}
        print(f"Average Metrics: {avg_metrics}")
        return avg_metrics


parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation on NYUV2 Dataset')

parser.add_argument('--input-size', type=int, default=518)
parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
parser.add_argument('--max-depth', type=float, default=20)

args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()
        

# Download the dataset from KaggleHub
dataset_path = kagglehub.dataset_download("aalniak/kitti-eigen-split-for-monocular-depth-estimation")
# Call the function to visualize depth maps correctly
run_kitti_test(dataset_path,depth_anything,args)
