import argparse
import numpy as np
import os
from datasets import load_dataset
from sklearn.metrics import mean_squared_error
import torch
from depth_anything_v2.dpt import DepthAnythingV2


def compute_depth_metrics(pred_depth, gt_depth):
    """Compute depth estimation error metrics: RMSE, log RMSE, AbsRel, SqRel, and SI Log Error."""
    valid_mask = gt_depth > 0  # Avoid invalid depth values
    pred_depth = pred_depth[valid_mask]
    gt_depth = gt_depth[valid_mask]
    
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

def process_dataset(dataset, args, model):
    print(f"Dataset length: {len(dataset)}")
    print(f"Sample structure: {dataset[0]}")
    print(f"Available splits: {dataset}")
    total_metrics = {"RMSE": 0, "Log RMSE": 0, "AbsRel": 0, "SqRel": 0, "SI Log Error": 0}
    num_samples = 0
    
    for idx, sample in enumerate(list(dataset)):
        #does not reach this line
        print(f"Processing Sample {idx}")
        rgb_image = sample['image']  # RGB image
        gt_depth = np.array(sample['depth_map'])  # Ground truth depth
        depth = model.infer_image(np.array(rgb_image), args.input_size)
        
        metrics = compute_depth_metrics(depth, gt_depth)
        
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        
        num_samples += 1
        print(f"Metrics: {metrics}")
    
    avg_metrics = {key: total_metrics[key] / num_samples for key in total_metrics}
    print(f"Average Metrics: {avg_metrics}")
    return avg_metrics

if __name__ == '__main__':
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


    home_dir = os.environ["HOME"] # to save the nyu_cache
    print(home_dir)
    dataset = load_dataset("sayakpaul/nyu_depth_v2", split="validation[:654]", cache_dir=home_dir+"/nyu_cache")
    #dataset = dataset.select(range(0, 654, 6))  # Sample every 6th data in dataset
    process_dataset(dataset, args, depth_anything)
    