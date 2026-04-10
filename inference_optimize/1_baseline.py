#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline performance measurement for BasicVSR++.
Measures inference speed (FPS) and GPU memory usage.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from utils.model_loader import load_model
from utils.inference import measure_inference_speed, measure_gpu_memory, profile_model_memory

def parse_args():
    parser = argparse.ArgumentParser(description='Baseline measurement for BasicVSR++')
    parser.add_argument('--config', type=str, 
                        default='configs/basicvsr_plusplus_reds4.py',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str,
                        default='chkpts/basicvsr_plusplus_reds4.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--input_size', type=int, nargs=2, default=[256, 256],
                        help='Input image size (height, width)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Number of frames in temporal dimension (sequence length)')
    parser.add_argument('--num_warmup', type=int, default=10,
                        help='Number of warmup iterations')
    parser.add_argument('--num_iter', type=int, default=100,
                        help='Number of measured iterations')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device ID')
    parser.add_argument('--output_json', type=str, default='results/baseline_metrics.json',
                        help='Output JSON file for metrics')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create results directory if not exists
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.config, args.checkpoint, device=device)
    
    # Create dummy input tensor with temporal dimension.
    # BasicVSR++ expects shape: (batch, time, channels, height, width)
    h, w = args.input_size
    dummy_input = torch.randn(args.batch_size, args.num_frames, 3, h, w).to(device)
    print(f"Input shape: {dummy_input.shape}")
    
    print(f"Measuring inference speed (num_iter={args.num_iter})...")
    
    # Measure speed (the inference functions inside utils already use forward_test)
    speed_metrics = measure_inference_speed(
        model, dummy_input, 
        num_warmup=args.num_warmup, 
        num_iter=args.num_iter,
        device=device
    )
    
    # Measure GPU memory after inference
    gpu_mem = measure_gpu_memory(device)
    peak_mem = profile_model_memory(model, dummy_input, device)
    
    # Combine metrics
    metrics = {
        'baseline': {
            'device': device,
            'input_shape': speed_metrics['input_shape'],
            'num_iter': speed_metrics['num_iter'],
            'avg_time_ms': speed_metrics['avg_time_ms'],
            'fps': speed_metrics['fps'],
            'gpu_memory_mb': gpu_mem,
            'peak_memory_mb': peak_mem['peak_memory_mb']
        }
    }
    
    # Print results
    print("\n" + "="*50)
    print("BASELINE PERFORMANCE METRICS")
    print("="*50)
    print(f"Device: {device}")
    print(f"Input shape: {speed_metrics['input_shape']}")
    print(f"Average inference time: {speed_metrics['avg_time_ms']:.2f} ms")
    print(f"Frames per second (FPS): {speed_metrics['fps']:.2f}")
    print(f"GPU memory allocated: {gpu_mem['allocated_mb']:.2f} MB")
    print(f"GPU memory cached: {gpu_mem['cached_mb']:.2f} MB")
    print(f"Peak memory usage: {peak_mem['peak_memory_mb']:.2f} MB")
    print("="*50)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {output_path}")

if __name__ == '__main__':
    main()