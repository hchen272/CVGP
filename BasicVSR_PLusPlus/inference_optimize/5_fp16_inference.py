#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mixed Precision (FP16) Inference for BasicVSR++.
Measures FP16 performance and saves metrics in the same format as baseline.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.model_loader import load_model
from utils.inference import measure_inference_speed, measure_gpu_memory, profile_model_memory

def parse_args():
    parser = argparse.ArgumentParser(description='FP16 mixed precision inference')
    parser.add_argument('--config', type=str, default='configs/basicvsr_plusplus_reds4.py',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='chkpts/basicvsr_plusplus_reds4.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--input_size', type=int, nargs=2, default=[256, 256],
                        help='Input image size (height, width)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Number of frames in temporal dimension')
    parser.add_argument('--num_warmup', type=int, default=10,
                        help='Number of warmup iterations')
    parser.add_argument('--num_iter', type=int, default=50,
                        help='Number of measured iterations')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device ID')
    parser.add_argument('--output_json', type=str, default='results/fp16_metrics.json',
                        help='Output JSON file for metrics')
    return parser.parse_args()

def main():
    args = parse_args()
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Load model in FP32
    print("Loading FP32 model...")
    model_fp32 = load_model(args.config, args.checkpoint, device=device)
    model_fp32.eval()

    # Create FP32 input
    h, w = args.input_size
    input_fp32 = torch.randn(args.batch_size, args.num_frames, 3, h, w).to(device)
    print(f"Input shape: {input_fp32.shape}")

    # Create FP16 model and input
    print("Converting model to FP16...")
    model_fp16 = model_fp32.half()
    model_fp16.eval()
    input_fp16 = input_fp32.half()

    # Measure FP16 performance (we only need FP16 for final output, but we can also measure FP32 for internal reference)
    print("\n" + "-"*50)
    print("Measuring FP16 performance...")
    try:
        speed_fp16 = measure_inference_speed(
            model_fp16, input_fp16,
            num_warmup=args.num_warmup,
            num_iter=args.num_iter,
            device=device
        )
        gpu_mem_fp16 = measure_gpu_memory(device)
        peak_mem_fp16 = profile_model_memory(model_fp16, input_fp16, device)
        fp16_success = True
    except Exception as e:
        print(f"FP16 measurement failed: {e}")
        fp16_success = False

    if not fp16_success:
        print("FP16 inference failed. Exiting.")
        return

    # Build metrics dict in the same format as baseline
    metrics = {
        'fp16': {
            'device': device,
            'input_shape': list(input_fp16.shape),
            'num_iter': args.num_iter,
            'avg_time_ms': speed_fp16['avg_time_ms'],
            'fps': speed_fp16['fps'],
            'gpu_memory_mb': gpu_mem_fp16,
            'peak_memory_mb': peak_mem_fp16['peak_memory_mb'],
        }
    }

    # Optionally also include FP32 results (for internal comparison, not required for final chart)
    # But we can skip to keep output clean.

    # Print summary
    print("\n" + "="*50)
    print("FP16 MIXED PRECISION RESULTS")
    print("="*50)
    print(f"FP16 - FPS: {speed_fp16['fps']:.2f}, Time: {speed_fp16['avg_time_ms']:.2f} ms")
    print(f"Peak Memory: {peak_mem_fp16['peak_memory_mb']:.1f} MB")
    print(f"GPU allocated: {gpu_mem_fp16['allocated_mb']:.1f} MB")
    print("="*50)

    # Save metrics
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {output_path}")

if __name__ == '__main__':
    main()