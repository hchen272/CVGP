#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance measurement with Torch-TensorRT optimization.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch_tensorrt  # 导入 Torch-TensorRT 库

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.model_loader import load_model
from utils.inference import measure_inference_speed, measure_gpu_memory, profile_model_memory

def parse_args():
    parser = argparse.ArgumentParser(description='Torch-TensorRT optimization for BasicVSR++')
    parser.add_argument('--config', type=str, default='configs/basicvsr_plusplus_reds4.py')
    parser.add_argument('--checkpoint', type=str, default='chkpts/basicvsr_plusplus_reds4.pth')
    parser.add_argument('--input_size', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=5)
    parser.add_argument('--num_warmup', type=int, default=10)
    parser.add_argument('--num_iter', type=int, default=50)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16'],
                        help='Precision mode for TensorRT (fp32, fp16)')
    parser.add_argument('--output_json', type=str, default='results/torch_tensorrt_metrics.json')
    return parser.parse_args()

def main():
    args = parse_args()
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = f'cuda:{args.device}'
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # 加载原始模型
    print("Loading original model...")
    original_model = load_model(args.config, args.checkpoint, device=device)

    # 准备示例输入
    h, w = args.input_size
    sample_input = torch.randn(args.batch_size, args.num_frames, 3, h, w).to(device)
    print(f"Sample input shape: {sample_input.shape}")

    # --- 应用 Torch-TensorRT 优化 ---
    print(f"Applying Torch-TensorRT with precision={args.precision}...")
    try:
        # 这里的关键是设置 backend="tensorrt"，其余用法与 torch.compile 完全一致
        optimized_model = torch.compile(
            original_model, 
            backend="tensorrt",
            options={
                "enabled_precisions": {torch.half} if args.precision == 'fp16' else {torch.float},
            }
        )
        # 触发编译
        print("Compiling model (this may take a few minutes on first run)...")
        with torch.no_grad():
            _ = optimized_model.forward_test(sample_input)
        print("Torch-TensorRT optimization applied successfully.")
    except Exception as e:
        print(f"Torch-TensorRT optimization failed: {e}")
        print("Falling back to original model.")
        optimized_model = original_model

    # 测量性能
    print(f"Measuring inference speed (num_iter={args.num_iter})...")
    speed_metrics = measure_inference_speed(
        optimized_model, sample_input,
        num_warmup=args.num_warmup,
        num_iter=args.num_iter,
        device=device
    )
    gpu_mem = measure_gpu_memory(device)
    peak_mem = profile_model_memory(optimized_model, sample_input, device)

    # 保存结果
    metrics = {
        'torch_tensorrt': {
            'device': device,
            'precision': args.precision,
            'input_shape': list(sample_input.shape),
            'num_iter': args.num_iter,
            'avg_time_ms': speed_metrics['avg_time_ms'],
            'fps': speed_metrics['fps'],
            'gpu_memory_mb': gpu_mem,
            'peak_memory_mb': peak_mem['peak_memory_mb'],
        }
    }
    print("\n" + "="*50)
    print("TORCH-TENSORRT PERFORMANCE METRICS")
    print("="*50)
    print(f"Average inference time: {speed_metrics['avg_time_ms']:.2f} ms")
    print(f"FPS: {speed_metrics['fps']:.2f}")
    print(f"Peak GPU memory: {peak_mem['peak_memory_mb']:.2f} MB")
    print("="*50)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {output_path}")

if __name__ == '__main__':
    main()