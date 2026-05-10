#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model quantization (Post-Training Static Quantization) for BasicVSR++.
Measures performance of INT8 quantized model.
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.ao.quantization as quant
from utils.model_loader import load_model
from utils.inference import measure_inference_speed, measure_gpu_memory, profile_model_memory

def parse_args():
    parser = argparse.ArgumentParser(description='Quantization for BasicVSR++')
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
    parser.add_argument('--num_iter', type=int, default=100,
                        help='Number of measured iterations')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device ID (quantization runs on CPU)')
    parser.add_argument('--quant_backend', type=str, default='fbgemm',
                        choices=['fbgemm', 'qnnpack'],
                        help='Quantization backend (fbgemm for x86, qnnpack for ARM)')
    parser.add_argument('--num_calib_batches', type=int, default=10,
                        help='Number of calibration batches (each batch has num_frames frames)')
    parser.add_argument('--output_json', type=str, default='results/quantize_metrics.json',
                        help='Output JSON file for metrics')
    parser.add_argument('--save_quantized', action='store_true',
                        help='Save quantized model to disk')
    parser.add_argument('--quantized_path', type=str, default='models/basicvsrpp_int8.pth',
                        help='Path to save quantized model state dict')
    return parser.parse_args()

def calibrate_model(model, calibration_dataloader, device='cpu'):
    """
    Run calibration on the model with inserted observers.
    Uses forward_test to avoid training path.
    """
    model.eval()
    with torch.no_grad():
        for batch in calibration_dataloader:
            batch = batch.to(device)
            # Use forward_test instead of __call__ to avoid requiring GT
            model.forward_test(batch)

def create_calibration_dataloader(input_shape, num_batches, batch_size, num_frames):
    """
    Create a dummy calibration dataloader using random tensors.
    Each batch is a tensor of shape (batch, time, C, H, W) representing LQ frames.
    """
    calib_data = []
    for _ in range(num_batches):
        dummy = torch.randn(batch_size, num_frames, 3, input_shape[0], input_shape[1])
        calib_data.append(dummy)
    return calib_data

def fuse_model(model):
    """
    Fuse eligible layers (Conv+BN+ReLU) to improve quantization accuracy.
    BasicVSR++ may not have many fusable patterns; we skip for now.
    """
    # Return model unchanged (no fusion implemented)
    return model

def main():
    args = parse_args()

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device_inference = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f"Inference device for measurement: {device_inference}")
    print(f"Quantization backend: {args.quant_backend}")

    # Load original model on CPU (quantization must be done on CPU)
    print("Loading original model on CPU...")
    model = load_model(args.config, args.checkpoint, device='cpu')
    model.eval()

    # Fuse layers (optional)
    model = fuse_model(model)

    # Set quantization configuration
    model.qconfig = torch.ao.quantization.get_default_qconfig(args.quant_backend)

    # Prepare model for calibration (insert observers)
    quant.prepare(model, inplace=True)

    # Create calibration data loader
    h, w = args.input_size
    calib_data = create_calibration_dataloader(
        input_shape=(h, w),
        num_batches=args.num_calib_batches,
        batch_size=args.batch_size,
        num_frames=args.num_frames
    )

    # Calibrate
    print(f"Calibrating with {args.num_calib_batches} batches...")
    calibrate_model(model, calib_data, device='cpu')

    # Convert to quantized model
    print("Converting to quantized INT8 model...")
    quantized_model = quant.convert(model, inplace=False)
    quantized_model.eval()

    # Keep quantized model on CPU for measurement (since GPU support for INT8 is limited)
    print("Quantized model will be measured on CPU.")

    # Prepare dummy input on CPU
    dummy_input = torch.randn(args.batch_size, args.num_frames, 3, h, w).to('cpu')
    print(f"Input shape: {dummy_input.shape}")

    # Measure baseline CPU performance (original model on CPU)
    print("Loading original model on CPU for baseline measurement...")
    model_cpu = load_model(args.config, args.checkpoint, device='cpu')
    model_cpu.eval()

    print("Measuring baseline CPU performance (original FP32 model)...")
    speed_cpu = measure_inference_speed(
        model_cpu, dummy_input,
        num_warmup=args.num_warmup,
        num_iter=args.num_iter,
        device='cpu'
    )

    print("Measuring quantized model performance on CPU...")
    speed_quant = measure_inference_speed(
        quantized_model, dummy_input,
        num_warmup=args.num_warmup,
        num_iter=args.num_iter,
        device='cpu'
    )

    # Save quantized model if requested
    if args.save_quantized:
        quantized_path = Path(args.quantized_path)
        quantized_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized_model.state_dict(), quantized_path)
        print(f"Quantized model state_dict saved to {quantized_path}")

    # Compile metrics
    metrics = {
        'quantization': {
            'backend': args.quant_backend,
            'num_calibration_batches': args.num_calib_batches,
            'input_shape': list(dummy_input.shape),
            'baseline_cpu': {
                'avg_time_ms': speed_cpu['avg_time_ms'],
                'fps': speed_cpu['fps'],
            },
            'quantized_cpu': {
                'avg_time_ms': speed_quant['avg_time_ms'],
                'fps': speed_quant['fps'],
            },
            'speedup': speed_cpu['avg_time_ms'] / speed_quant['avg_time_ms'],
        }
    }

    print("\n" + "="*50)
    print("QUANTIZATION PERFORMANCE METRICS (CPU)")
    print("="*50)
    print(f"Baseline CPU FPS: {speed_cpu['fps']:.2f}")
    print(f"Quantized CPU FPS: {speed_quant['fps']:.2f}")
    print(f"Speedup: {metrics['quantization']['speedup']:.2f}x")
    print("="*50)

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {output_path}")

if __name__ == '__main__':
    main()