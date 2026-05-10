#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process Vimeo-90K dataset with BasicVSR++ while preserving folder structure.
Assumes input_root contains subfolders (e.g., 000, 001, ...), each containing PNG frames.
Outputs super-resolved frames in same subfolder structure under output_root.
Supports FP32 and FP16 precision.

To run: python scripts/process_reds_preserve_structure.py data/reds/input_videos results/fp32 --precision fp32
"""

import argparse
import sys
import time
from pathlib import Path
import torch
import cv2
import numpy as np
from tqdm import tqdm
import os

# Add inference_optimize to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'inference_optimize'))
from utils.model_loader import load_model

def get_sequence_dirs(root_dir):
    """
    Return list of immediate subdirectories under root_dir.
    Each subdirectory is considered a sequence folder.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input root not found: {root}")
    # List only directories (skip files)
    seq_dirs = [d for d in root.iterdir() if d.is_dir()]
    # Optionally filter those that contain PNG files (optional but robust)
    valid_dirs = []
    for d in seq_dirs:
        if any(d.glob("*.png")):
            valid_dirs.append(d)
        else:
            print(f"Warning: {d} contains no PNG files, skipping.")
    return valid_dirs

def process_sequence(model, input_dir, output_dir, device='cuda:0', precision='fp32'):
    """
    Run super-resolution on a single sequence (all PNG files in input_dir).
    Input frames are sorted alphanumerically; output frames keep same filenames.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all PNG files in input_dir, sort naturally
    frame_paths = sorted(input_dir.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not frame_paths:
        raise RuntimeError(f"No PNG files found in {input_dir}")

    # Build input tensor (1 x T x 3 x H x W)
    frames = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        frames.append(img)
    input_tensor = torch.stack(frames).unsqueeze(0).to(device)  # [1, T, 3, H, W]
    if precision == 'fp16':
        input_tensor = input_tensor.half()

    with torch.no_grad():
        output_dict = model.forward_test(input_tensor)
        output_tensor = output_dict['output']  # [1, T, 3, 4H, 4W]

    # Save output frames with same filenames
    for j, out_path in enumerate(frame_paths):
        out_img = output_tensor[0, j].cpu().float().permute(1,2,0).numpy() * 255.0
        out_img = np.clip(out_img, 0, 255).astype(np.uint8)
        target_path = output_dir / out_path.name
        cv2.imwrite(str(target_path), out_img)

def main():
    parser = argparse.ArgumentParser(description="Process Vimeo-90K preserving folder structure")
    parser.add_argument("input_root", type=str, help="Root directory containing sequence subfolders")
    parser.add_argument("output_root", type=str, help="Root directory for output (subfolders will be mirrored)")
    parser.add_argument("--config", type=str, default="configs/basicvsr_plusplus_reds4.py")
    parser.add_argument("--checkpoint", type=str, default="chkpts/basicvsr_plusplus_reds4.pth")
    parser.add_argument("--precision", type=str, choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}, Precision: {args.precision}")

    # Load model
    print("Loading model...")
    model = load_model(args.config, args.checkpoint, device=device)
    if args.precision == 'fp16':
        model = model.half()
    model.eval()

    # Get sequence directories (immediate subdirs of input_root)
    seq_dirs = get_sequence_dirs(args.input_root)
    print(f"Found {len(seq_dirs)} sequence folders.")
    if not seq_dirs:
        print("No sequence folders found. Please check input_root path and folder structure.")
        return

    for input_seq_dir in tqdm(seq_dirs, desc="Processing sequences"):
        rel_path = input_seq_dir.relative_to(args.input_root)
        output_seq_dir = Path(args.output_root) / rel_path
        try:
            process_sequence(model, input_seq_dir, output_seq_dir, device, args.precision)
        except Exception as e:
            print(f"Error processing {input_seq_dir}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()