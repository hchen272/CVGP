#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for Direction C:
- Read BasicVSR++ results (HR frames)
- Compute confidence map using uncertainty.py
- Enhance low-confidence regions with Real-ESRGAN
- Save final results

To run: python run_direction_c.py --input_root D:\VideoSuperResolution\BasicVSR_PlusPlus\results\fp16 --output_root D:\VideoSuperResolution\BasicVSR_PlusPlus\results\direction_c --esrgan_exe D:\VideoSuperResolution\ESRGAN\realesrgan-ncnn-vulkan.exe --confidence_method combined --conf_threshold 0.3 --min_area 100 --feather_radius 15 --save_vis
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Add direction_c to path
sys.path.insert(0, str(Path(__file__).parent))
from uncertainty import UncertaintyEstimator, visualize_confidence_overlay
from patch_processor import enhance_frame_with_esrgan_patches


def process_sequence(
    input_seq_dir: Path,
    output_seq_dir: Path,
    uncertainty_method: str = "combined",
    conf_threshold: float = 0.3,
    min_area: int = 100,
    esrgan_exe: str = "",
    esrgan_model: str = "realesr-animevideov3",
    esrgan_scale: int = 2,
    feather_radius: int = 15,
    save_confidence_vis: bool = False
):
    """
    Process a single sequence folder.
    """
    # Get all PNG frames
    frame_paths = sorted(input_seq_dir.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not frame_paths:
        print(f"No PNG frames found in {input_seq_dir}")
        return
    
    output_seq_dir.mkdir(parents=True, exist_ok=True)
    if save_confidence_vis:
        vis_dir = output_seq_dir / "confidence_vis"
        vis_dir.mkdir(exist_ok=True)
    
    # Initialize uncertainty estimator (heuristic, no model needed)
    estimator = UncertaintyEstimator(model=None, device="cpu")
    
    for frame_path in tqdm(frame_paths, desc=f"Processing {input_seq_dir.name}"):
        # Read HR frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        # Compute confidence map from the frame itself (texture/gradient based)
        # Need to convert to tensor format expected by UncertaintyEstimator
        # The estimator expects [1, T, C, H, W], so we create a dummy batch.
        # We'll use the last frame (only one frame) for heuristic methods.
        frame_tensor = torch.from_numpy(frame).permute(2,0,1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(1)  # [1,1,C,H,W]
        confidence_map, _ = estimator.compute_confidence_map(frame_tensor, method=uncertainty_method)
        
        # confidence_map is already at same resolution as frame (since we used frame_tensor)
        # Ensure it's float32 and range [0,1]
        confidence_map = confidence_map.astype(np.float32)
        confidence_map = np.clip(confidence_map, 0, 1)
        
        # Enhance frame using ESRGAN patches
        enhanced_frame = enhance_frame_with_esrgan_patches(
            frame.copy(),
            confidence_map,
            esrgan_exe_path=esrgan_exe,
            conf_threshold=conf_threshold,
            min_area=min_area,
            expand_patch=10,
            esrgan_scale=esrgan_scale,
            feather_radius=feather_radius,
            model_name=esrgan_model
        )
        
        # Save result
        out_path = output_seq_dir / frame_path.name
        cv2.imwrite(str(out_path), enhanced_frame)
        
        # Optional: save confidence map visualization
        if save_confidence_vis:
            overlay = visualize_confidence_overlay(confidence_map, frame, alpha=0.5)
            vis_path = vis_dir / frame_path.name
            cv2.imwrite(str(vis_path), overlay)
    
    print(f"Finished {input_seq_dir.name}, saved to {output_seq_dir}")


def main():
    parser = argparse.ArgumentParser(description="Direction C: Uncertainty-aware refinement with ESRGAN")
    parser.add_argument("--input_root", type=str,
                        default=r"D:\VideoSuperResolution\BasicVSR_PlusPlus\results\fp32",
                        help="Root directory containing BasicVSR++ HR results (subfolders per sequence)")
    parser.add_argument("--output_root", type=str,
                        default=r"D:\VideoSuperResolution\BasicVSR_PlusPlus\results\direction_c",
                        help="Output directory for Direction C results")
    parser.add_argument("--esrgan_exe", type=str,
                        default=r"D:\VideoSuperResolution\ESRGAN\realesrgan-ncnn-vulkan.exe",
                        help="Path to Real-ESRGAN exe")
    parser.add_argument("--esrgan_model", type=str, default="realesr-animevideov3",
                        help="ESRGAN model name")
    parser.add_argument("--esrgan_scale", type=int, default=2, choices=[1,2,4],
                        help="ESRGAN scale factor (2: enhance texture, then downscale back)")
    parser.add_argument("--confidence_method", type=str, default="combined",
                        choices=["texture", "gradient", "combined"],
                        help="Uncertainty estimation method")
    parser.add_argument("--conf_threshold", type=float, default=0.3,
                        help="Confidence threshold (below this -> low confidence)")
    parser.add_argument("--min_area", type=int, default=100,
                        help="Minimum region area to process")
    parser.add_argument("--feather_radius", type=int, default=15,
                        help="Feathering radius for blending")
    parser.add_argument("--save_vis", action="store_true",
                        help="Save confidence map overlays")
    args = parser.parse_args()
    
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    # modify to match expected folder structure: input_root should contain subfolders like '000', '001', etc.
    
    if not input_root.exists():
        print(f"Input root not found: {input_root}")
        return
    
    # Get all sequence folders (immediate subdirectories)
    seq_dirs = [d for d in input_root.iterdir() if d.is_dir()]
    if not seq_dirs:
        print(f"No sequence folders found in {input_root}")
        return
    
    print(f"Found {len(seq_dirs)} sequences: {[d.name for d in seq_dirs]}")
    
    # Import torch only here (if needed for dummy tensor)
    global torch
    import torch
    
    for seq_dir in seq_dirs:
        output_seq_dir = output_root / seq_dir.name
        process_sequence(
            input_seq_dir=seq_dir,
            output_seq_dir=output_seq_dir,
            uncertainty_method=args.confidence_method,
            conf_threshold=args.conf_threshold,
            min_area=args.min_area,
            esrgan_exe=args.esrgan_exe,
            esrgan_model=args.esrgan_model,
            esrgan_scale=args.esrgan_scale,
            feather_radius=args.feather_radius,
            save_confidence_vis=args.save_vis
        )
    
    print("Direction C processing complete.")


if __name__ == "__main__":
    main()