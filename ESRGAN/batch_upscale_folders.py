#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch upscale all sequence folders using Real-ESRGAN exe.
"""

import subprocess
import sys
from pathlib import Path
import argparse


def batch_upscale(
    input_root: str,
    output_root: str,
    exe_path: str,
    scale: int = 4,
    model: str = "realesrgan-x4plus",
    frame_format: str = "png",
    verbose: bool = True
):
    """
    Process each subfolder in input_root with Real-ESRGAN.

    Args:
        input_root: Directory containing sequence subfolders (e.g., '000', '001')
        output_root: Base output directory; results will be placed in output_root/<seq_name>/realesr_x<scale>
        exe_path: Path to realesrgan-ncnn-vulkan.exe
        scale: Upscaling factor (2 or 4)
        model: Model name
        frame_format: Output image format ('png', 'jpg')
        verbose: Print progress
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    exe_path = Path(exe_path)

    if not exe_path.exists():
        raise FileNotFoundError(f"ESRGAN executable not found: {exe_path}")
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    # Find all immediate subdirectories
    subfolders = [d for d in input_root.iterdir() if d.is_dir()]
    if not subfolders:
        print(f"No subfolders found in {input_root}")
        return

    print(f"Found {len(subfolders)} sequence folders to process.")

    for seq_folder in subfolders:
        seq_name = seq_folder.name
        input_dir = seq_folder
        output_dir = output_root / seq_name / f"realesr_x{scale}"
        output_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\nProcessing: {seq_name}")
            print(f"  Input:  {input_dir}")
            print(f"  Output: {output_dir}")

        cmd = [
            str(exe_path),
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-s", str(scale),
            "-n", model,
            "-f", frame_format,
            "-v"
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if verbose:
                print(f"  Done. (exit code {result.returncode})")
        except subprocess.CalledProcessError as e:
            print(f"  Upscaling failed for {seq_name}")
            print(f"  stderr: {e.stderr}")
            raise

    print("\nAll folders processed.")


def main():
    parser = argparse.ArgumentParser(description="Batch upscale video frames using Real-ESRGAN")
    parser.add_argument("--input_root", type=str,
                        default=r"D:\VideoSuperResolution\BasicVSR_PlusPlus\data\reds\input_videos",
                        help="Root directory containing sequence subfolders")
    parser.add_argument("--output_root", type=str,
                        default=r"D:\VideoSuperResolution\ESRGAN\results",
                        help="Output root directory")
    parser.add_argument("--exe", type=str,
                        default=r"D:\VideoSuperResolution\ESRGAN\realesrgan-ncnn-vulkan.exe",
                        help="Path to Real-ESRGAN exe")
    parser.add_argument("--scale", type=int, default=4, choices=[2,4])
    parser.add_argument("--model", type=str, default="realesrgan-x4plus",
                        help="ESRGAN model name (e.g., realesrgan-x4plus, realesrgan-animevideov3)")
    parser.add_argument("--format", type=str, default="png")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()
    # Modify to match the expected folder structure: input_root should contain subfolders like '000', '001', etc.

    batch_upscale(
        input_root=args.input_root,
        output_root=args.output_root,
        exe_path=args.exe,
        scale=args.scale,
        model=args.model,
        frame_format=args.format,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
