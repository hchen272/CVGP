#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main pipeline for Part 1: Baseline methods.
Supports both MP4 videos and image sequences (e.g., REDS dataset).
Outputs can be saved as MP4 or PNG sequences.
"""

import os
import sys
import time
import json
import re
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.video_io import write_video_frames
from part1.spatial_upsample import bicubic_upsample, lanczos_upsample
from part1.temporal_average import apply_temporal_average_to_video
from part1.unsharp_mask import apply_unsharp_mask_to_video
from part1.srcnn_inference import load_srcnn_model, srcnn_upsample_frame


def read_video_frames(video_path):
    """Read frames from an MP4 video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def read_image_sequence(folder_path, ext='png'):
    """
    Read a sequence of images from a folder (natural order).
    Returns list of frames (BGR uint8).
    """
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(ext)]
    # Natural sort (e.g., 00000001.png, 00000002.png)
    def natural_sort_key(s):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]
    files.sort(key=natural_sort_key)

    frames = []
    for f in files:
        img_path = os.path.join(folder_path, f)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: cannot read {img_path}")
            continue
        frames.append(frame)
    if not frames:
        raise ValueError(f"No images found in {folder_path}")
    return frames


def load_frames(path):
    """
    Automatically detect whether path is a video file or a folder containing image sequence.
    Returns list of frames (BGR uint8).
    """
    if os.path.isfile(path):
        return read_video_frames(path)
    elif os.path.isdir(path):
        for ext in ['png', 'jpg', 'jpeg', 'bmp']:
            if any(f.lower().endswith(ext) for f in os.listdir(path)):
                return read_image_sequence(path, ext)
        raise ValueError(f"No valid image files found in {path}")
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")


def save_frames_as_images(frames, output_folder, base_name="%08d", ext="png"):
    """
    Save a list of frames as image sequence in output_folder.
    frames: list of numpy arrays (BGR uint8)
    """
    os.makedirs(output_folder, exist_ok=True)
    for idx, frame in enumerate(frames):
        filename = f"{base_name % idx}.{ext}"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, frame)


def process_single_video(input_path, output_dir, scale=2, use_srcnn=True, output_format='png'):
    """
    Process a single video / image sequence: generate all baseline outputs.
    output_format: 'png' (save as image sequence) or 'mp4' (save as single video file).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load low-resolution frames
    print("Reading low-resolution input...")
    lr_frames = load_frames(input_path)
    if not lr_frames:
        print("  No frames read. Skipping.")
        return
    n_frames = len(lr_frames)
    timing_results = {}

    # Helper to save outputs (sequence or video)
    def save_output(frames, method_name):
        if output_format == 'png':
            out_folder = os.path.join(output_dir, method_name)
            save_frames_as_images(frames, out_folder, base_name="%08d", ext="png")
            print(f"  Saved {len(frames)} frames to {out_folder}")
        else:
            out_video = os.path.join(output_dir, f"{method_name}.mp4")
            write_video_frames(frames, out_video)
            print(f"  Saved video to {out_video}")

    # ------------------------------------------------------------------
    # 1. Bicubic pipeline
    # ------------------------------------------------------------------
    print("\n--- Bicubic pipeline ---")
    start = time.time()
    bicubic_frames = [bicubic_upsample(f, scale) for f in lr_frames]
    elapsed = time.time() - start
    save_output(bicubic_frames, f"bicubic_x{scale}")
    timing_results[f"bicubic_x{scale}"] = {
        "inference_time_s": elapsed,
        "inference_fps": n_frames / elapsed if elapsed > 0 else 0
    }

    start = time.time()
    bicubic_temporal = apply_temporal_average_to_video(bicubic_frames)
    elapsed = time.time() - start
    save_output(bicubic_temporal, f"bicubic_x{scale}_temporal_avg")
    timing_results[f"bicubic_x{scale}_temporal_avg"] = {
        "inference_time_s": elapsed,
        "inference_fps": n_frames / elapsed if elapsed > 0 else 0
    }

    start = time.time()
    bicubic_temporal_unsharp = apply_unsharp_mask_to_video(bicubic_temporal)
    elapsed = time.time() - start
    save_output(bicubic_temporal_unsharp, f"bicubic_x{scale}_temporal_avg_unsharp")
    timing_results[f"bicubic_x{scale}_temporal_avg_unsharp"] = {
        "inference_time_s": elapsed,
        "inference_fps": n_frames / elapsed if elapsed > 0 else 0
    }

    # ------------------------------------------------------------------
    # 2. Lanczos pipeline
    # ------------------------------------------------------------------
    print("\n--- Lanczos pipeline ---")
    start = time.time()
    lanczos_frames = [lanczos_upsample(f, scale) for f in lr_frames]
    elapsed = time.time() - start
    save_output(lanczos_frames, f"lanczos_x{scale}")
    timing_results[f"lanczos_x{scale}"] = {
        "inference_time_s": elapsed,
        "inference_fps": n_frames / elapsed if elapsed > 0 else 0
    }

    start = time.time()
    lanczos_temporal = apply_temporal_average_to_video(lanczos_frames)
    elapsed = time.time() - start
    save_output(lanczos_temporal, f"lanczos_x{scale}_temporal_avg")
    timing_results[f"lanczos_x{scale}_temporal_avg"] = {
        "inference_time_s": elapsed,
        "inference_fps": n_frames / elapsed if elapsed > 0 else 0
    }

    start = time.time()
    lanczos_temporal_unsharp = apply_unsharp_mask_to_video(lanczos_temporal)
    elapsed = time.time() - start
    save_output(lanczos_temporal_unsharp, f"lanczos_x{scale}_temporal_avg_unsharp")
    timing_results[f"lanczos_x{scale}_temporal_avg_unsharp"] = {
        "inference_time_s": elapsed,
        "inference_fps": n_frames / elapsed if elapsed > 0 else 0
    }

    # ------------------------------------------------------------------
    # 3. SRCNN pipeline (if enabled)
    # ------------------------------------------------------------------
    if use_srcnn:
        print("\n--- SRCNN pipeline ---")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent

        weights_path = project_root / 'models' / 'srcnn_x4.pth'
        if os.path.exists(weights_path):
            model = load_srcnn_model(weights_path, device=device)
            start = time.time()
            srcnn_frames = []
            for frame in tqdm(lr_frames, desc="  SRCNN inference"):
                hr_frame = srcnn_upsample_frame(frame, model, device, scale)
                srcnn_frames.append(hr_frame)
            elapsed = time.time() - start
            save_output(srcnn_frames, f"srcnn_x{scale}")
            timing_results[f"srcnn_x{scale}"] = {
                "inference_time_s": elapsed,
                "inference_fps": n_frames / elapsed if elapsed > 0 else 0
            }
        else:
            print(f"  SRCNN weights not found at {weights_path}. Skipping SRCNN.")

    # Save timing results
    timing_path = os.path.join(output_dir, "timing_results.json")
    with open(timing_path, 'w') as f:
        json.dump(timing_results, f, indent=4)
    print(f"Timing results saved to {timing_path}")

    print(f"\nAll outputs saved in {output_dir}")


def main():
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_dir = os.path.normpath(os.path.join(script_dir, "../.."))  # D:\VideoSuperResolution
    input_dir = os.path.join(base_dir, "BasicVSR_PlusPlus", "data", "reds", "input_videos")
    output_root = os.path.join(base_dir, "BasicVSR_PlusPlus", "outputs")
    scale = 4
    use_srcnn = True
    output_format = 'png'                    # 'png' for image sequences, 'mp4' for video files

    # List all items in input_dir
    items = os.listdir(input_dir)
    # Determine if items are video files or folders
    video_names = []
    for item in items:
        item_path = os.path.join(input_dir, item)
        if os.path.isfile(item_path) and item.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # MP4 file: treat video name as filename without extension
            video_names.append(os.path.splitext(item)[0])
        elif os.path.isdir(item_path):
            # Folder: treat folder name as video name
            video_names.append(item)
        else:
            print(f"Skipping unknown item: {item}")

    if not video_names:
        print(f"No valid inputs found in {input_dir}")
        return

    for video_name in video_names:
        # Determine exact input path
        input_path = os.path.join(input_dir, video_name)
        if not os.path.exists(input_path):
            # Try with .mp4 extension
            input_path = os.path.join(input_dir, video_name + ".mp4")
            if not os.path.exists(input_path):
                print(f"Input not found for {video_name}, skipping.")
                continue

        output_dir = os.path.join(output_root, video_name)
        print(f"\n===== Processing video: {video_name} =====")
        process_single_video(input_path, output_dir, scale, use_srcnn, output_format)

    print("\nAll videos processed.")


if __name__ == "__main__":
    main()