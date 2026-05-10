#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video Super-Resolution Pipeline using BasicVSR++
Totally based on OpenCV implement video frame splitting and merging without ffmpeg
No longer directly import mmedit to avoid DLL errors
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
import cv2
from tqdm import tqdm
import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def extract_frames_opencv(video_path, output_dir, target_fps=None):
    """
    Extract frames from video using OpenCV and save them as PNG images.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if target_fps is not None and target_fps < original_fps:
        sample_interval = int(round(original_fps / target_fps))
    else:
        sample_interval = 1
        target_fps = original_fps
    
    frame_idx = 0
    saved_idx = 0
    pbar = tqdm(total=total_frames_original, desc="Extracting video frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            out_path = output_dir / f"{saved_idx:08d}.png"
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1
        frame_idx += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    print(f"Extraction completed: original {total_frames_original} frames -> {saved_idx} frames (sampling interval {sample_interval})")
    return original_fps, saved_idx

def rename_output_files(output_dir):
    """Rename output frames from sliding windows to continuous sequence."""
    output_dir = Path(output_dir)
    frame_files = sorted(output_dir.glob("*.png"), key=lambda x: int(x.stem))
    for new_idx, file_path in enumerate(frame_files):
        new_name = f"{new_idx:08d}.png"
        file_path.rename(output_dir / new_name)
    print(f"Output files renumbered, total {len(frame_files)} frames")

def run_super_resolution(config_path, checkpoint_path, input_dir, output_dir, device_id=0, window_size=30):
    """
    Call BasicVSR++ official inference script in sliding window manner.
    Each window outputs to a separate subdirectory, finally merge all.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(input_dir.glob("*.png"))
    total_frames = len(frame_paths)
    if total_frames == 0:
        print("Error: No PNG images found in input directory")
        return

    num_windows = (total_frames + window_size - 1) // window_size
    print(f"Total frames: {total_frames}, window size: {window_size}, total windows: {num_windows}")

    # Create temporary directory to store outputs of all windows
    temp_output_root = output_dir / "_temp_windows"
    if temp_output_root.exists():
        shutil.rmtree(temp_output_root)
    temp_output_root.mkdir(parents=True)

    demo_script = Path(__file__).parent.parent / "demo" / "restoration_video_demo.py"
    if not demo_script.exists():
        raise RuntimeError(f"Cannot find inference script: {demo_script}")

    # Progress bar
    pbar = tqdm(total=total_frames, desc="Super-resolution progress", unit="frame", ncols=80)

    for start_idx in range(0, total_frames, window_size):
        end_idx = min(start_idx + window_size, total_frames)
        current_window_path = input_dir / f"window_{start_idx}_{end_idx}"
        current_window_path.mkdir(exist_ok=True)

        # Copy frames of current window
        for i, frame_path in enumerate(frame_paths[start_idx:end_idx]):
            target_path = current_window_path / f"{i:08d}.png"
            shutil.copy2(frame_path, target_path)

        # Create independent output subdirectory for this window
        window_out_dir = temp_output_root / f"out_{start_idx:06d}_{end_idx:06d}"
        window_out_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable, str(demo_script),
            str(config_path), str(checkpoint_path),
            str(current_window_path), str(window_out_dir),
            "--device", str(device_id),
            "--max-seq-len", str(window_size)
        ]

        # Run subprocess silently to avoid cluttering progress bar
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"\nWindow {start_idx//window_size + 1} processing failed, error code: {e.returncode}")
            raise

        # Clean up temporary input directory
        shutil.rmtree(current_window_path)
        torch.cuda.empty_cache()

        # Update progress bar
        pbar.update(end_idx - start_idx)
        pbar.set_postfix_str(f"Window {start_idx//window_size + 1}/{num_windows}")

    pbar.close()

    # Merge all window output files and renumber
    print("\nMerging output frames...")
    all_frames = []
    # Sort by window index to ensure correct order
    for win_dir in sorted(temp_output_root.glob("out_*")):
        frames = sorted(win_dir.glob("*.png"), key=lambda x: int(x.stem))
        all_frames.extend(frames)

    if len(all_frames) != total_frames:
        print(f"Warning: expected {total_frames} frames, got {len(all_frames)} frames")

    # Copy to final output directory and rename
    for new_idx, src_path in enumerate(all_frames):
        dst_path = output_dir / f"{new_idx:08d}.png"
        shutil.copy2(src_path, dst_path)

    # Delete temporary directory
    shutil.rmtree(temp_output_root)
    print(f"Merging completed, total {len(all_frames)} frames saved to {output_dir}")

def merge_frames_to_video_opencv(frame_dir, output_video_path, fps):
    """
    Use OpenCV to merge frame sequence into a video.
    """
    frame_dir = Path(frame_dir)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get dimensions from first frame
    first_frame_path = list(frame_dir.glob("*.png"))[0]
    frame = cv2.imread(str(first_frame_path))
    h, w = frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))
    
    frame_files = sorted(frame_dir.glob("*.png"), key=lambda x: int(x.stem))
    for fpath in tqdm(frame_files, desc="Merging video"):
        img = cv2.imread(str(fpath))
        out.write(img)
    out.release()
    print(f"Video saved: {output_video_path}")

def cleanup_dirs(dirs):
    for d in dirs:
        if d and Path(d).exists():
            shutil.rmtree(d)
            print(f"Removed temporary directory: {d}")

def main():
    parser = argparse.ArgumentParser(description="Video Super-Resolution Pipeline (pure OpenCV, no ffmpeg required)")
    parser.add_argument("input_video", type=str, help="Path to input video")
    parser.add_argument("output_video", type=str, help="Path to output video")
    parser.add_argument("--config", type=str, default="configs/basicvsr_plusplus_reds4.py",
                        help="Model config file path")
    parser.add_argument("--checkpoint", type=str, default="chkpts/basicvsr_plusplus_reds4.pth",
                        help="Model checkpoint file path")
    parser.add_argument("--target_fps", type=int, default=None,
                        help="Target output video FPS (if not specified, use original FPS)")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--temp_dir", type=str, default="./temp_sr",
                        help="Temporary directory for intermediate frames")
    parser.add_argument("--keep_temp", action="store_true",
                        help="Keep temporary directory (deleted by default)")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    input_video = Path(args.input_video)
    output_video = Path(args.output_video)
    config_path = base_dir / args.config
    checkpoint_path = base_dir / args.checkpoint
    temp_root = Path(args.temp_dir)
    input_frames_dir = temp_root / "input_frames"
    output_frames_dir = temp_root / "output_frames"
    
    if not input_video.exists():
        print(f"Error: Input video does not exist: {input_video}")
        sys.exit(1)
    if not config_path.exists():
        print(f"Error: Config file does not exist: {config_path}")
        sys.exit(1)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file does not exist: {checkpoint_path}")
        sys.exit(1)
    
    try:
        original_fps, total_frames = extract_frames_opencv(
            input_video, input_frames_dir, args.target_fps
        )
        fps = args.target_fps if args.target_fps else original_fps
        
        run_super_resolution(config_path, checkpoint_path,
                             input_frames_dir, output_frames_dir,
                             args.device, window_size=5)
        
        merge_frames_to_video_opencv(output_frames_dir, output_video, fps)
        
        print(f"\n✅ Success! Super-resolved video saved to: {output_video}")
    
    finally:
        if not args.keep_temp:
            cleanup_dirs([input_frames_dir, output_frames_dir])
            if temp_root.exists() and not any(temp_root.iterdir()):
                temp_root.rmdir()
        else:
            print(f"Temporary directory kept at: {temp_root}")

if __name__ == "__main__":
    main()