#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a side-by-side comparison video with 2 rows:
Row1: LR (upscaled) + Method1 + Method2
Row2: Method3 + Method4 + Method5 (if exists)
"""

import cv2
import numpy as np
from pathlib import Path

def resize_to_match(frame, target_h, target_w):
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

def main():
    # ========== CONFIGURATION ==========
    video_name = "02"                     # Name of the video
    scale = 2
    # Methods in order (will be arranged row-wise)
    methods = [
        "bicubic_x2",
        "lanczos_x2",
        "bicubic_x2_temporal_avg",
        "bicubic_x2_temporal_avg_unsharp",
        "srcnn_x2"
    ]
    labels = [
        "Bicubic",
        "Lanczos",
        "Temporal Avg",
        "Temporal Avg + Unsharp",
        "SRCNN"
    ]
    # ====================================

    # Paths
    lq_video = Path(f"data/input_videos/{video_name}.mp4")
    output_dir = Path(f"outputs/{video_name}")
    output_video = output_dir / "comparison_2x3.mp4"

    if not lq_video.exists():
        print(f"Error: {lq_video} not found.")
        return
    if not output_dir.exists():
        print(f"Error: {output_dir} not found.")
        return

    # Collect existing method videos
    method_paths = []
    valid_labels = []
    for m, lbl in zip(methods, labels):
        p = output_dir / f"{m}.mp4"
        if p.exists():
            method_paths.append(p)
            valid_labels.append(lbl)
        else:
            print(f"Warning: {p} not found, skipping.")
    if not method_paths:
        print("No method videos found.")
        return

    # Open captures
    cap_lq = cv2.VideoCapture(str(lq_video))
    caps_method = [cv2.VideoCapture(str(p)) for p in method_paths]

    # Get properties
    fps = cap_lq.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_lq.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get HR size from first method
    ret, frame0 = caps_method[0].read()
    if not ret:
        print("Cannot read first method video.")
        return
    h_hr, w_hr = frame0.shape[:2]
    caps_method[0].set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Determine layout
    num_methods = len(method_paths)
    # Row1: LR + first 2 methods (if available)
    row1_methods = method_paths[:2]
    row1_labels = valid_labels[:2]
    # Row2: remaining methods (max 3)
    row2_methods = method_paths[2:]
    row2_labels = valid_labels[2:]

    # Output dimensions: 3 columns, each of width w_hr, height h_hr
    # Row1: 3 columns (LR, method1, method2)
    # Row2: 3 columns (method3, method4, method5) or fewer (fill with black if needed)
    cols = 3
    out_width = w_hr * cols
    out_height = h_hr * 2  # three rows

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (out_width, out_height))
    if not out.isOpened():
        print("Error: Could not open VideoWriter. Try a different codec.")
        return

    # Helper to create a blank black frame
    def blank_frame():
        return np.zeros((h_hr, w_hr, 3), dtype=np.uint8)

    for frame_idx in range(total_frames):
        # Read LQ and upscale
        ret_lq, frame_lq = cap_lq.read()
        if not ret_lq:
            break
        frame_lq_up = resize_to_match(frame_lq, h_hr, w_hr)

        # Read method frames for row1
        row1_frames = [frame_lq_up]
        row1_caps = caps_method[:2]
        for cap in row1_caps:
            ret, f = cap.read()
            if not ret:
                f = blank_frame()
            row1_frames.append(f)

        # Read method frames for row2
        row2_frames = []
        row2_caps = caps_method[2:]
        for cap in row2_caps:
            ret, f = cap.read()
            if not ret:
                f = blank_frame()
            row2_frames.append(f)
        # Pad row2 to 3 columns if necessary
        while len(row2_frames) < cols:
            row2_frames.append(blank_frame())

        # Build rows
        row1_img = np.hstack(row1_frames)
        row2_img = np.hstack(row2_frames)
        comparison = np.vstack([row1_img, row2_img])

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Row1 labels
        cv2.putText(comparison, "LR (Bicubic upscaled)", (50, 50), font, 0.8, (255,255,255), 2)
        for i, lbl in enumerate(row1_labels):
            x_offset = w_hr * (i+1) + 50
            cv2.putText(comparison, lbl, (x_offset, 50), font, 0.8, (255,255,255), 2)
        # Row2 labels (y = h_hr + 50)
        for i, lbl in enumerate(row2_labels):
            x_offset = w_hr * i + 50
            cv2.putText(comparison, lbl, (x_offset, h_hr + 50), font, 0.8, (255,255,255), 2)

        out.write(comparison)

        if (frame_idx+1) % 50 == 0:
            print(f"Processed {frame_idx+1}/{total_frames} frames")

    # Release
    cap_lq.release()
    for cap in caps_method:
        cap.release()
    out.release()
    print(f"Comparison video saved to {output_video}")

if __name__ == '__main__':
    main()