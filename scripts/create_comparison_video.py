#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a side-by-side comparison video:
Left: Low-resolution input (upscaled to match SR size)
Middle: FP32 super-resolved
Right: FP16 super-resolved
"""

import cv2
import numpy as np
from pathlib import Path

def resize_to_match(frame, target_h, target_w):
    """Resize frame to target dimensions."""
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

def main():
    # Paths (update these to your actual file locations)
    lq_video = Path("data/lq.mp4")                     # Low-resolution input
    sr_fp32_video = Path("results/eval/fp32/output.mp4")
    sr_fp16_video = Path("results/eval/fp16/output.mp4")
    output_video = Path("results/eval/comparison.mp4")

    # Check existence
    for p in [lq_video, sr_fp32_video, sr_fp16_video]:
        if not p.exists():
            print(f"Error: {p} not found.")
            return

    # Open videos
    cap_lq = cv2.VideoCapture(str(lq_video))
    cap_fp32 = cv2.VideoCapture(str(sr_fp32_video))
    cap_fp16 = cv2.VideoCapture(str(sr_fp16_video))

    # Get FPS and total frames (assume all same)
    fps = cap_lq.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_lq.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get output dimensions from FP32 video
    ret, frame_fp32 = cap_fp32.read()
    if not ret:
        print("Cannot read FP32 video.")
        return
    h_sr, w_sr = frame_fp32.shape[:2]
    cap_fp32.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (w_sr * 3, h_sr))

    for _ in range(total_frames):
        ret_lq, frame_lq = cap_lq.read()
        ret_fp32, frame_fp32 = cap_fp32.read()
        ret_fp16, frame_fp16 = cap_fp16.read()
        if not (ret_lq and ret_fp32 and ret_fp16):
            break

        # Upscale LQ to match SR size
        frame_lq_up = resize_to_match(frame_lq, h_sr, w_sr)

        # Concatenate horizontally
        comparison = np.hstack([frame_lq_up, frame_fp32, frame_fp16])

        # Add text labels (optional)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Low-res Input", (50, 50), font, 1, (255,255,255), 2)
        cv2.putText(comparison, "FP32 Super-res", (w_sr + 50, 50), font, 1, (255,255,255), 2)
        cv2.putText(comparison, "FP16 Super-res", (2*w_sr + 50, 50), font, 1, (255,255,255), 2)

        out.write(comparison)

    cap_lq.release()
    cap_fp32.release()
    cap_fp16.release()
    out.release()
    print(f"Comparison video saved to {output_video}")

if __name__ == '__main__':
    main()