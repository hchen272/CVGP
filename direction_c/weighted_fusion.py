#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Weighted average fusion between BasicVSR++ and Direction C outputs.

Usage example:
    python weighted_fusion.py --alpha 0.5
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def load_frames(folder_path):
    """Load all PNG frames from folder, sorted naturally."""
    folder = Path(folder_path)
    frame_paths = sorted(folder.glob("*.png"),
                         key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    frames = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Cannot read {p}")
        frames.append(img)
    return frames, frame_paths


def weighted_fusion(basicvsr_frames, dir_c_frames, alpha):
    """
    Pixel-wise weighted average: result = alpha * basicvsr + (1-alpha) * directionc
    alpha: weight for BasicVSR++ (0~1). alpha=1 -> pure BasicVSR++, alpha=0 -> pure Direction C.
    """
    assert len(basicvsr_frames) == len(dir_c_frames)
    fused = []
    for b, d in zip(basicvsr_frames, dir_c_frames):
        # Ensure same size (should be)
        if b.shape != d.shape:
            d = cv2.resize(d, (b.shape[1], b.shape[0]))
        fused_frame = cv2.addWeighted(b, alpha, d, 1 - alpha, 0)
        fused.append(fused_frame)
    return fused


def main():
    parser = argparse.ArgumentParser(description="Weighted fusion of BasicVSR++ and Direction C")
    parser.add_argument("--basicvsr_root", type=str,
                        default=r"D:\VideoSuperResolution\BasicVSR_PlusPlus\results\fp32",
                        help="Root folder containing BasicVSR++ results (subfolders like '000')")
    parser.add_argument("--directionc_root", type=str,
                        default=r"D:\VideoSuperResolution\direction_c\results\direction_c_variants\threshold_0.3",
                        help="Root folder containing Direction C results (same subfolder structure)")
    parser.add_argument("--output_root", type=str,
                        default=r"D:\VideoSuperResolution\BasicVSR_PlusPlus\results\weighted_avg",
                        help="Output root (will create subfolders for each video and alpha)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for BasicVSR++ (0~1). Default 0.5")
    parser.add_argument("--video_names", type=str, nargs="+", default=None,
                        help="List of video names (e.g., 000 011). If not given, auto-detect.")
    args = parser.parse_args()

    basic_root = Path(args.basicvsr_root)
    dir_c_root = Path(args.directionc_root)
    out_root = Path(args.output_root) / f"alpha_{args.alpha:.2f}".replace('.', '_')
    out_root.mkdir(parents=True, exist_ok=True)

    # Get video names
    if args.video_names is None:
        video_names = [d.name for d in basic_root.iterdir() if d.is_dir()]
    else:
        video_names = args.video_names

    print(f"Processing {len(video_names)} videos with alpha={args.alpha}")
    for video in tqdm(video_names, desc="Videos"):
        basic_path = basic_root / video
        dir_c_path = dir_c_root / video
        if not basic_path.exists() or not dir_c_path.exists():
            print(f"  Skipping {video}: one of roots missing")
            continue
        try:
            b_frames, _ = load_frames(basic_path)
            d_frames, _ = load_frames(dir_c_path)
            if len(b_frames) != len(d_frames):
                print(f"  Frame count mismatch for {video}, truncating to min")
                min_len = min(len(b_frames), len(d_frames))
                b_frames = b_frames[:min_len]
                d_frames = d_frames[:min_len]
            fused = weighted_fusion(b_frames, d_frames, args.alpha)
            out_video_dir = out_root / video
            out_video_dir.mkdir(exist_ok=True)
            for idx, frame in enumerate(fused):
                out_path = out_video_dir / f"{idx:08d}.png"
                cv2.imwrite(str(out_path), frame)
        except Exception as e:
            print(f"  Error processing {video}: {e}")

    print(f"Done. Results saved to {out_root}")


if __name__ == "__main__":
    main()