#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified evaluation for FP32 vs FP16 video super-resolution.
Includes peak GPU memory measurement.
"""
# to run: python scripts/evaluate_fp32_fp16.py data/lq.mp4 --gt_video data/gt.mp4 --output_dir ./results/eval --keep_temp

import argparse
import json
import sys
import time
import shutil
from pathlib import Path
import cv2
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Add inference_optimize to path so that utils.model_loader and utils.inference can be imported
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'inference_optimize'))
from utils.model_loader import load_model
from utils.inference import profile_model_memory

# ----------------------------------------------------------------------
# Frame extraction and video merging (same as original script)
# ----------------------------------------------------------------------
def extract_frames(video_path, output_dir, target_fps=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_dir.mkdir(parents=True, exist_ok=True)
    if target_fps is not None and target_fps < original_fps:
        interval = int(round(original_fps / target_fps))
    else:
        interval = 1
        target_fps = original_fps
    saved = 0
    pbar = tqdm(total=total_original, desc="Extracting frames")
    for idx in range(total_original):
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            cv2.imwrite(str(output_dir / f"{saved:08d}.png"), frame)
            saved += 1
        pbar.update(1)
    cap.release()
    pbar.close()
    print(f"Extracted {saved} frames (original {total_original}, interval {interval})")
    return original_fps, saved

def merge_frames_to_video(frame_dir, video_path, fps):
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob("*.png"), key=lambda x: int(x.stem))
    if not frames:
        raise RuntimeError("No frames found")
    h, w = cv2.imread(str(frames[0])).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    for f in tqdm(frames, desc="Merging video"):
        out.write(cv2.imread(str(f)))
    out.release()

# ----------------------------------------------------------------------
# Super-resolution processing for a given precision
# ----------------------------------------------------------------------
def process_video_with_precision(input_video, output_video, config_path, checkpoint_path,
                                 precision='fp32', window_size=8, device='cuda:0',
                                 temp_dir=None):
    """
    Run the entire pipeline: extract frames -> super-resolve -> merge.
    Returns a dict with timing and memory info (including peak memory).
    """
    temp_root = Path(temp_dir) if temp_dir else Path("./temp_" + precision)
    input_frames_dir = temp_root / "input_frames"
    output_frames_dir = temp_root / "output_frames"
    input_frames_dir.mkdir(parents=True, exist_ok=True)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    # Record start time
    start_total = time.time()

    # Step 1: extract frames
    start_extract = time.time()
    original_fps, total_frames = extract_frames(input_video, input_frames_dir)
    extract_time = time.time() - start_extract

    # Load model in FP32, then convert if needed
    print(f"Loading model for {precision.upper()}...")
    model = load_model(config_path, checkpoint_path, device=device)
    if precision == 'fp16':
        model = model.half()
    model.eval()

    # Prepare sliding window processing
    frame_paths = sorted(input_frames_dir.glob("*.png"))
    num_windows = (total_frames + window_size - 1) // window_size
    print(f"Processing {total_frames} frames with window size {window_size} ({num_windows} windows)")

    # Temporary directory for per-window outputs
    temp_out_root = output_frames_dir / "_temp_windows"
    temp_out_root.mkdir(exist_ok=True)

    # Measure peak memory during first window (representative)
    peak_memory_mb = 0
    start_infer = time.time()
    pbar = tqdm(total=total_frames, desc=f"{precision.upper()} inference", unit="frame")
    for start_idx in range(0, total_frames, window_size):
        end_idx = min(start_idx + window_size, total_frames)
        # Prepare input tensor
        batch = []
        for i in range(start_idx, end_idx):
            img = cv2.imread(str(frame_paths[i]))
            img = torch.from_numpy(img).permute(2,0,1).float() / 255.0  # [C,H,W]
            batch.append(img)
        input_tensor = torch.stack(batch).unsqueeze(0).to(device)  # [1, T, C, H, W]
        if precision == 'fp16':
            input_tensor = input_tensor.half()

        # Measure peak memory for the first window (if not already done)
        if start_idx == 0:
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                output_dict = model.forward_test(input_tensor)
                output_tensor = output_dict['output']
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            with torch.no_grad():
                output_dict = model.forward_test(input_tensor)
                output_tensor = output_dict['output']

        # Save output frames
        window_out_dir = temp_out_root / f"out_{start_idx:06d}_{end_idx:06d}"
        window_out_dir.mkdir(exist_ok=True)
        for j in range(output_tensor.size(1)):
            out_img = output_tensor[0, j].cpu().float().permute(1,2,0).numpy() * 255.0
            out_img = np.clip(out_img, 0, 255).astype(np.uint8)
            cv2.imwrite(str(window_out_dir / f"{j:08d}.png"), out_img)

        pbar.update(end_idx - start_idx)
        torch.cuda.empty_cache()
    pbar.close()
    infer_time = time.time() - start_infer

    # Merge window outputs
    start_merge = time.time()
    all_frames = []
    for win_dir in sorted(temp_out_root.glob("out_*")):
        frames = sorted(win_dir.glob("*.png"), key=lambda x: int(x.stem))
        all_frames.extend(frames)
    for new_idx, src in enumerate(all_frames):
        dst = output_frames_dir / f"{new_idx:08d}.png"
        shutil.copy2(src, dst)
    shutil.rmtree(temp_out_root)
    merge_time = time.time() - start_merge

    # Merge to video
    merge_frames_to_video(output_frames_dir, output_video, original_fps)

    total_time = time.time() - start_total

    # Cleanup temporary directories
    shutil.rmtree(temp_root)

    return {
        'precision': precision,
        'total_frames': total_frames,
        'extract_time_s': extract_time,
        'inference_time_s': infer_time,
        'merge_time_s': merge_time,
        'total_time_s': total_time,
        'fps_video': total_frames / total_time,
        'fps_inference': total_frames / infer_time if infer_time > 0 else 0,
        'peak_memory_mb': peak_memory_mb,   # <-- 新增峰值显存
    }

# ----------------------------------------------------------------------
# Quality evaluation against ground truth
# ----------------------------------------------------------------------
def evaluate_quality(sr_video_path, gt_video_path):
    """
    Compute PSNR and SSIM between super-resolved video and ground truth.
    Both videos must have the same number of frames and resolution.
    """
    cap_sr = cv2.VideoCapture(str(sr_video_path))
    cap_gt = cv2.VideoCapture(str(gt_video_path))
    if not cap_sr.isOpened() or not cap_gt.isOpened():
        raise RuntimeError("Cannot open one of the videos for evaluation")

    n_frames = int(min(cap_sr.get(cv2.CAP_PROP_FRAME_COUNT),
                       cap_gt.get(cv2.CAP_PROP_FRAME_COUNT)))
    psnr_list = []
    ssim_list = []
    for _ in tqdm(range(n_frames), desc="Evaluating frames"):
        ret_sr, frame_sr = cap_sr.read()
        ret_gt, frame_gt = cap_gt.read()
        if not ret_sr or not ret_gt:
            break
        # Resize SR to match GT if necessary (should be same size)
        if frame_sr.shape != frame_gt.shape:
            frame_sr = cv2.resize(frame_sr, (frame_gt.shape[1], frame_gt.shape[0]))
        psnr = peak_signal_noise_ratio(frame_gt, frame_sr, data_range=255)
        ssim = structural_similarity(frame_gt, frame_sr, multichannel=True,
                                     data_range=255, channel_axis=2)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
    cap_sr.release()
    cap_gt.release()
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    return {'psnr': avg_psnr, 'ssim': avg_ssim}

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare FP32 vs FP16 video super-resolution")
    parser.add_argument("input_lq", type=str, help="Low-resolution input video")
    parser.add_argument("--gt_video", type=str, default=None, help="Ground truth video (optional)")
    parser.add_argument("--config", type=str, default="configs/basicvsr_plusplus_reds4.py")
    parser.add_argument("--checkpoint", type=str, default="chkpts/basicvsr_plusplus_reds4.pth")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./results/evaluation",
                        help="Root directory for storing outputs (subdirs fp32, fp16)")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary files")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    results = {}

    # Process FP32
    print("\n" + "="*60)
    print("RUNNING FP32 (BASELINE)")
    print("="*60)
    fp32_out_video = out_root / "fp32" / "output.mp4"
    fp32_temp = out_root / "fp32" / "temp" if args.keep_temp else None
    stats_fp32 = process_video_with_precision(
        args.input_lq, fp32_out_video, args.config, args.checkpoint,
        precision='fp32', window_size=args.window_size, device=device,
        temp_dir=fp32_temp
    )
    results['fp32'] = stats_fp32

    # Process FP16
    print("\n" + "="*60)
    print("RUNNING FP16 (OPTIMIZED)")
    print("="*60)
    fp16_out_video = out_root / "fp16" / "output.mp4"
    fp16_temp = out_root / "fp16" / "temp" if args.keep_temp else None
    stats_fp16 = process_video_with_precision(
        args.input_lq, fp16_out_video, args.config, args.checkpoint,
        precision='fp16', window_size=args.window_size, device=device,
        temp_dir=fp16_temp
    )
    results['fp16'] = stats_fp16

    # Quality evaluation if GT provided
    if args.gt_video:
        print("\n" + "="*60)
        print("QUALITY EVALUATION AGAINST GROUND TRUTH")
        print("="*60)
        qual_fp32 = evaluate_quality(fp32_out_video, args.gt_video)
        qual_fp16 = evaluate_quality(fp16_out_video, args.gt_video)
        results['fp32']['quality'] = qual_fp32
        results['fp16']['quality'] = qual_fp16

    # Print comparison table
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Metric':<25} {'FP32':<15} {'FP16':<15} {'Speedup/Change':<15}")
    print("-"*70)
    fps_base = results['fp32']['fps_video']
    fps_opt = results['fp16']['fps_video']
    print(f"{'End-to-end FPS':<25} {fps_base:<15.2f} {fps_opt:<15.2f} {fps_opt/fps_base:<15.2f}x")
    time_base = results['fp32']['inference_time_s']
    time_opt = results['fp16']['inference_time_s']
    print(f"{'Inference time (s)':<25} {time_base:<15.2f} {time_opt:<15.2f} {time_base/time_opt:<15.2f}x")
    # 新增峰值显存打印
    mem_base = results['fp32']['peak_memory_mb']
    mem_opt = results['fp16']['peak_memory_mb']
    print(f"{'Peak GPU Memory (MB)':<25} {mem_base:<15.2f} {mem_opt:<15.2f} {mem_base/mem_opt:<15.2f}x reduction")
    if 'quality' in results['fp32']:
        psnr_base = results['fp32']['quality']['psnr']
        psnr_opt = results['fp16']['quality']['psnr']
        print(f"{'PSNR (dB)':<25} {psnr_base:<15.2f} {psnr_opt:<15.2f} {psnr_opt - psnr_base:<+15.2f}")
        ssim_base = results['fp32']['quality']['ssim']
        ssim_opt = results['fp16']['quality']['ssim']
        print(f"{'SSIM':<25} {ssim_base:<15.4f} {ssim_opt:<15.4f} {ssim_opt - ssim_base:<+15.4f}")
    print("="*60)

    # Save results to JSON
    report_path = out_root / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed report saved to {report_path}")

if __name__ == '__main__':
    main()