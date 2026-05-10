#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate multiple confidence thresholds for Direction C and compare with BasicVSR++ baseline.

Usage:
    python evaluate_thresholds.py
"""

import os
import sys
import json
import re
import cv2
import numpy as np
from tqdm import tqdm
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import uncertainty (if needed) - but not required
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional FID
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not installed. FID will be skipped.")

lpips_fn = lpips.LPIPS(net='alex', verbose=False)

# ----------------------------- Helper functions (same as evaluation_full) -----------------------------
def resize_frame(frame, target_h, target_w):
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

def compute_lpips(img1, img2):
    try:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t1 = torch.from_numpy(img1_rgb).permute(2,0,1).unsqueeze(0)
        t2 = torch.from_numpy(img2_rgb).permute(2,0,1).unsqueeze(0)
        with torch.no_grad():
            d = lpips_fn(t1, t2)
        return d.item()
    except Exception as e:
        print(f"LPIPS error: {e}")
        return 1.0

def compute_tlpips(frames):
    if len(frames) < 2:
        return 0.0
    vals = []
    for i in range(len(frames)-1):
        vals.append(compute_lpips(frames[i], frames[i+1]))
    return np.mean(vals)

def compute_fid(gt_frames, sr_frames):
    if not TORCHMETRICS_AVAILABLE:
        return None
    def frames_to_tensor(frames):
        tensors = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0)
            tensors.append(t)
        return torch.cat(tensors, dim=0).to(torch.uint8)
    gt_tensor = frames_to_tensor(gt_frames)
    sr_tensor = frames_to_tensor(sr_frames)
    fid = FrechetInceptionDistance(feature=2048, normalize=False)
    fid.update(gt_tensor, real=True)
    fid.update(sr_tensor, real=False)
    return fid.compute().item()

def read_image_sequence(folder_path, ext='png'):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(ext)]
    def natural_sort_key(s):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]
    files.sort(key=natural_sort_key)
    frames = []
    for f in files:
        img = cv2.imread(os.path.join(folder_path, f))
        if img is not None:
            frames.append(img)
    if not frames:
        raise ValueError(f"No images in {folder_path}")
    return frames

def load_frames(path):
    if os.path.isfile(path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    else:
        return read_image_sequence(path)

def evaluate_single_video(gt_frames, sr_frames, method_name, compute_lpips_flag=True, compute_fid_flag=False):
    n = min(len(gt_frames), len(sr_frames))
    gt_frames = gt_frames[:n]
    sr_frames = sr_frames[:n]

    psnr_list, ssim_list, lpips_list = [], [], []
    for i in range(n):
        gt = gt_frames[i]
        sr = sr_frames[i]
        if gt.shape != sr.shape:
            sr = resize_frame(sr, gt.shape[0], gt.shape[1])
        psnr_list.append(psnr(gt, sr, data_range=255))
        ssim_list.append(ssim(gt, sr, multichannel=True, data_range=255, channel_axis=2))
        if compute_lpips_flag:
            lpips_list.append(compute_lpips(gt, sr))

    sr_aligned = [resize_frame(sr_frames[i], gt_frames[i].shape[0], gt_frames[i].shape[1]) if gt_frames[i].shape != sr_frames[i].shape else sr_frames[i] for i in range(n)]
    tlpips_val = compute_tlpips(sr_aligned) if compute_lpips_flag else 0.0

    results = {
        "method": method_name,
        "avg_psnr": float(np.mean(psnr_list)),
        "avg_ssim": float(np.mean(ssim_list)),
    }
    if compute_lpips_flag:
        results["avg_lpips"] = float(np.mean(lpips_list)) if lpips_list else 0.0
        results["tlpips"] = float(tlpips_val)
    if compute_fid_flag and TORCHMETRICS_AVAILABLE:
        fid_val = compute_fid(gt_frames, sr_aligned)
        if fid_val is not None:
            results["fid"] = float(fid_val)
    return results

# ----------------------------- Main evaluation -----------------------------
def main():
    # Paths (adjust if needed, but using relative from script location)
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent   # VideoSuperResolution
    basicvsr_root = project_root / "BasicVSR_PlusPlus"
    gt_root = basicvsr_root / "data" / "reds" / "gt_videos"
    baseline_path = basicvsr_root / "results" / "fp32"   # BasicVSR++ original
    direction_c_variants_root = script_dir / "results" / "direction_c_variants"   # contains threshold_0.2, 0.3, ...
    output_dir = script_dir / "results" / "evaluation_thresholds"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define methods: baseline and each threshold
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    method_dict = {}
    method_dict["BasicVSR++ (FP32)"] = str(baseline_path)
    for th in thresholds:
        method_dict[f"DirectionC_th_{th}"] = str(direction_c_variants_root / f"threshold_{th}")

    # Get video names from baseline (e.g., 000, 011, ...)
    if not baseline_path.exists():
        print(f"Baseline path {baseline_path} not found. Exiting.")
        return
    video_names = [d.name for d in baseline_path.iterdir() if d.is_dir()]
    if not video_names:
        print("No video folders found.")
        return

    print(f"Found videos: {video_names}")
    all_results = []  # each entry: {video, method, metrics}

    # Flags
    compute_lpips = True
    compute_fid = True

    for video_name in video_names:
        print(f"\n=== Processing video: {video_name} ===")
        gt_path = gt_root / video_name
        if not gt_path.exists():
            alt_path = gt_root / f"{video_name}.mp4"
            if alt_path.exists():
                gt_path = alt_path
            else:
                print(f"GT not found for {video_name}. Skipping.")
                continue
        try:
            gt_frames = load_frames(str(gt_path))
            print(f"  Loaded {len(gt_frames)} GT frames")
        except Exception as e:
            print(f"Failed to load GT: {e}")
            continue

        for method_name, method_root in method_dict.items():
            sr_path = Path(method_root) / video_name
            if not sr_path.exists():
                print(f"  {method_name} result not found at {sr_path}, skipping.")
                continue
            try:
                sr_frames = load_frames(str(sr_path))
                if len(sr_frames) > len(gt_frames):
                    sr_frames = sr_frames[:len(gt_frames)]
                elif len(sr_frames) < len(gt_frames):
                    last = sr_frames[-1]
                    sr_frames += [last] * (len(gt_frames) - len(sr_frames))
                metrics = evaluate_single_video(gt_frames, sr_frames, method_name,
                                                compute_lpips_flag=compute_lpips,
                                                compute_fid_flag=compute_fid)
                metrics['video'] = video_name
                all_results.append(metrics)
                print(f"    {method_name}: PSNR={metrics['avg_psnr']:.2f}, LPIPS={metrics.get('avg_lpips',0):.4f}")
            except Exception as e:
                print(f"  Error evaluating {method_name}: {e}")
                continue

    if not all_results:
        print("No results to aggregate.")
        return

    # Save per-video JSON
    per_video_json = output_dir / "per_video_metrics.json"
    with open(per_video_json, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved per-video metrics to {per_video_json}")

    # Aggregate across videos
    method_agg = {}
    for item in all_results:
        method = item['method']
        keys = ['avg_psnr', 'avg_ssim']
        if 'avg_lpips' in item:
            keys.append('avg_lpips')
        if 'tlpips' in item:
            keys.append('tlpips')
        if 'fid' in item:
            keys.append('fid')
        if method not in method_agg:
            method_agg[method] = {k: [] for k in keys}
        for k in keys:
            method_agg[method][k].append(item[k])

    avg_results = []
    for method, vals in method_agg.items():
        entry = {'method': method}
        for k, lst in vals.items():
            entry[f'avg_{k}'] = float(np.mean(lst))
            entry[f'std_{k}'] = float(np.std(lst))
        avg_results.append(entry)

    avg_json = output_dir / "average_metrics.json"
    with open(avg_json, 'w') as f:
        json.dump(avg_results, f, indent=4)
    print(f"Saved aggregate metrics to {avg_json}")

    # Generate plots: threshold vs metrics for DirectionC variants (plus baseline as horizontal line)
    # Extract data for thresholds only
    threshold_data = {}
    for res in avg_results:
        name = res['method']
        if name.startswith("DirectionC_th_"):
            th = float(name.split('_')[-1])
            threshold_data[th] = {
                'psnr': res['avg_avg_psnr'],
                'ssim': res['avg_avg_ssim'],
                'lpips': res.get('avg_avg_lpips', 0),
                'tlpips': res.get('avg_tlpips', 0),
                'fid': res.get('avg_fid', 0)
            }
    baseline_psnr = next((r['avg_avg_psnr'] for r in avg_results if r['method'] == "BasicVSR++ (FP32)"), None)
    baseline_lpips = next((r.get('avg_avg_lpips', None) for r in avg_results if r['method'] == "BasicVSR++ (FP32)"), None)

    # Sort thresholds
    sorted_th = sorted(threshold_data.keys())
    if sorted_th:
        # PSNR vs threshold
        plt.figure()
        psnr_vals = [threshold_data[th]['psnr'] for th in sorted_th]
        plt.plot(sorted_th, psnr_vals, 'o-', label='Direction C')
        if baseline_psnr is not None:
            plt.axhline(y=baseline_psnr, color='r', linestyle='--', label='BasicVSR++ baseline')
        plt.xlabel('Confidence threshold')
        plt.ylabel('PSNR (dB)')
        plt.title('PSNR vs Confidence Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / 'psnr_vs_threshold.png')
        plt.close()

        # LPIPS vs threshold
        if 'lpips' in threshold_data[sorted_th[0]]:
            plt.figure()
            lpips_vals = [threshold_data[th]['lpips'] for th in sorted_th]
            plt.plot(sorted_th, lpips_vals, 'o-', label='Direction C')
            if baseline_lpips is not None:
                plt.axhline(y=baseline_lpips, color='r', linestyle='--', label='BasicVSR++ baseline')
            plt.xlabel('Confidence threshold')
            plt.ylabel('LPIPS (lower better)')
            plt.title('LPIPS vs Confidence Threshold')
            plt.legend()
            plt.grid(True)
            plt.savefig(output_dir / 'lpips_vs_threshold.png')
            plt.close()

    # Bar chart comparing all methods (including baseline and all thresholds)
    methods_all = [r['method'] for r in avg_results]
    psnr_all = [r['avg_avg_psnr'] for r in avg_results]
    lpips_all = [r.get('avg_avg_lpips', 0) for r in avg_results]

    # PSNR bar chart
    plt.figure(figsize=(12,6))
    bars = plt.bar(methods_all, psnr_all, color='skyblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('PSNR (dB)')
    plt.title('Average PSNR across videos')
    for bar, val in zip(bars, psnr_all):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig(output_dir / 'psnr_comparison_all.png')
    plt.close()

    # LPIPS bar chart
    if any(lpips_all):
        plt.figure(figsize=(12,6))
        bars = plt.bar(methods_all, lpips_all, color='lightgreen', edgecolor='black')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('LPIPS (lower better)')
        plt.title('Average LPIPS across videos')
        for bar, val in zip(bars, lpips_all):
            plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}', ha='center')
        plt.tight_layout()
        plt.savefig(output_dir / 'lpips_comparison_all.png')
        plt.close()

    print(f"\nAll evaluation outputs saved to {output_dir}")

if __name__ == "__main__":
    main()