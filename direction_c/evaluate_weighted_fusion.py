#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate weighted fusion results (all alpha values) against GT.
Automatically scans output_root for alpha_* subfolders.
Usage: python evaluate_weighted_fusion.py
"""

import os
import re
import json
import cv2
import numpy as np
from tqdm import tqdm
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from pathlib import Path

# Optional FID
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not installed. FID will be skipped.")

lpips_fn = lpips.LPIPS(net='alex', verbose=False)


# ----------------------------- Helper functions (same as earlier) -----------------------------
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

def evaluate_video(gt_frames, sr_frames, method_name, compute_lpips_flag=True, compute_fid_flag=False):
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

    sr_aligned = []
    for i in range(n):
        if gt_frames[i].shape != sr_frames[i].shape:
            sr_aligned.append(resize_frame(sr_frames[i], gt_frames[i].shape[0], gt_frames[i].shape[1]))
        else:
            sr_aligned.append(sr_frames[i])
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
    # Paths (adjust to your original VideoSuperResolution structure)
    gt_root = r"D:\VideoSuperResolution\BasicVSR_PlusPlus\data\reds\gt_videos"
    weighted_avg_root = r"D:\VideoSuperResolution\BasicVSR_PlusPlus\results\weighted_avg"
    output_dir = Path(weighted_avg_root) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scan for alpha subfolders (e.g., alpha_0_0, alpha_0_5, alpha_1_0)
    alpha_folders = []
    for d in Path(weighted_avg_root).iterdir():
        if d.is_dir() and d.name.startswith("alpha_"):
            # convert folder name to a readable method name
            alpha_str = d.name.replace("alpha_", "").replace("_", ".")
            method_name = f"Weighted (alpha={alpha_str})"
            alpha_folders.append((method_name, d))
    # Sort by alpha value for consistent order
    alpha_folders.sort(key=lambda x: float(x[0].split('=')[1].replace(')', '')))

    if not alpha_folders:
        print(f"No alpha_* folders found in {weighted_avg_root}")
        return

    # Also include original BasicVSR++ and Direction C? They are already included as alpha=1 and alpha=0 if present.
    # But we can add them explicitly in case not generated. For safety, we'll rely on the folders.

    # Get list of video names from first available method (assuming all have same subfolders)
    first_method_root = alpha_folders[0][1]
    video_names = [d.name for d in first_method_root.iterdir() if d.is_dir()]
    if not video_names:
        print(f"No video subfolders in {first_method_root}")
        return

    print(f"Found {len(video_names)} videos: {video_names}")
    print(f"Evaluating {len(alpha_folders)} weightings: {[name for name,_ in alpha_folders]}")

    compute_lpips = True
    compute_fid = True
    all_results = []  # each item: {video, method, metrics}

    for video_name in tqdm(video_names, desc="Videos"):
        gt_path = Path(gt_root) / video_name
        if not gt_path.exists():
            alt = Path(gt_root) / f"{video_name}.mp4"
            if alt.exists():
                gt_path = alt
            else:
                print(f"GT not found for {video_name}, skipping")
                continue
        try:
            gt_frames = load_frames(str(gt_path))
        except Exception as e:
            print(f"Failed to load GT for {video_name}: {e}")
            continue

        for method_name, folder in alpha_folders:
            sr_path = folder / video_name
            if not sr_path.exists():
                print(f"  Missing {method_name} for {video_name}, skipping")
                continue
            try:
                sr_frames = load_frames(str(sr_path))
                if len(sr_frames) > len(gt_frames):
                    sr_frames = sr_frames[:len(gt_frames)]
                elif len(sr_frames) < len(gt_frames):
                    last = sr_frames[-1]
                    sr_frames += [last] * (len(gt_frames) - len(sr_frames))
                metrics = evaluate_video(gt_frames, sr_frames, method_name,
                                         compute_lpips_flag=compute_lpips,
                                         compute_fid_flag=compute_fid)
                metrics['video'] = video_name
                all_results.append(metrics)
            except Exception as e:
                print(f"  Error evaluating {method_name} for {video_name}: {e}")

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

    # Sort by alpha value (numerical) for plotting
    avg_results.sort(key=lambda x: float(x['method'].split('=')[1].replace(')', '')))

    # Save aggregate JSON
    avg_json = output_dir / "average_metrics.json"
    with open(avg_json, 'w') as f:
        json.dump(avg_results, f, indent=4)
    print(f"Saved aggregate metrics to {avg_json}")

    # Generate line plots: metrics vs alpha
    alphas = [float(r['method'].split('=')[1].replace(')', '')) for r in avg_results]
    psnr_vals = [r['avg_avg_psnr'] for r in avg_results]
    ssim_vals = [r['avg_avg_ssim'] for r in avg_results]
    lpips_vals = [r['avg_avg_lpips'] for r in avg_results if 'avg_avg_lpips' in r]
    tlpips_vals = [r['avg_tlpips'] for r in avg_results if 'avg_tlpips' in r]
    fid_vals = [r['avg_fid'] for r in avg_results if 'avg_fid' in r]

    # PSNR vs alpha
    plt.figure()
    plt.plot(alphas, psnr_vals, 'o-', color='b')
    plt.xlabel('Alpha (weight on BasicVSR++)')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs Fusion Weight')
    plt.grid(True)
    plt.savefig(output_dir / 'psnr_vs_alpha.png')
    plt.close()

    # SSIM vs alpha
    plt.figure()
    plt.plot(alphas, ssim_vals, 's-', color='g')
    plt.xlabel('Alpha')
    plt.ylabel('SSIM')
    plt.title('SSIM vs Fusion Weight')
    plt.grid(True)
    plt.savefig(output_dir / 'ssim_vs_alpha.png')
    plt.close()

    if lpips_vals:
        plt.figure()
        plt.plot(alphas, lpips_vals, '^-', color='r')
        plt.xlabel('Alpha')
        plt.ylabel('LPIPS (lower better)')
        plt.title('LPIPS vs Fusion Weight')
        plt.grid(True)
        plt.savefig(output_dir / 'lpips_vs_alpha.png')
        plt.close()

    if tlpips_vals:
        plt.figure()
        plt.plot(alphas, tlpips_vals, 'd-', color='m')
        plt.xlabel('Alpha')
        plt.ylabel('tLPIPS (lower better)')
        plt.title('tLPIPS vs Fusion Weight')
        plt.grid(True)
        plt.savefig(output_dir / 'tlpips_vs_alpha.png')
        plt.close()

    if fid_vals:
        plt.figure()
        plt.plot(alphas, fid_vals, 'x-', color='c')
        plt.xlabel('Alpha')
        plt.ylabel('FID (lower better)')
        plt.title('FID vs Fusion Weight')
        plt.grid(True)
        plt.savefig(output_dir / 'fid_vs_alpha.png')
        plt.close()

    # Also generate a comparative bar chart for selected alphas (e.g., 0, 0.3, 0.5, 0.7, 1)
    selected_alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    selected_results = [r for r in avg_results if float(r['method'].split('=')[1].replace(')', '')) in selected_alphas]
    if selected_results:
        methods = [r['method'] for r in selected_results]
        psnr_sel = [r['avg_avg_psnr'] for r in selected_results]
        lpips_sel = [r['avg_avg_lpips'] for r in selected_results]

        plt.figure(figsize=(10,5))
        x = np.arange(len(methods))
        width = 0.35
        plt.bar(x - width/2, psnr_sel, width, label='PSNR (dB)')
        plt.bar(x + width/2, lpips_sel, width, label='LPIPS (lower better)')
        plt.xticks(x, methods, rotation=45)
        plt.ylabel('Score')
        plt.title('Comparison of Selected Fusion Weights')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'selected_alpha_comparison.png')
        plt.close()

    print(f"\nAll evaluation outputs saved to {output_dir}")

if __name__ == "__main__":
    main()