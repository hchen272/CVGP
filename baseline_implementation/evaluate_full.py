#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified evaluation script for all methods:
- Baseline (bicubic, lanczos, temporal_avg, unsharp, srcnn, ...)
- BasicVSR++ (fp16, fp32)
- ESRGAN (standalone)
- Direction C

Assumes each method's output is stored either as:
  Type A: root/video_name/method_name/   (Baseline)
  Type B: root/video_name/               (BasicVSR++, Direction C, etc.)
  Type C: root/video_name/subfolder/     (ESRGAN)
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

# Optional FID
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not installed. FID will be skipped.")

lpips_fn = lpips.LPIPS(net='alex', verbose=False)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def resize_frame(frame, target_h, target_w):
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

def compute_lpips(img1, img2):
    """Compute LPIPS between two images. Returns float."""
    try:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t1 = torch.from_numpy(img1_rgb).permute(2, 0, 1).unsqueeze(0)
        t2 = torch.from_numpy(img2_rgb).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            d = lpips_fn(t1, t2)
        return d.item()
    except Exception as e:
        print(f"    [ERROR in compute_lpips] {e}")
        return 1.0  # high error as fallback

def compute_tlpips(frames):
    if len(frames) < 2:
        return 0.0
    lpips_vals = []
    for i in range(len(frames) - 1):
        lpips_vals.append(compute_lpips(frames[i], frames[i+1]))
    return np.mean(lpips_vals)

def compute_fid(gt_frames, sr_frames):
    if not TORCHMETRICS_AVAILABLE:
        return None
    print(f"      compute_fid: {len(gt_frames)} GT, {len(sr_frames)} SR frames")
    def frames_to_tensor(frames):
        rgb_tensors = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            rgb_tensors.append(t)
        return torch.cat(rgb_tensors, dim=0).to(torch.uint8)
    gt_tensor = frames_to_tensor(gt_frames)
    sr_tensor = frames_to_tensor(sr_frames)
    print(f"      Tensors created: GT shape {gt_tensor.shape}, SR shape {sr_tensor.shape}")
    fid = FrechetInceptionDistance(feature=2048, normalize=False)
    fid.update(gt_tensor, real=True)
    fid.update(sr_tensor, real=False)
    fid_value = fid.compute().item()
    print(f"      Computed FID = {fid_value:.2f}")
    return fid_value

# ----------------------------------------------------------------------
# Frame loading utilities
# ----------------------------------------------------------------------
def read_video_frames(video_path):
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
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(ext)]
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
    if os.path.isfile(path):
        return read_video_frames(path)
    elif os.path.isdir(path):
        for ext in ['png', 'jpg', 'jpeg', 'bmp']:
            if any(f.lower().endswith(ext) for f in os.listdir(path)):
                return read_image_sequence(path, ext)
        raise ValueError(f"No valid image files found in {path}")
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

# ----------------------------------------------------------------------
# Evaluation core (with intermediate prints but no debug labels)
# ----------------------------------------------------------------------
def evaluate_video(gt_frames, sr_frames, method_name,
                   compute_lpips_flag=True, compute_fid_flag=False):
    n_frames = min(len(gt_frames), len(sr_frames))
    gt_frames = gt_frames[:n_frames]
    sr_frames = sr_frames[:n_frames]

    psnr_list = []
    ssim_list = []
    lpips_list = []

    print(f"    {method_name}: evaluating {n_frames} frames (LPIPS={compute_lpips_flag})")
    for i in tqdm(range(n_frames), desc=f"  {method_name}", leave=False):
        gt = gt_frames[i]
        sr = sr_frames[i]
        if gt.shape != sr.shape:
            sr = resize_frame(sr, gt.shape[0], gt.shape[1])
        psnr_val = psnr(gt, sr, data_range=255)
        ssim_val = ssim(gt, sr, multichannel=True, data_range=255, channel_axis=2)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        if compute_lpips_flag:
            try:
                lpips_val = compute_lpips(gt, sr)
                lpips_list.append(lpips_val)
            except Exception as e:
                print(f"      [ERROR] frame {i} LPIPS failed: {e}")
                lpips_list.append(1.0)

    # Align SR frames for tLPIPS
    sr_aligned = []
    for i in range(n_frames):
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
        if lpips_list:
            results["avg_lpips"] = float(np.mean(lpips_list))
            results["tlpips"] = float(tlpips_val)
            print(f"    {method_name}: computed LPIPS (avg={results['avg_lpips']:.4f}) and tLPIPS={results['tlpips']:.4f}")
        else:
            print(f"    {method_name}: LPIPS list is empty, skipping LPIPS/tLPIPS")
            results["avg_lpips"] = 0.0
            results["tlpips"] = 0.0

    if compute_fid_flag and TORCHMETRICS_AVAILABLE:
        print(f"    Starting FID computation for {method_name}...")
        fid_val = compute_fid(gt_frames, sr_aligned)
        if fid_val is not None:
            results["fid"] = float(fid_val)
            print(f"    FID = {results['fid']:.2f}")
        else:
            print(f"    FID computation returned None for {method_name}")

    return results

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def save_bar_chart(data_dict, metric_name, ylabel, title, output_path, ylim=None):
    methods = list(data_dict.keys())
    values = list(data_dict.values())
    # For metrics where lower is better (LPIPS, tLPIPS, FID), reverse sort order for display
    reverse = metric_name not in ['LPIPS', 'tLPIPS', 'FID']
    sorted_pairs = sorted(zip(methods, values), key=lambda x: x[1], reverse=reverse)
    methods_sorted = [p[0] for p in sorted_pairs]
    values_sorted = [p[1] for p in sorted_pairs]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(methods_sorted, values_sorted, color='skyblue', edgecolor='black')
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if ylim:
        plt.ylim(ylim)

    for bar, val in zip(bars, values_sorted):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(values_sorted)*0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {metric_name} chart to {output_path}")

# ----------------------------------------------------------------------
# MAIN (with relative paths)
# ----------------------------------------------------------------------
def main():
    # Determine root directory based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))        # .../baseline_implementation
    project_root = os.path.dirname(script_dir)                     # .../VideoSuperResolution
    basicvsr_root = os.path.join(project_root, "BasicVSR_PlusPlus")
    esrgan_root = os.path.join(project_root, "ESRGAN")

    # Define paths relative to project root
    gt_root = os.path.join(basicvsr_root, "data", "reds", "gt_videos")
    baseline_output_root = os.path.join(basicvsr_root, "outputs")
    basicvsr_fp16_root = os.path.join(basicvsr_root, "results", "fp16")
    basicvsr_fp32_root = os.path.join(basicvsr_root, "results", "fp32")
    direction_c_root = os.path.join(basicvsr_root, "results", "direction_c")
    esrgan_results_root = os.path.join(esrgan_root, "results")
    output_dir = os.path.join(basicvsr_root, "results", "evaluation")

    # Methods definition (using relative paths)
    methods = {
        # Baseline (Type A)
        "bicubic_x4": {
            "root": baseline_output_root,
            "type": "A",
            "subdir": "bicubic_x4"
        },
        "lanczos_x4": {
            "root": baseline_output_root,
            "type": "A",
            "subdir": "lanczos_x4"
        },
        "bicubic_temporal_avg": {
            "root": baseline_output_root,
            "type": "A",
            "subdir": "bicubic_x4_temporal_avg"
        },
        "bicubic_temporal_unsharp": {
            "root": baseline_output_root,
            "type": "A",
            "subdir": "bicubic_x4_temporal_avg_unsharp"
        },
        "lanczos_temporal_avg": {
            "root": baseline_output_root,
            "type": "A",
            "subdir": "lanczos_x4_temporal_avg"
        },
        "lanczos_temporal_unsharp": {
            "root": baseline_output_root,
            "type": "A",
            "subdir": "lanczos_x4_temporal_avg_unsharp"
        },
        "srcnn_x4": {
            "root": baseline_output_root,
            "type": "A",
            "subdir": "srcnn_x4"
        },
        # BasicVSR++ (Type B)
        "BasicVSR++ (FP16)": {
            "root": basicvsr_fp16_root,
            "type": "B"
        },
        "BasicVSR++ (FP32)": {
            "root": basicvsr_fp32_root,
            "type": "B"
        },
        # Direction C (Type B)
        "Direction C": {
            "root": direction_c_root,
            "type": "B"
        },
        # ESRGAN standalone (Type C)
        "ESRGAN (4x)": {
            "root": esrgan_results_root,
            "type": "C",
            "subdir": "realesr_x4"
        },
    }

    compute_lpips_flag = True
    compute_fid_flag = True

    os.makedirs(output_dir, exist_ok=True)

    # Determine video names from first available method root
    video_names = None
    for method_name, cfg in methods.items():
        root = cfg["root"]
        if not os.path.exists(root):
            print(f"Warning: root not found for {method_name}: {root}")
            continue
        items = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        if items:
            video_names = items
            break
    if not video_names:
        print("No video sequences found in any method root. Exiting.")
        return

    print(f"Found {len(video_names)} video sequences: {video_names}")
    print(f"Evaluating methods: {list(methods.keys())}")

    all_results = []

    for video_name in video_names:
        print(f"\n=== Processing video: {video_name} ===")

        # Load GT frames
        gt_path = os.path.join(gt_root, video_name)
        if not os.path.exists(gt_path):
            alt_path = os.path.join(gt_root, f"{video_name}.mp4")
            if os.path.exists(alt_path):
                gt_path = alt_path
            else:
                print(f"GT not found for {video_name}. Skipping.")
                continue
        try:
            gt_frames = load_frames(gt_path)
            print(f"  Loaded {len(gt_frames)} GT frames")
        except Exception as e:
            print(f"Failed to load GT from {gt_path}: {e}")
            continue

        # Evaluate each method
        for method_name, cfg in methods.items():
            root = cfg["root"]
            if not os.path.exists(root):
                print(f"  Root missing for {method_name}, skipping.")
                continue

            if cfg["type"] == "A":
                sr_path = os.path.join(root, video_name, cfg["subdir"])
            elif cfg["type"] == "B":
                sr_path = os.path.join(root, video_name)
            elif cfg["type"] == "C":
                sr_path = os.path.join(root, video_name, cfg["subdir"])
            else:
                continue

            if not os.path.exists(sr_path):
                print(f"  {method_name} result not found at {sr_path}, skipping.")
                continue

            try:
                sr_frames = load_frames(sr_path)
                print(f"  Loaded {len(sr_frames)} frames for {method_name}")
                # Trim or pad to match GT length
                if len(sr_frames) > len(gt_frames):
                    sr_frames = sr_frames[:len(gt_frames)]
                elif len(sr_frames) < len(gt_frames):
                    print(f"  Warning: {method_name} has fewer frames than GT. Padding with last frame.")
                    last = sr_frames[-1]
                    sr_frames += [last] * (len(gt_frames) - len(sr_frames))
                metrics = evaluate_video(gt_frames, sr_frames, method_name,
                                         compute_lpips_flag, compute_fid_flag)
                metrics['video'] = video_name
                all_results.append(metrics)
                # Print summary line
                summary = f"    => {method_name} PSNR={metrics['avg_psnr']:.2f}, SSIM={metrics['avg_ssim']:.4f}"
                if 'avg_lpips' in metrics:
                    summary += f", LPIPS={metrics['avg_lpips']:.4f}, tLPIPS={metrics['tlpips']:.4f}"
                if 'fid' in metrics:
                    summary += f", FID={metrics['fid']:.2f}"
                print(summary)
            except Exception as e:
                print(f"  Failed to evaluate {method_name} for {video_name}: {e}")
                import traceback
                traceback.print_exc()

    if not all_results:
        print("No results to aggregate.")
        return

    # Save per-video JSON
    per_video_json = os.path.join(output_dir, "per_video_metrics.json")
    with open(per_video_json, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Per-video metrics saved to {per_video_json}")

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

    # Print aggregated results
    print("\n=== AGGREGATED RESULTS (averages across videos) ===")
    for res in avg_results:
        line = f"  {res['method']}: PSNR={res.get('avg_avg_psnr',0):.2f}, SSIM={res.get('avg_avg_ssim',0):.4f}"
        if 'avg_avg_lpips' in res:
            line += f", LPIPS={res['avg_avg_lpips']:.4f}"
        if 'avg_tlpips' in res:
            line += f", tLPIPS={res['avg_tlpips']:.4f}"
        if 'avg_fid' in res:
            line += f", FID={res['avg_fid']:.2f}"
        print(line)

    # Save aggregate JSON
    avg_json_path = os.path.join(output_dir, "average_metrics.json")
    with open(avg_json_path, 'w') as f:
        json.dump(avg_results, f, indent=4)
    print(f"Aggregate metrics saved to {avg_json_path}")

    # Generate bar charts
    # PSNR
    psnr_dict = {r['method']: r['avg_avg_psnr'] for r in avg_results}
    if psnr_dict:
        save_bar_chart(psnr_dict, 'PSNR', 'PSNR (dB)', 'Average PSNR across videos',
                       os.path.join(output_dir, 'psnr_comparison.png'), ylim=(0, 40))
    # SSIM
    ssim_dict = {r['method']: r['avg_avg_ssim'] for r in avg_results}
    if ssim_dict:
        save_bar_chart(ssim_dict, 'SSIM', 'SSIM', 'Average SSIM across videos',
                       os.path.join(output_dir, 'ssim_comparison.png'), ylim=(0, 1))
    # LPIPS (if exists)
    if avg_results and 'avg_avg_lpips' in avg_results[0]:
        lpips_dict = {r['method']: r['avg_avg_lpips'] for r in avg_results}
        save_bar_chart(lpips_dict, 'LPIPS', 'LPIPS (lower better)', 'Average LPIPS across videos',
                       os.path.join(output_dir, 'lpips_comparison.png'), ylim=(0, 1))
    else:
        print("LPIPS key not found in avg_results, skipping LPIPS chart.")

    # tLPIPS
    if avg_results and 'avg_tlpips' in avg_results[0]:
        tlpips_dict = {r['method']: r['avg_tlpips'] for r in avg_results}
        save_bar_chart(tlpips_dict, 'tLPIPS', 'tLPIPS (lower better)', 'Average tLPIPS across videos',
                       os.path.join(output_dir, 'tlpips_comparison.png'), ylim=(0, 1))
    else:
        print("tLPIPS key not found in avg_results, skipping tLPIPS chart.")

     # FID (if exists)
    if avg_results and 'avg_fid' in avg_results[0]:
        fid_dict = {r['method']: r['avg_fid'] for r in avg_results}
        save_bar_chart(fid_dict, 'FID', 'FID (lower better)', 'Average FID across videos',
                       os.path.join(output_dir, 'fid_comparison.png'), ylim=None)
    else:
        print("FID key not found in avg_results, skipping FID chart.")

    print(f"\nEvaluation complete. Outputs saved to {output_dir}")

if __name__ == '__main__':
    main()