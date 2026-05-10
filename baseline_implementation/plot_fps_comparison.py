#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot inference FPS comparison bar chart from existing timing_results.json files
and BasicVSR++ results (basicvsrpp_results.json).
"""

import os
import json
import glob
import matplotlib.pyplot as plt

def collect_fps_data(output_root="outputs", basicvsr_json="data/basicvsrpp/basicvsrpp_results.json"):
    """Collect inference FPS from all video subfolders and BasicVSR results."""
    all_data = []
    
    # Collect Part 1 inference timing results
    for video_dir in glob.glob(os.path.join(output_root, "*/")):
        timing_file = os.path.join(video_dir, "timing_results.json")
        if os.path.exists(timing_file):
            with open(timing_file) as f:
                timing = json.load(f)
            for method, vals in timing.items():
                fps_val = vals.get("inference_fps", vals.get("fps", 0))
                all_data.append({
                    "video": os.path.basename(video_dir.rstrip("/")),
                    "method": method,
                    "fps": fps_val
                })
    
    # Load BasicVSR results
    if os.path.exists(basicvsr_json):
        with open(basicvsr_json) as f:
            basicvsr = json.load(f)
        for prec in ["fp32"]:
            if prec in basicvsr:
                fps_val = basicvsr[prec].get("fps_inference", 0)
                all_data.append({
                    "video": "BasicVSR++",
                    "method": f"BasicVSR++ ({prec})",
                    "fps": fps_val
                })
    return all_data

def plot_fps_comparison(all_data, output_path="outputs/inference_fps_comparison.png"):
    """Generate bar chart of average inference FPS per method."""
    # Aggregate FPS per method (average across videos)
    method_fps = {}
    for item in all_data:
        method = item["method"]
        fps = item["fps"]
        if method not in method_fps:
            method_fps[method] = []
        method_fps[method].append(fps)
    
    methods = []
    avg_fps = []
    for method, fps_list in method_fps.items():
        methods.append(method)
        avg_fps.append(sum(fps_list) / len(fps_list))
    
    # Sort descending by FPS
    sorted_pairs = sorted(zip(methods, avg_fps), key=lambda x: x[1], reverse=True)
    methods_sorted = [p[0] for p in sorted_pairs]
    fps_sorted = [p[1] for p in sorted_pairs]
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(methods_sorted, fps_sorted, color='skyblue', edgecolor='black')
    plt.ylabel('Inference FPS (frames/second)', fontsize=12)
    plt.title('Inference Speed Comparison (Higher is Better)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar, fps in zip(bars, fps_sorted):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fps_sorted)*0.01,
                 f'{fps:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"FPS comparison chart saved to {output_path}")

def main():
    # Configuration
    output_root = "outputs"
    basicvsr_json = "data/basicvsrpp/basicvsrpp_results.json"
    
    print("Collecting FPS data from timing_results.json and BasicVSR results...")
    all_data = collect_fps_data(output_root, basicvsr_json)
    if not all_data:
        print("No FPS data found. Please run main_pipeline_part1.py first.")
        return
    
    # Print summary table
    print("\n===== Inference FPS (average across videos) =====")
    method_avg = {}
    for item in all_data:
        m = item["method"]
        if m not in method_avg:
            method_avg[m] = []
        method_avg[m].append(item["fps"])
    for m, fps_list in sorted(method_avg.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True):
        avg = sum(fps_list)/len(fps_list)
        print(f"{m:<40} {avg:.2f} fps")
    
    # Generate chart
    plot_fps_comparison(all_data)
    print("Done.")

if __name__ == "__main__":
    main()