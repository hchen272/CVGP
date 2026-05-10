#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate comparison charts from evaluation_report.json.
Separate charts for PSNR and SSIM because their scales differ.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_report(report_path):
    with open(report_path, 'r') as f:
        return json.load(f)

def plot_speed_comparison(data, output_path):
    fp32 = data['fp32']
    fp16 = data['fp16']
    
    metrics = ['End-to-end FPS', 'Inference FPS']
    fp32_vals = [fp32['fps_video'], fp32['fps_inference']]
    fp16_vals = [fp16['fps_video'], fp16['fps_inference']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, fp32_vals, width, label='FP32', color='#1f77b4')
    bars2 = ax.bar(x + width/2, fp16_vals, width, label='FP16', color='#ff7f0e')
    
    ax.set_ylabel('Frames Per Second')
    ax.set_title('Inference Speed Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Speed chart saved to {output_path}")

def plot_psnr_comparison(data, output_path):
    fp32_qual = data['fp32']['quality']
    fp16_qual = data['fp16']['quality']
    
    labels = ['PSNR (dB)']
    fp32_val = [fp32_qual['psnr']]
    fp16_val = [fp16_qual['psnr']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(6, 5))
    bars1 = ax.bar(x - width/2, fp32_val, width, label='FP32', color='#1f77b4')
    bars2 = ax.bar(x + width/2, fp16_val, width, label='FP16', color='#ff7f0e')
    
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('PSNR Comparison (higher is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 40)   
    ax.legend()
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"PSNR chart saved to {output_path}")

def plot_ssim_comparison(data, output_path):
    fp32_qual = data['fp32']['quality']
    fp16_qual = data['fp16']['quality']
    
    labels = ['SSIM']
    fp32_val = [fp32_qual['ssim']]
    fp16_val = [fp16_qual['ssim']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(6, 5))
    bars1 = ax.bar(x - width/2, fp32_val, width, label='FP32', color='#1f77b4')
    bars2 = ax.bar(x + width/2, fp16_val, width, label='FP16', color='#ff7f0e')
    
    ax.set_ylabel('SSIM')
    ax.set_title('SSIM Comparison (higher is better, 1 = perfect)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)   
    ax.legend()
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"SSIM chart saved to {output_path}")

def main():
    report_path = Path("results/eval/evaluation_report.json")
    if not report_path.exists():
        print(f"Report not found at {report_path}")
        return
    
    data = load_report(report_path)
    output_dir = report_path.parent
    
    plot_speed_comparison(data, output_dir / "speed_comparison.png")
    plot_psnr_comparison(data, output_dir / "psnr_comparison.png")
    plot_ssim_comparison(data, output_dir / "ssim_comparison.png")
    
    print("All charts generated.")

if __name__ == '__main__':
    main()