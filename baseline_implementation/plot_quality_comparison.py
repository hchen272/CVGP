#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot PSNR and SSIM comparison between SRCNN and BasicVSR++ (fp32) as a table.
Reads from:
- outputs/average_metrics_across_videos.json (for SRCNN)
- data/basicvsrpp/basicvsrpp_results.json (for BasicVSR++ fp32)
Saves a table image.
"""

import os
import json
import matplotlib.pyplot as plt

def load_srcnn_metrics(avg_metrics_path="outputs/average_metrics_across_videos.json"):
    """Extract SRCNN metrics."""
    with open(avg_metrics_path, 'r') as f:
        data = json.load(f)
    for item in data:
        if "srcnn" in item["method"].lower():
            return {
                "psnr": item["avg_psnr"],
                "ssim": item["avg_ssim"]
            }
    raise ValueError("SRCNN metrics not found")

def load_basicvsrpp_fp32_metrics(basicvsrpp_path="data/basicvsrpp/basicvsrpp_results.json"):
    """Extract BasicVSR++ fp32 metrics."""
    with open(basicvsrpp_path, 'r') as f:
        data = json.load(f)
    if "fp32" not in data:
        raise ValueError("fp32 data not found in BasicVSR++ results")
    quality = data["fp32"].get("quality", {})
    return {
        "psnr": quality.get("psnr", 0),
        "ssim": quality.get("ssim", 0)
    }

def plot_quality_table(srcnn, basicvsrpp_fp32, output_path="outputs/quality_comparison.png"):
    """Generate a table image with PSNR and SSIM values."""
    # Prepare table data
    methods = ["SRCNN", "BasicVSR++ (fp32)"]
    psnr_vals = [srcnn["psnr"], basicvsrpp_fp32["psnr"]]
    ssim_vals = [srcnn["ssim"], basicvsrpp_fp32["ssim"]]

    # Create figure and axis (hide axes)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    ax.axis('tight')

    # Table content: rows = methods, columns = Method, PSNR (dB), SSIM
    col_labels = ["Method", "PSNR (dB)", "SSIM"]
    table_data = [
        [methods[0], f"{psnr_vals[0]:.2f}", f"{ssim_vals[0]:.4f}"],
        [methods[1], f"{psnr_vals[1]:.2f}", f"{ssim_vals[1]:.4f}"]
    ]

    # Create table
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # Style: header background
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # header row
            cell.set_facecolor('#40466e')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#f2f2f2')
        cell.set_edgecolor('black')

    plt.title("Quality Comparison: SRCNN vs BasicVSR++ (fp32)", fontsize=14, pad=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Quality comparison table saved to {output_path}")

def main():
    try:
        srcnn = load_srcnn_metrics()
        print(f"SRCNN: PSNR={srcnn['psnr']:.2f}, SSIM={srcnn['ssim']:.4f}")
    except Exception as e:
        print(f"Error loading SRCNN metrics: {e}")
        return

    try:
        basicvsrpp_fp32 = load_basicvsrpp_fp32_metrics()
        print(f"BasicVSR++ (fp32): PSNR={basicvsrpp_fp32['psnr']:.2f}, SSIM={basicvsrpp_fp32['ssim']:.4f}")
    except Exception as e:
        print(f"Error loading BasicVSR++ metrics: {e}")
        return

    plot_quality_table(srcnn, basicvsrpp_fp32)

if __name__ == "__main__":
    main()