import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_part1_metrics(json_path="outputs/average_metrics_across_videos.json"):
    """Load Part 1 average metrics (from evaluation script)."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    # data is a list of dicts with keys: method, avg_psnr, avg_ssim, std_psnr, std_ssim
    return data

def load_basicvsrpp_metrics(json_path="data/basicvsrpp/basicvsrpp_results.json", precision="fp32"):
    """Load BasicVSR++ metrics from its JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    if precision not in data:
        raise ValueError(f"Precision {precision} not found in {json_path}")
    metrics = data[precision]["quality"]
    return {
        "method": f"BasicVSR++ ({precision})",
        "avg_psnr": metrics["psnr"],
        "avg_ssim": metrics["ssim"],
        "std_psnr": 0.0,
        "std_ssim": 0.0
    }

def save_psnr_chart(methods, psnr_vals, psnr_stds, output_dir):
    """Save PSNR bar chart with y-axis limit 40."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.6
    bars = ax.bar(x, psnr_vals, width, yerr=psnr_stds, capsize=5,
                  color='skyblue', edgecolor='black', error_kw={'linewidth': 1})
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('PSNR Comparison (Average across videos)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 40)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, val in zip(bars, psnr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "psnr_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PSNR chart saved to {save_path}")

def save_ssim_chart(methods, ssim_vals, ssim_stds, output_dir):
    """Save SSIM bar chart with y-axis limit 1."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.6
    bars = ax.bar(x, ssim_vals, width, yerr=ssim_stds, capsize=5,
                  color='lightgreen', edgecolor='black', error_kw={'linewidth': 1})
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_title('SSIM Comparison (Average across videos)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, val in zip(bars, ssim_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "ssim_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SSIM chart saved to {save_path}")

def main():
    # Paths
    part1_json = "outputs/average_metrics_across_videos.json"
    basicvsrpp_json = "data/basicvsrpp/basicvsrpp_results.json"
    precision = "fp32"

    if not os.path.exists(part1_json):
        print(f"Part 1 metrics not found at {part1_json}. Please run evaluation first.")
        return
    if not os.path.exists(basicvsrpp_json):
        print(f"BasicVSR++ results not found at {basicvsrpp_json}. Please provide the file.")
        return

    part1_results = load_part1_metrics(part1_json)
    extra_result = load_basicvsrpp_metrics(basicvsrpp_json, precision)
    
    # merge results and prepare for plotting
    all_results = part1_results + [extra_result]
    methods = [r['method'] for r in all_results]
    psnr_vals = [r['avg_psnr'] for r in all_results]
    psnr_stds = [r['std_psnr'] for r in all_results]
    ssim_vals = [r['avg_ssim'] for r in all_results]
    ssim_stds = [r['std_ssim'] for r in all_results]

    # 分别保存两张图
    save_psnr_chart(methods, psnr_vals, psnr_stds, output_dir="outputs")
    save_ssim_chart(methods, ssim_vals, ssim_stds, output_dir="outputs")

if __name__ == "__main__":
    main()