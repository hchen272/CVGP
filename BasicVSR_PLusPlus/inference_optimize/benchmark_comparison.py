import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics(filepath, key):
    with open(filepath) as f:
        data = json.load(f)
    return data[key]

def main():
    proj_root = Path(__file__).parent.parent
    baseline = load_metrics(proj_root / 'results/baseline_metrics.json', 'baseline')
    fp16 = load_metrics(proj_root / 'results/fp16_metrics.json', 'fp16')

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ----- Subplot 1: FPS comparison -----
    labels = ['FP32', 'FP16']
    fps_vals = [baseline['fps'], fp16['fps']]
    bars = ax1.bar(labels, fps_vals, color=['#1f77b4', '#ff7f0e'])
    ax1.set_ylabel('Frames Per Second (FPS)')
    ax1.set_title('Inference Speed')
    for bar, val in zip(bars, fps_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=10)

    # ----- Subplot 2: Time & Memory comparison (grouped bar) -----
    metrics = ['Avg Time (ms)', 'Peak Memory (MB)']
    base_vals = [baseline['avg_time_ms'], baseline['peak_memory_mb']]
    fp16_vals = [fp16['avg_time_ms'], fp16['peak_memory_mb']]

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax2.bar(x - width/2, base_vals, width, label='FP32', color='#1f77b4')
    bars2 = ax2.bar(x + width/2, fp16_vals, width, label='FP16', color='#ff7f0e')
    ax2.set_ylabel('Value')
    ax2.set_title('Time & Memory Usage')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = proj_root / 'results/performance_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")
    plt.show()

    # ----- Text summary (unchanged) -----
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON (FP32 Baseline vs FP16 Optimized)")
    print("="*60)
    print(f"{'Metric':<20} {'FP32 Baseline':<15} {'FP16 Optimized':<15} {'Speedup/Reduction':<20}")
    print("-"*70)
    print(f"{'FPS':<20} {baseline['fps']:<15.2f} {fp16['fps']:<15.2f} {fp16['fps']/baseline['fps']:<20.2f}x")
    print(f"{'Avg Time (ms)':<20} {baseline['avg_time_ms']:<15.2f} {fp16['avg_time_ms']:<15.2f} {baseline['avg_time_ms']/fp16['avg_time_ms']:<20.2f}x")
    print(f"{'Peak Memory (MB)':<20} {baseline['peak_memory_mb']:<15.2f} {fp16['peak_memory_mb']:<15.2f} {baseline['peak_memory_mb']/fp16['peak_memory_mb']:<20.2f}x")
    print("="*60)

if __name__ == '__main__':
    main()