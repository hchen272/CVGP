import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics_from_file(filepath):
    """Load FPS from various JSON structures."""
    if not filepath.exists():
        return None
    with open(filepath, 'r') as f:
        data = json.load(f)
    # 处理不同脚本的输出格式
    if 'baseline' in data:
        return data['baseline']['fps']
    elif 'fp16_comparison' in data:
        # fp16_metrics.json
        if 'fp16' in data['fp16_comparison']:
            return data['fp16_comparison']['fp16']['fps']
        elif 'fp32' in data['fp16_comparison']:
            return data['fp16_comparison']['fp32']['fps']
    else:
        return None

def main():
    # 定位到项目根目录下的 results 文件夹
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    
    baseline_fps = load_metrics_from_file(results_dir / 'baseline_metrics.json')
    fp16_fps = load_metrics_from_file(results_dir / 'fp16_metrics.json')
    
    # 只保留 baseline 和 fp16
    labels = []
    values = []
    if baseline_fps is not None:
        labels.append('FP32 Baseline')
        values.append(baseline_fps)
    if fp16_fps is not None:
        labels.append('FP16 Optimized')
        values.append(fp16_fps)
    
    if len(labels) < 2:
        print("Not enough metrics found. Need both baseline and fp16 JSON files.")
        print(f"Checked in: {results_dir}")
        return
    
    # 绘制柱状图
    plt.figure(figsize=(6, 4))
    colors = ['#1f77b4', '#ff7f0e']
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel('Frames Per Second (FPS)')
    plt.title('BasicVSR++ Performance Comparison (FP32 vs FP16)')
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    output_path = results_dir / 'performance_chart.png'
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")
    plt.show()

if __name__ == '__main__':
    main()