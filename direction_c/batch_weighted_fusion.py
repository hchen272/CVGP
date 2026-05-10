#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch run weighted fusion for multiple alpha values.
"""
import subprocess
import sys
from pathlib import Path

BASICVSR_ROOT = r"D:\VideoSuperResolution\BasicVSR_PlusPlus\results\fp32"
DIRECTIONC_ROOT = r"D:\VideoSuperResolution\direction_c\results\direction_c_variants\threshold_0.3"
OUTPUT_BASE = r"D:\VideoSuperResolution\BasicVSR_PlusPlus\results\weighted_avg"

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

SCRIPT_DIR = Path(__file__).parent
FUSION_SCRIPT = SCRIPT_DIR / "weighted_fusion.py"

if not FUSION_SCRIPT.exists():
    print(f"Error: {FUSION_SCRIPT} not found.")
    sys.exit(1)


def run_fusion(alpha):
    cmd = [
        sys.executable, str(FUSION_SCRIPT),
        "--basicvsr_root", BASICVSR_ROOT,
        "--directionc_root", DIRECTIONC_ROOT,
        "--output_root", OUTPUT_BASE,
        "--alpha", str(alpha)
    ]
    print(f"\n>>> Fusion for alpha={alpha}")
    subprocess.run(cmd, check=True)


def main():
    for alpha in ALPHAS:
        run_fusion(alpha)
    print("\nAll fusions completed.")
    print("Now you can evaluate the results by adding them to evaluate_full.py or evaluate_thresholds.py")
    print("Example: add entries to methods dict pointing to:")
    for alpha in ALPHAS:
        out_dir = Path(OUTPUT_BASE) / f"alpha_{alpha:.2f}".replace('.', '_')
        print(f"  Weighted_alpha_{alpha:.2f}: {out_dir}")


if __name__ == "__main__":
    main()