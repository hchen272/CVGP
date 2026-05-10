#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run Direction C with multiple confidence thresholds and save results to separate folders.

Usage:
    python run_multiple_thresholds.py
"""

import subprocess
import sys
from pathlib import Path

# Configuration
THRESHOLDS = [0.1]          # confidence thresholds to test
INPUT_ROOT = r"D:\VideoSuperResolution\BasicVSR_PlusPlus\results\fp32"   # BasicVSR++ results (HR)
OUTPUT_BASE = r"D:\VideoSuperResolution\direction_c\results\direction_c_variants"  # base output for all thresholds
ESRGAN_EXE = r"D:\VideoSuperResolution\ESRGAN\realesrgan-ncnn-vulkan.exe"
ESRGAN_MODEL = "realesr-animevideov3"
CONFIDENCE_METHOD = "combined"
MIN_AREA = 100
FEATHER_RADIUS = 15
ESRGAN_SCALE = 2
SAVE_VIS = True   # set to False to skip saving confidence overlays

# Path to run_direction_c.py script
SCRIPT_DIR = Path(__file__).parent
RUN_DC_SCRIPT = SCRIPT_DIR / "run_direction_c.py"

if not RUN_DC_SCRIPT.exists():
    print(f"Error: {RUN_DC_SCRIPT} not found.")
    sys.exit(1)

def main():
    OUTPUT_BASE_PATH = Path(OUTPUT_BASE)
    OUTPUT_BASE_PATH.mkdir(parents=True, exist_ok=True)

    for thresh in THRESHOLDS:
        output_dir = OUTPUT_BASE_PATH / f"threshold_{thresh}"
        print(f"\n=== Running Direction C with conf_threshold = {thresh} ===")
        print(f"Output will be saved to: {output_dir}")
        
        cmd = [
            sys.executable, str(RUN_DC_SCRIPT),
            "--input_root", INPUT_ROOT,
            "--output_root", str(output_dir),
            "--esrgan_exe", ESRGAN_EXE,
            "--esrgan_model", ESRGAN_MODEL,
            "--confidence_method", CONFIDENCE_METHOD,
            "--conf_threshold", str(thresh),
            "--min_area", str(MIN_AREA),
            "--feather_radius", str(FEATHER_RADIUS),
            "--esrgan_scale", str(ESRGAN_SCALE)
        ]
        if SAVE_VIS:
            cmd.append("--save_vis")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=False)  # let output print to console
        if result.returncode != 0:
            print(f"Error running Direction C for threshold {thresh}. Check logs.")
            # Optionally break or continue
        else:
            print(f"Finished threshold {thresh}")

    print("\nAll thresholds processed.")

if __name__ == "__main__":
    main()