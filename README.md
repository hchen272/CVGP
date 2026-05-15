# Video Super-Resolution: From Handcrafted to Uncertainty-Aware Hybrid Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

Final Project for AIAA 3201 Introduction to Computer Vision

Team: Hongliang Chen (hchen272@connect.hkust-gz.edu.cn), Boyong Hou (bhou204@connect.hkust-gz.edu.cn)

This repository provides a complete video super‑resolution (VSR) pipeline that follows the three‑part roadmap:

- Part 1 (Handcrafted) – Bicubic/Lanczos interpolation, temporal averaging, unsharp masking, SRCNN.

- Part 2 (AI‑driven) – BasicVSR++ (bidirectional propagation + deformable alignment) and Real‑ESRGAN standalone.

- Part 3 (Exploration – Direction C) – Two uncertainty‑aware hybrid methods:

  - UAH (Uncertainty‑Aware Hybrid) – Hard‑threshold patch selection + ESRGAN enhancement.

  - UGF (Uncertainty‑Guided Fusion) – Soft pixel‑wise fusion based on temporal residual, texture complexity, and structural confidence.

Additionally, we implement FP16 inference and ROI‑based selective processing to accelerate BasicVSR++.

All experiments are performed on the REDS dataset (four sequences: ```000```, ```011```, ```015```, ```020```). Evaluation metrics include PSNR, SSIM, LPIPS, tLPIPS, and FID.

## Repository Structure
```text
├── baseline_implementation/            # Part 1 & Part 2 core scripts
│   ├── part1/                          # Handcrafted baseline pipeline
│   │   └── main_pipeline_part1.py
│   ├── generate_lr_from_gt.py          # Downscale GT to LR (if needed)
│   └── evaluate_full.py                # Unified evaluation script (supports multiple methods)
├── BasicVSR_PlusPlus/                  # BasicVSR++ code (clone from official repo)
│   ├── scripts/
│   │   └── process_reds_preserve_structure.py
│   ├── inference_optimize/             # Model loader (provided)
│   └── data/reds/                      # Place REDS 
|   ├── configs/basicvsr_plusplus_reds4.py # Modified
dataset here
├── direction_c/                        # Direction C (UAH & UGF) and variants
│   ├── uncertainty.py                  # Heuristic confidence map estimation
│   ├── patch_processor.py              # Patch extraction, batch ESRGAN, feathering
│   ├── run_direction_c.py              # Main UAH entry point
│   ├── weighted_fusion.py              # Global weighted fusion of BasicVSR++ and UAH
│   ├── run_multiple_thresholds.py      # Sweep confidence thresholds (0.2–0.5)
│   ├── evaluate_thresholds.py          # Evaluate all thresholds and generate plots
│   └── results/                        # Outputs (frames, videos, evaluation)
├── ESRGAN/                             # Real-ESRGAN executable (download from official release)
│   ├── realesrgan-ncnn-vulkan.exe
│   └── models/                         # .bin and .param files (e.g., realesr-animevideov3)
├── process_video_all.py                # Single‑video pipeline (extract frames → super‑resolve → video)
├── environment.yml                     # Conda environment specification
└── README.md
```
## Setup

### 1. Clone the repository
```bash
git clone https://github.com/hchen272/Video-Super-Resolution-Based-on-BasicVSR_PlusPlus-and-Real-ESRGAN.git
cd Video-Super-Resolution-Based-on-BasicVSR_PlusPlus-and-Real-ESRGAN
```

### 2. Create conda environment
We recommend using Python 3.9 with PyTorch 1.13 and CUDA 11.7.

```bash
conda create -n vsr python=3.9
conda activate vsr
```
Install PyTorch (adjust CUDA version according to your system):

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```
Install other dependencies:

```bash
pip install opencv-python tqdm lpips scikit-image matplotlib
```

### 3. Get BasicVSR++ code
The ```BasicVSR_PlusPlus/``` folder already contains the experimental implementation. If missing, clone it:

```bash
# Install PyTorch (already done)
pip install openmim
mim install mmcv-full
git clone https://github.com/ckkelvinchan/BasicVSR_PlusPlus.git
cd BasicVSR_PlusPlus
pip install -v -e .
```
Place the downloaded pre‑trained REDS4 checkpoint into ```BasicVSR_PlusPlus/chkpts/``` (download from [BasicVSR++ official page](https://github.com/ckkelvinchan/BasicVSR_PlusPlus)).

### 4. Get Real-ESRGAN executable
Download ```realesrgan-ncnn-vulkan``` from [the official release](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases). Extract and place the executable (```realesrgan-ncnn-vulkan.exe```) and model files into the ESRGAN/ folder. We use the ```realesr-animevideov3``` model for best alignment.

Dataset Preparation
We use the REDS dataset (training/validation splits). For this project, only four validation sequences (```000```, ```011```, ```015```, ```020```) are required.

Organize the data as follows:

```text
BasicVSR_PlusPlus/data/reds/
├── input_videos/          # Low‑resolution input frames (PNG sequences)
│   ├── 000/
│   │   ├── 00000000.png
│   │   └── ...
│   ├── 011/
│   └── ...
└── gt_videos/             # Ground truth HR frames (same structure)
    ├── 000/
    └── ...
```

If you have HR ground truth videos, you can generate LR inputs using ```baseline_implementation/generate_lr_from_gt.py``` (downscale factor 4).

## Running the Pipelines
All commands should be executed from the repository root (```Video-Super-Resolution-Based-on-BasicVSR_PlusPlus-and-Real-ESRGAN```).

### Part 1 – Handcrafted Baselines
Run all Part 1 methods (bicubic, Lanczos, temporal averaging, unsharp, SRCNN) on the REDS sequences:

```bash
python baseline_implementation/part1/main_pipeline_part1.py
```
The script will read LR frames from ```BasicVSR_PlusPlus/data/reds/input_videos/``` and save results to ```BasicVSR_PlusPlus/outputs/``` (subfolders per method per sequence).

Output example: ```BasicVSR_PlusPlus/outputs/000/bicubic_x4/00000000.png```, etc.

## Part 2 – BasicVSR++ (FP32/FP16)
Process a folder of LR frames with BasicVSR++:

```bash
python BasicVSR_PlusPlus/scripts/process_reds_preserve_structure.py \
    BasicVSR_PlusPlus/data/reds/input_videos \
    BasicVSR_PlusPlus/results/fp32 \
    --precision fp32
```
For FP16 inference, change ```--precision``` fp16. Results are saved as image sequences in ```results/fp32/``` or ```results/fp16/```.

### Part 3 – Direction C (UAH)
The main uncertainty‑aware hybrid (patch‑based) is ```direction_c/run_direction_c.py```. To run with default parameters (θ=0.3):

```bash
python direction_c/run_direction_c.py \
    --input_root BasicVSR_PlusPlus/results/fp32 \
    --output_root BasicVSR_PlusPlus/results/direction_c \
    --esrgan_exe ESRGAN/realesrgan-ncnn-vulkan.exe \
    --esrgan_model realesr-animevideov3 \
    --conf_threshold 0.3 \
    --save_vis
```
You can also sweep multiple thresholds using ```run_multiple_thresholds.py``` (edit the paths inside first):

```bash
python direction_c/run_multiple_thresholds.py
```
### Additional Experiments
#### Weighted Fusion (BasicVSR++ + UAH)
```bash
python direction_c/weighted_fusion.py --alpha 0.6
```
This produces a pixel‑wise weighted average between BasicVSR++ and UAH outputs. The best LPIPS is achieved at α=0.6.

#### Process a Single Video (End‑to‑End)
For arbitrary input videos (e.g., a self‑captured clip), use:

```bash
python process_video_all.py \
    --input_video path/to/your_video.mp4 \
    --output_dir ./results \
    --method directionc
```
Available methods: ```basicvsr```, ```directionc```, ```esrgan```. The script extracts frames, runs the selected method, and outputs both a video (MP4) and the frame sequence.

#### Standalone ESRGAN (4× upscaling)
```bash
ESRGAN/realesrgan-ncnn-vulkan.exe -i input_folder -o output_folder -s 4 -n realesr-animevideov3 -f png -v -g 0
```

## Evaluation
We provide a unified evaluation script ```baseline_implementation/evaluate_full.py``` that computes PSNR, SSIM, LPIPS, tLPIPS, and FID for all methods.

First, ensure you have the ground truth frames in ```BasicVSR_PlusPlus/data/reds/gt_videos/```. Then run:

```bash
python baseline_implementation/evaluate_full.py
```
The script automatically detects output folders (```outputs/```, ```results/fp32/```, ```results/direction_c/```, etc.) and generates:

- ```per_video_metrics.json``` – metrics per video per method.

- ```average_metrics.json``` – averages across videos.

- Bar charts (```psnr_comparison.png```, ```lpips_comparison.png```, etc.) in ```BasicVSR_PlusPlus/results/evaluation/```.

To generate charts from an existing JSON file (e.g., after running on a different machine), use:

```bash
python baseline_implementation/generate_charts_from_json.py --json_path path/to/average_metrics.json
```
For evaluation of different confidence thresholds (UAH variants), run:

```bash
python direction_c/evaluate_thresholds.py
This will produce per‑threshold metrics and plots in direction_c/results/evaluation_thresholds/.
```

## Results Summary
The table below reports average metrics over four REDS sequences (000, 011, 015, 020).
UAH = Uncertainty‑Aware Hybrid (θ=0.3), UGF = Uncertainty‑Guided Fusion (soft fusion).

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | tLPIPS ↓ | FID ↓ |
|--------|--------|--------|---------|----------|-------|
| Bicubic 4x | 24.51 | 0.6775 | 0.4539 | 0.1971 | 52.12 |
| Lanczos 4x | 24.63 | 0.6837 | 0.4530 | 0.1956 | 44.06 |
| BasicVSR++ (FP32) | 29.32 | 0.8729 | 0.1406 | 0.2160 | 1.44 |
| BasicVSR++ (FP16) | 28.79 | 0.8578 | 0.1440 | 0.2189 | 1.47 |
| UAH | 27.30 | 0.8502 | 0.1460 | 0.2279 | 8.93 |
| Weighted Fusion (α=0.6) | 28.91 | 0.8690 | 0.1357 | 0.2195 | 2.95 |
| UGF | 26.11 | 0.7865 | 0.2144 | – | – |
| ESRGAN (4x) | 24.22 | 0.6572 | 0.3772 | 0.2135 | 97.12 |
**Best LPIPS** is achieved by weighted fusion (α=0.6), demonstrating that a moderate blend of BasicVSR++ and generative enhancement improves perceptual quality while preserving high fidelity.

## Acceleration Experiments
### FP16 Inference
BasicVSR++ FP16 provides 4.57× speedup with negligible quality loss (PSNR −0.04 dB, SSIM −0.0018).
Run with --precision fp16 as shown above.

### ROI‑Based Selective Processing
We implement a motion‑driven ROI pipeline. BasicVSR++ is applied only to the region of interest (≈13% of the frame), while the background is bicubic upsampled. Results:

- Total time: 186.5 s → 73.1 s (2.55× acceleration)

- Peak GPU memory: 7808 MiB → 4726 MiB

- PSNR: 29.53 → 29.49 (almost unchanged)

- SSIM: 0.8482 → 0.8557

To run the ROI experiment, refer to the code in team member [Boyong Hou's repository](https://github.com/bhou204/vsrplusplusproject).

## Notes & Troubleshooting

- Real-ESRGAN alignment shift: The model realesr-animevideov3 works correctly; realesrgan-x4plus may cause pixel shifts. We recommend using the anime model.

- Out‑of‑memory: Reduce batch size or use FP16. The patch‑based Direction C processes only low‑confidence areas, which reduces memory compared to full‑frame ESRGAN.

## References
We cite the following works:

[1] Dong et al. – SRCNN, TPAMI 2015

[2] Ledig et al. – SRGAN, CVPR 2017

[3] Lim et al. – EDSR, CVPRW 2017

[4] Wang et al. – EDVR, CVPRW 2019

[5] Tian et al. – TDAN, CVPR 2020

[6] Chan et al. – BasicVSR, CVPR 2021

[7] Chan et al. – BasicVSR++, CVPR 2022

[8] Wang et al. – Real-ESRGAN, ICCVW 2021

[9] Saharia et al. – SR3, TPAMI 2022

[10] Lipman et al. – Flow Matching, ICLR 2023

[11] Zhang et al. – ControlNet, ICCV 2023

[12] Nah et al. – REDS dataset, CVPRW 2019

[13] Xue et al. – Vimeo-90K, IJCV 2019

[14] Wang et al. – PSNR/SSIM, TIP 2004

[15] Zhang et al. – LPIPS, CVPR 2018

[16] Rombach et al. – Stable Diffusion, CVPR 2022

[17] Hu et al. – LoRA, ICLR 2022

[18] Chu et al. – tLPIPS, TOG 2020
