#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video Super-Resolution Pipeline using BasicVSR++
完全基于 OpenCV 实现视频拆帧和合帧，无需 ffmpeg
不再直接导入 mmedit，避免 DLL 错误
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
import cv2
from tqdm import tqdm
import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def extract_frames_opencv(video_path, output_dir, target_fps=None):
    """
    使用 OpenCV 从视频中提取帧，保存为 PNG 图片
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if target_fps is not None and target_fps < original_fps:
        sample_interval = int(round(original_fps / target_fps))
    else:
        sample_interval = 1
        target_fps = original_fps
    
    frame_idx = 0
    saved_idx = 0
    pbar = tqdm(total=total_frames_original, desc="提取视频帧")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            out_path = output_dir / f"{saved_idx:08d}.png"
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1
        frame_idx += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    print(f"提取完成: 原视频 {total_frames_original} 帧 -> {saved_idx} 帧 (采样间隔 {sample_interval})")
    return original_fps, saved_idx

def rename_output_files(output_dir):
    """合并滑动窗口输出的帧，重新编号"""
    output_dir = Path(output_dir)
    frame_files = sorted(output_dir.glob("*.png"), key=lambda x: int(x.stem))
    for new_idx, file_path in enumerate(frame_files):
        new_name = f"{new_idx:08d}.png"
        file_path.rename(output_dir / new_name)
    print(f"输出文件已重新编号，共 {len(frame_files)} 帧")

def run_super_resolution(config_path, checkpoint_path, input_dir, output_dir, device_id=0, window_size=30):
    """
    滑动窗口方式调用 BasicVSR++ 官方推理脚本，每个窗口输出到独立子目录，最后合并
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(input_dir.glob("*.png"))
    total_frames = len(frame_paths)
    if total_frames == 0:
        print("错误：输入目录中没有找到 PNG 图片")
        return

    num_windows = (total_frames + window_size - 1) // window_size
    print(f"总帧数: {total_frames}，窗口大小: {window_size}，共 {num_windows} 批")

    # 创建临时目录存放所有窗口的输出
    temp_output_root = output_dir / "_temp_windows"
    if temp_output_root.exists():
        shutil.rmtree(temp_output_root)
    temp_output_root.mkdir(parents=True)

    demo_script = Path(__file__).parent.parent / "demo" / "restoration_video_demo.py"
    if not demo_script.exists():
        raise RuntimeError(f"找不到推理脚本: {demo_script}")

    # 进度条
    pbar = tqdm(total=total_frames, desc="超分进度", unit="帧", ncols=80)

    for start_idx in range(0, total_frames, window_size):
        end_idx = min(start_idx + window_size, total_frames)
        current_window_path = input_dir / f"window_{start_idx}_{end_idx}"
        current_window_path.mkdir(exist_ok=True)

        # 复制当前窗口的帧
        for i, frame_path in enumerate(frame_paths[start_idx:end_idx]):
            target_path = current_window_path / f"{i:08d}.png"
            shutil.copy2(frame_path, target_path)

        # 为该窗口创建独立的输出子目录
        window_out_dir = temp_output_root / f"out_{start_idx:06d}_{end_idx:06d}"
        window_out_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable, str(demo_script),
            str(config_path), str(checkpoint_path),
            str(current_window_path), str(window_out_dir),
            "--device", str(device_id),
            "--max-seq-len", str(window_size)
        ]

        # 静默运行子进程，避免输出干扰进度条
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"\n窗口 {start_idx//window_size + 1} 处理失败，错误码: {e.returncode}")
            raise

        # 清理临时输入目录
        shutil.rmtree(current_window_path)
        torch.cuda.empty_cache()

        # 更新进度条
        pbar.update(end_idx - start_idx)
        pbar.set_postfix_str(f"窗口 {start_idx//window_size + 1}/{num_windows}")

    pbar.close()

    # 合并所有窗口的输出文件并重新编号
    print("\n正在合并输出帧...")
    all_frames = []
    # 按窗口序号排序，确保顺序正确
    for win_dir in sorted(temp_output_root.glob("out_*")):
        frames = sorted(win_dir.glob("*.png"), key=lambda x: int(x.stem))
        all_frames.extend(frames)

    if len(all_frames) != total_frames:
        print(f"警告：期望 {total_frames} 帧，实际得到 {len(all_frames)} 帧")

    # 复制到最终输出目录并重命名
    for new_idx, src_path in enumerate(all_frames):
        dst_path = output_dir / f"{new_idx:08d}.png"
        shutil.copy2(src_path, dst_path)

    # 删除临时目录
    shutil.rmtree(temp_output_root)
    print(f"合并完成，共 {len(all_frames)} 帧，已保存到 {output_dir}")

def merge_frames_to_video_opencv(frame_dir, output_video_path, fps):
    """
    使用 OpenCV 将帧序列合成为视频
    """
    frame_dir = Path(frame_dir)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 获取第一帧的尺寸
    first_frame_path = list(frame_dir.glob("*.png"))[0]
    frame = cv2.imread(str(first_frame_path))
    h, w = frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))
    
    frame_files = sorted(frame_dir.glob("*.png"), key=lambda x: int(x.stem))
    for fpath in tqdm(frame_files, desc="合成视频"):
        img = cv2.imread(str(fpath))
        out.write(img)
    out.release()
    print(f"视频已保存: {output_video_path}")

def cleanup_dirs(dirs):
    for d in dirs:
        if d and Path(d).exists():
            shutil.rmtree(d)
            print(f"已删除临时目录: {d}")

def main():
    parser = argparse.ArgumentParser(description="视频超分流水线 (纯 OpenCV 版，无需 ffmpeg)")
    parser.add_argument("input_video", type=str, help="输入视频路径")
    parser.add_argument("output_video", type=str, help="输出视频路径")
    parser.add_argument("--config", type=str, default="configs/basicvsr_plusplus_reds4.py",
                        help="模型配置文件路径")
    parser.add_argument("--checkpoint", type=str, default="chkpts/basicvsr_plusplus_reds4.pth",
                        help="模型权重文件路径")
    parser.add_argument("--target_fps", type=int, default=None,
                        help="输出视频帧率（若不指定则使用原视频帧率）")
    parser.add_argument("--device", type=int, default=0, help="GPU 设备 ID")
    parser.add_argument("--temp_dir", type=str, default="./temp_sr",
                        help="临时目录，存放中间帧")
    parser.add_argument("--keep_temp", action="store_true",
                        help="保留临时目录（默认处理完后删除）")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    input_video = Path(args.input_video)
    output_video = Path(args.output_video)
    config_path = base_dir / args.config
    checkpoint_path = base_dir / args.checkpoint
    temp_root = Path(args.temp_dir)
    input_frames_dir = temp_root / "input_frames"
    output_frames_dir = temp_root / "output_frames"
    
    if not input_video.exists():
        print(f"错误: 输入视频不存在 {input_video}")
        sys.exit(1)
    if not config_path.exists():
        print(f"错误: 配置文件不存在 {config_path}")
        sys.exit(1)
    if not checkpoint_path.exists():
        print(f"错误: 模型权重不存在 {checkpoint_path}")
        sys.exit(1)
    
    try:
        original_fps, total_frames = extract_frames_opencv(
            input_video, input_frames_dir, args.target_fps
        )
        fps = args.target_fps if args.target_fps else original_fps
        
        run_super_resolution(config_path, checkpoint_path,
                             input_frames_dir, output_frames_dir,
                             args.device, window_size=5)
        
        merge_frames_to_video_opencv(output_frames_dir, output_video, fps)
        
        print(f"\n✅ 成功！超分视频已保存至: {output_video}")
    
    finally:
        if not args.keep_temp:
            cleanup_dirs([input_frames_dir, output_frames_dir])
            if temp_root.exists() and not any(temp_root.iterdir()):
                temp_root.rmdir()
        else:
            print(f"临时目录保留在: {temp_root}")

if __name__ == "__main__":
    main()