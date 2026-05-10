import os
import cv2
import shutil
from pathlib import Path

video_path = Path("data/my_input.mp4")
# 2. 你想要存放“完美”帧序列的目录
output_dir = Path("temp_sr/input_frames_corrected")
# -----------------------

# 如果输出目录已存在，先删除它，确保是一个全新的开始
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

# 使用 OpenCV 读取视频
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {video_path}")

frame_count = 0
print(f"开始处理视频: {video_path}")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 按照脚本要求的格式命名：8位数字，从0开始，后缀为.png
    # 例如：00000000.png, 00000001.png ...
    save_path = output_dir / f"{frame_count:08d}.png"
    cv2.imwrite(str(save_path), frame)
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"已处理 {frame_count} 帧...")

cap.release()
print(f"完成！共提取 {frame_count} 帧，保存至 {output_dir}")