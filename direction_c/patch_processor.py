#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patch processing for Direction C:
- Extract patches from low-confidence regions
- Batch process all patches via a temporary folder (single ESRGAN call)
- Fuse back with feathering to avoid artifacts
"""

import cv2
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple


def extract_patches_from_boxes(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    expand: int = 10,
    min_patch_size: int = 32
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    h, w = image.shape[:2]
    patches = []
    new_boxes = []
    for (x1, y1, x2, y2) in boxes:
        nx1 = max(0, x1 - expand)
        ny1 = max(0, y1 - expand)
        nx2 = min(w, x2 + expand)
        ny2 = min(h, y2 + expand)
        if (nx2 - nx1) < min_patch_size or (ny2 - ny1) < min_patch_size:
            continue
        patch = image[ny1:ny2, nx1:nx2]
        patches.append(patch)
        new_boxes.append((nx1, ny1, nx2, ny2))
    return patches, new_boxes


def enhance_patches_batch_folder(
    patches: List[np.ndarray],
    exe_path: str,
    model_name: str,
    scale: int = 2
) -> List[np.ndarray]:
    if not patches:
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        for idx, patch in enumerate(patches):
            fname = input_dir / f"patch_{idx:04d}.png"
            cv2.imwrite(str(fname), patch)

        cmd = [
            exe_path,
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-s", str(scale),
            "-n", model_name,
            "-f", "png",
            "-v",
            "-g", "0"
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        except Exception as e:
            print(f"Batch ESRGAN failed: {e}, falling back to single patch mode")
            # Fallback to sequential single-patch enhancement
            enhanced = []
            for p in patches:
                with tempfile.TemporaryDirectory() as tmp2:
                    inp = Path(tmp2) / "in.png"
                    outp = Path(tmp2) / "out.png"
                    cv2.imwrite(str(inp), p)
                    cmd_single = cmd[:]  # copy base command
                    cmd_single[cmd_single.index("-i")+1] = str(inp)
                    cmd_single[cmd_single.index("-o")+1] = str(outp)
                    try:
                        subprocess.run(cmd_single, capture_output=True, check=True, timeout=30)
                        en = cv2.imread(str(outp))
                        if en is not None and scale != 1:
                            h, w = p.shape[:2]
                            en = cv2.resize(en, (w, h), interpolation=cv2.INTER_LANCZOS4)
                        enhanced.append(en if en is not None else p)
                    except:
                        enhanced.append(p)
            return enhanced

        enhanced = []
        for idx, orig_patch in enumerate(patches):
            out_path = output_dir / f"patch_{idx:04d}.png"
            if not out_path.exists():
                enhanced.append(orig_patch)
                continue
            en_img = cv2.imread(str(out_path))
            if en_img is None:
                enhanced.append(orig_patch)
                continue

            if scale != 1:
                h, w = orig_patch.shape[:2]
                target_w = scale * w
                target_h = scale * h
                if en_img.shape[1] != target_w or en_img.shape[0] != target_h:
                    en_img = cv2.resize(en_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                en_img = cv2.resize(en_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
            enhanced.append(en_img)
        return enhanced


def feather_merge(
    base: np.ndarray,
    patch: np.ndarray,
    box: Tuple[int, int, int, int],
    feather_radius: int = 15
) -> np.ndarray:
    x1, y1, x2, y2 = box
    h_patch = y2 - y1
    w_patch = x2 - x1

    weight = np.ones((h_patch, w_patch), dtype=np.float32)
    if feather_radius > 0:
        rows, cols = np.indices((h_patch, w_patch))
        dist_top = rows
        dist_bottom = h_patch - 1 - rows
        dist_left = cols
        dist_right = w_patch - 1 - cols
        min_dist = np.minimum(np.minimum(dist_top, dist_bottom),
                              np.minimum(dist_left, dist_right))
        weight = np.clip(min_dist / feather_radius, 0.0, 1.0)
        weight = weight ** 2

    for c in range(3):
        base_region = base[y1:y2, x1:x2, c].astype(np.float32)
        patch_region = patch[:, :, c].astype(np.float32)
        blended = base_region * (1 - weight) + patch_region * weight
        base[y1:y2, x1:x2, c] = np.clip(blended, 0, 255).astype(np.uint8)
    return base


def enhance_frame_with_esrgan_patches(
    frame: np.ndarray,
    confidence_map: np.ndarray,
    esrgan_exe_path: str,
    conf_threshold: float = 0.3,
    min_area: int = 100,
    expand_patch: int = 10,
    esrgan_scale: int = 2,
    feather_radius: int = 15,
    model_name: str = "realesr-animevideov3"
) -> np.ndarray:
    from uncertainty import threshold_low_confidence

    boxes = threshold_low_confidence(confidence_map, threshold=conf_threshold, min_area=min_area)
    if not boxes:
        return frame

    patches, expanded_boxes = extract_patches_from_boxes(frame, boxes, expand=expand_patch)
    if not patches:
        return frame

    enhanced_patches = enhance_patches_batch_folder(patches, esrgan_exe_path, model_name, scale=esrgan_scale)

    result = frame.copy()
    for enhanced, box in zip(enhanced_patches, expanded_boxes):
        result = feather_merge(result, enhanced, box, feather_radius=feather_radius)

    return result