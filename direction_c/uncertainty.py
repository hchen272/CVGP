#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Uncertainty estimation module for video super-resolution.

Supported methods:
- mc_dropout: Monte Carlo Dropout (requires model with dropout layers)
- texture: Local variance as uncertainty proxy (fast, no model needed)
- gradient: Gradient magnitude based uncertainty
- combined: Weighted combination of texture, gradient, and edge density

Usage example:
    from direction_c.uncertainty import UncertaintyEstimator
    estimator = UncertaintyEstimator(model, device='cuda:0')
    conf_map, var_map = estimator.compute_confidence_map(lr_tensor, method='combined')
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Union, Optional
from pathlib import Path
import sys

# Add BasicVSR_PlusPlus to path to import model_loader
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "BasicVSR_PlusPlus"))

# OpenCV is used for image processing; import here but allow fallback
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV (cv2) not found. Some uncertainty methods will be limited.")


class UncertaintyEstimator:
    """
    Main class for computing pixel-wise confidence maps.
    """

    def __init__(self, model: Optional[nn.Module] = None, device: str = "cuda:0"):
        """
        Args:
            model: BasicVSR++ model (already loaded). If None, only heuristic
                   methods (texture, gradient, combined) can be used.
            device: 'cuda:0' or 'cpu'
        """
        self.model = model
        self.device = device
        if model is not None:
            self.model.eval()

    def compute_confidence_map(
        self,
        lr_tensor: torch.Tensor,
        method: str = "combined",
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence map from input low-resolution tensor.

        Args:
            lr_tensor: Shape [1, T, C, H, W] (T = number of frames)
            method: 'mc_dropout', 'texture', 'gradient', 'combined'
            **kwargs: Additional method-specific parameters.

        Returns:
            confidence_map: 2D numpy array (H_out, W_out) values in [0,1], high = confident
            raw_uncertainty: 2D numpy array raw uncertainty (e.g., variance, texture)
        """
        if method == "mc_dropout":
            return self._mc_dropout_uncertainty(lr_tensor, **kwargs)
        elif method == "texture":
            return self._texture_based_uncertainty(lr_tensor, **kwargs)
        elif method == "gradient":
            return self._gradient_based_uncertainty(lr_tensor, **kwargs)
        elif method == "combined":
            return self._combined_uncertainty(lr_tensor, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _mc_dropout_uncertainty(
        self,
        lr_tensor: torch.Tensor,
        num_passes: int = 10,
        keep_fp16: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout uncertainty.
        """
        if self.model is None:
            raise RuntimeError("Model required for mc_dropout method.")

        # Enable dropout layers temporarily
        def enable_dropout(m):
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

        def disable_dropout(m):
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.eval()

        self.model.apply(enable_dropout)

        predictions = []
        for _ in range(num_passes):
            with torch.no_grad():
                if keep_fp16 and lr_tensor.dtype == torch.float16:
                    with torch.cuda.amp.autocast():
                        out = self.model.forward_test(lr_tensor)["output"]
                else:
                    out = self.model.forward_test(lr_tensor)["output"]
                predictions.append(out.cpu())

        self.model.apply(disable_dropout)

        # predictions: list of [1, T, C, H, W]
        pred_stack = torch.stack(predictions, dim=0)          # [num, 1, T, C, H, W]
        variance = pred_stack.var(dim=0, unbiased=False)      # [1, T, C, H, W]
        # Use last frame, average over channels
        var_map = variance[0, -1].mean(dim=0).numpy()          # [H, W]

        # Convert variance to confidence: exponential decay
        var_norm = var_map / (var_map.max() + 1e-8)
        confidence = np.exp(-5 * var_norm)
        return confidence, var_map

    def _texture_based_uncertainty(
        self,
        lr_tensor: torch.Tensor,
        patch_size: int = 7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use local variance as proxy for uncertainty.
        """
        if not HAS_CV2:
            raise ImportError("OpenCV required for texture-based uncertainty.")
        # Use last frame, convert to grayscale
        frame = lr_tensor[0, -1].cpu().numpy()                 # [C, H, W]
        if frame.shape[0] == 3:
            gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
        else:
            gray = frame[0]

        kernel = np.ones((patch_size, patch_size)) / (patch_size**2)
        local_mean = cv2.filter2D(gray, -1, kernel)
        local_sq_mean = cv2.filter2D(gray**2, -1, kernel)
        local_var = local_sq_mean - local_mean**2

        var_norm = local_var / (local_var.max() + 1e-8)
        confidence = np.exp(-5 * var_norm)
        return confidence, var_norm

    def _gradient_based_uncertainty(
        self,
        lr_tensor: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use gradient magnitude as uncertainty proxy.
        """
        if not HAS_CV2:
            raise ImportError("OpenCV required for gradient-based uncertainty.")
        frame = lr_tensor[0, -1].cpu().numpy()
        if frame.shape[0] == 3:
            gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
        else:
            gray = frame[0]

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_norm = grad_mag / (grad_mag.max() + 1e-8)
        confidence = np.exp(-3 * grad_norm)
        return confidence, grad_norm

    def _combined_uncertainty(
        self,
        lr_tensor: torch.Tensor,
        texture_weight: float = 0.4,
        gradient_weight: float = 0.3,
        edge_weight: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine texture, gradient, and edge density.
        """
        if not HAS_CV2:
            raise ImportError("OpenCV required for combined uncertainty.")

        frame = lr_tensor[0, -1].cpu().numpy()
        if frame.shape[0] == 3:
            gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
        else:
            gray = frame[0]

        # Texture (local variance)
        patch_size = 7
        kernel = np.ones((patch_size, patch_size)) / (patch_size**2)
        local_mean = cv2.filter2D(gray, -1, kernel)
        local_sq_mean = cv2.filter2D(gray**2, -1, kernel)
        texture = local_sq_mean - local_mean**2
        texture = texture / (texture.max() + 1e-8)

        # Gradient
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(grad_x**2 + grad_y**2)
        grad = grad / (grad.max() + 1e-8)

        # Edge density (Canny + Gaussian blur)
        edge = cv2.Canny((gray * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
        edge_density = cv2.GaussianBlur(edge, (15, 15), 3)

        combined = (texture_weight * texture +
                    gradient_weight * grad +
                    edge_weight * edge_density)
        combined = combined / (combined.max() + 1e-8)

        confidence = np.exp(-5 * combined)
        return confidence, combined


def threshold_low_confidence(
    confidence_map: np.ndarray,
    threshold: float = 0.4,
    min_area: int = 64
) -> List[Tuple[int, int, int, int]]:
    """
    Extract bounding boxes of low-confidence regions.

    Args:
        confidence_map: 2D array in [0,1]
        threshold: confidence below this value considered 'low'
        min_area: minimum area to keep a region

    Returns:
        List of (x1, y1, x2, y2) bounding boxes.
    """
    if not HAS_CV2:
        raise ImportError("OpenCV required for thresholding.")
    low_mask = (confidence_map < threshold).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(low_mask, connectivity=8)
    boxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            boxes.append((x, y, x + w, y + h))
    return boxes


def visualize_confidence_overlay(
    confidence_map: np.ndarray,
    image: np.ndarray,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay confidence map on image for visualization.
    """
    if not HAS_CV2:
        raise ImportError("OpenCV required for visualization.")
    conf_vis = (confidence_map * 255).astype(np.uint8)
    conf_vis = cv2.applyColorMap(conf_vis, colormap)
    overlay = cv2.addWeighted(image, 1 - alpha, conf_vis, alpha, 0)
    return overlay


if __name__ == "__main__":
    # Quick test without actual model
    print("Testing UncertaintyEstimator with mock data...")
    # Create dummy LR tensor
    dummy_lr = torch.randn(1, 5, 3, 128, 128)
    estimator = UncertaintyEstimator(model=None, device='cpu')
    for method in ['texture', 'gradient', 'combined']:
        conf, raw = estimator.compute_confidence_map(dummy_lr, method=method)
        print(f"{method}: confidence shape {conf.shape}, range [{conf.min():.3f}, {conf.max():.3f}]")
    print("Test passed. For mc_dropout, a real model is needed.")