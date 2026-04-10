import time
import torch
import numpy as np
from tqdm import tqdm

def measure_inference_speed(model, input_tensor, num_warmup=10, num_iter=100, device='cuda:0'):
    """
    Measure average inference time and FPS of a PyTorch model using forward_test.
    Displays a progress bar during the measured iterations.

    Args:
        model: PyTorch model (already on correct device and in eval mode).
        input_tensor (torch.Tensor): Dummy input tensor of appropriate shape.
        num_warmup (int): Number of warmup iterations.
        num_iter (int): Number of measured iterations.
        device (str): Device identifier.

    Returns:
        dict: Contains 'avg_time_ms', 'fps', 'num_iter', 'input_shape'.
    """
    model.eval()
    # Ensure model and input are on same device
    input_tensor = input_tensor.to(device)

    # Warmup - use forward_test to avoid training path
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.forward_test(input_tensor)

    # Synchronize before timing
    if device.startswith('cuda'):
        torch.cuda.synchronize()

    # Measure with progress bar
    times = []
    with torch.no_grad():
        # Use tqdm to show progress
        iterator = tqdm(range(num_iter), desc="Inference progress", unit="iter", leave=False)
        for _ in iterator:
            start = time.perf_counter()
            _ = model.forward_test(input_tensor)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000.0
            times.append(elapsed_ms)
            # Optionally update postfix with current time
            iterator.set_postfix({"time_ms": f"{elapsed_ms:.1f}"})

    total_time = sum(times) / 1000.0  # convert to seconds
    avg_time_ms = sum(times) / len(times)
    fps = 1000.0 / avg_time_ms

    return {
        'avg_time_ms': avg_time_ms,
        'fps': fps,
        'num_iter': num_iter,
        'input_shape': tuple(input_tensor.shape)
    }

def measure_gpu_memory(device='cuda:0'):
    """
    Measure current GPU memory usage in MB.

    Args:
        device (str): CUDA device.

    Returns:
        dict: Contains 'allocated_mb', 'cached_mb', 'free_mb', 'total_mb'.
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    cached = torch.cuda.memory_reserved(device) / 1024**2
    total = torch.cuda.get_device_properties(device).total_memory / 1024**2
    free = total - allocated

    return {
        'allocated_mb': allocated,
        'cached_mb': cached,
        'free_mb': free,
        'total_mb': total
    }

def profile_model_memory(model, input_tensor, device='cuda:0'):
    """
    Run forward pass using forward_test and measure peak GPU memory usage.

    Args:
        model: PyTorch model.
        input_tensor: Input tensor.
        device: CUDA device.

    Returns:
        dict: Peak memory usage in MB.
    """
    torch.cuda.reset_peak_memory_stats(device)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        _ = model.forward_test(input_tensor)
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    return {'peak_memory_mb': peak_memory}