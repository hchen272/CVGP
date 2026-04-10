import torch
from pathlib import Path
import sys

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mmedit.apis import init_model

def load_model(config_path, checkpoint_path, device='cuda:0'):
    """
    Load BasicVSR++ model from config and checkpoint.
    
    Args:
        config_path (str or Path): Path to config file.
        checkpoint_path (str or Path): Path to checkpoint file.
        device (str): Device to load model on, e.g., 'cuda:0' or 'cpu'.
    
    Returns:
        model: Loaded PyTorch model in eval mode.
    """
    model = init_model(str(config_path), str(checkpoint_path), device=torch.device(device))
    model.eval()
    return model