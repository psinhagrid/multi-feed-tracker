"""Device selection utilities."""

import torch


def get_device():
    """
    Get the best available device for PyTorch computations.
    
    Priority: CUDA > MPS > CPU
    
    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    return device
