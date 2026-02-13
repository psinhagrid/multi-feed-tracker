"""Utility functions for the application."""

import torch


def get_device():
    """
    Detect and return the best available device.
    
    Returns:
        str: Device string ('mps', 'cuda', or 'cpu')
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def print_device_info():
    """Print information about available devices."""
    device = get_device()
    print(f"Using device: {device}")
    
    if device == "mps":
        print("Apple Silicon GPU (MPS) detected")
    elif device == "cuda":
        print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected, using CPU")
    
    return device
