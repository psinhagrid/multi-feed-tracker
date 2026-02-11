"""Image loading utilities."""

import requests
from PIL import Image
from pathlib import Path


def load_image(source):
    """
    Load an image from a file path or URL.
    
    Args:
        source (str): File path or URL to the image
        
    Returns:
        PIL.Image: Loaded image
        
    Raises:
        ValueError: If the source is invalid
        FileNotFoundError: If the local file doesn't exist
    """
    if source.startswith(('http://', 'https://')):
        # Load from URL
        try:
            response = requests.get(source, stream=True)
            response.raise_for_status()
            return Image.open(response.raw)
        except Exception as e:
            raise ValueError(f"Failed to load image from URL: {e}")
    else:
        # Load from local path
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {source}")
        return Image.open(source)
