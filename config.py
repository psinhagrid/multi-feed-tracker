"""Configuration settings for the object detection system."""

import os
from pathlib import Path

# Model configuration
MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# Detection thresholds
DETECTION_THRESHOLD = 0.65  # Increased to filter out low confidence detections
TEXT_THRESHOLD = 0.4

# Tracking settings
TRACK_THRESH = 0.6   # ByteTrack confidence threshold (should be ≤ DETECTION_THRESHOLD)
TRACK_BUFFER = 90    # Number of frames to keep lost tracks before removing (frame retention ~3 seconds at 30fps)
MATCH_THRESH = 0.75  # IoU threshold for matching (higher = stricter matching, fewer ID switches)

# Video processing settings
DETECTION_FRAME_INTERVAL = 20  # Run detection every N frames (balanced speed/accuracy)
RESIZE_WIDTH = 640  # Resize frame width for faster detection (maintains aspect ratio)

# Visualization settings
BOX_COLOR = 'red'
BOX_LINEWIDTH = 2
TEXT_FONTSIZE = 10
TEXT_COLOR = 'red'
TEXT_BACKGROUND = 'white'

# Default labels (can be overridden)
DEFAULT_LABELS = ["Person"]

# Re-ID (fast-reid) settings
# Set FAST_REID_PATH to the cloned fast-reid repo root, e.g. /path/to/fast-reid
# Or set REID_CONFIG_PATH and REID_WEIGHTS_PATH directly via env
FAST_REID_PATH = os.getenv("FAST_REID_PATH", str(Path(__file__).parent / "fast-reid"))
REID_CONFIG_PATH = os.getenv(
    "REID_CONFIG_PATH",
    str(Path(FAST_REID_PATH) / "configs" / "Market1501" / "bagtricks_R50.yml"),
)
REID_WEIGHTS_PATH = os.getenv(
    "REID_WEIGHTS_PATH",
    str(Path(FAST_REID_PATH) / "model.pth"),
)
REID_DEVICE = os.getenv("REID_DEVICE", "cuda")  # or "cpu", "mps"
REID_MATCH_THRESHOLD = float(os.getenv("REID_MATCH_THRESHOLD", "0.5"))
