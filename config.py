"""Configuration settings for the object detection system."""

# Model configuration
MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# Detection thresholds
DETECTION_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3

# Video processing settings
DETECTION_FRAME_INTERVAL = 20  # Run detection every N frames
RESIZE_WIDTH = 640  # Resize frame width for faster detection (maintains aspect ratio)

# Visualization settings
BOX_COLOR = 'red'
BOX_LINEWIDTH = 2
TEXT_FONTSIZE = 10
TEXT_COLOR = 'red'
TEXT_BACKGROUND = 'white'

# Default labels (can be overridden)
DEFAULT_LABELS = ["Person"]
