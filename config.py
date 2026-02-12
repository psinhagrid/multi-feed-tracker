"""Configuration settings for the object detection system."""

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
