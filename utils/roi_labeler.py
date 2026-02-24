"""LLM-based ROI labeling utilities."""

import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
from dotenv import load_dotenv

# Import here to avoid circular imports; app.py imports from llm
load_dotenv()


def label_roi_from_frame(frame, bbox_px: tuple, api_key: Optional[str] = None) -> str:
    """
    Crop ROI from frame, send to LLM for labeling.
    Returns short label (e.g. "person", "car") or "tracked object" on failure.
    """
    from llm.image_describer import describe_image

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    x_px, y_px, w_px, h_px = bbox_px
    y_end = min(y_px + h_px, frame.shape[0])
    x_end = min(x_px + w_px, frame.shape[1])
    roi = frame[y_px:y_end, x_px:x_end]

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, roi)
            tmp_path = tmp.name
        label = describe_image(tmp_path, api_key)
        os.unlink(tmp_path)
        return label
    except Exception as e:
        print(f"LLM labeling failed: {e}")
        return "tracked object"


def save_roi_debug(frame, bbox_px: tuple, session_id: str, tracker_id: str) -> Path:
    """Save ROI to debug folder for visualization. Returns path."""
    x_px, y_px, w_px, h_px = bbox_px
    y_end = min(y_px + h_px, frame.shape[0])
    x_end = min(x_px + w_px, frame.shape[1])
    roi = frame[y_px:y_end, x_px:x_end]

    debug_dir = Path("debug_rois")
    debug_dir.mkdir(exist_ok=True)
    debug_path = debug_dir / f"roi_{session_id}_{tracker_id}.jpg"
    cv2.imwrite(str(debug_path), roi)
    return debug_path
