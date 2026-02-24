"""Coordinate conversion utilities for bounding boxes."""

from typing import Dict, Optional, Tuple


def percentage_to_pixels(
    bbox: Dict[str, float], frame_width: int, frame_height: int
) -> Tuple[int, int, int, int]:
    """Convert percentage coordinates to pixel coordinates (x, y, w, h)."""
    x = int(bbox["x"] * frame_width / 100)
    y = int(bbox["y"] * frame_height / 100)
    w = int(bbox["width"] * frame_width / 100)
    h = int(bbox["height"] * frame_height / 100)
    return (x, y, w, h)


def pixels_to_percentage(
    x: int, y: int, w: int, h: int, frame_width: int, frame_height: int
) -> Dict[str, float]:
    """Convert pixel coordinates to percentage coordinates."""
    return {
        "x": (x / frame_width) * 100,
        "y": (y / frame_height) * 100,
        "width": (w / frame_width) * 100,
        "height": (h / frame_height) * 100,
    }


def pct_to_px(
    bx: float,
    by: float,
    bw: float,
    bh: float,
    orig_w: int,
    orig_h: int,
    render_w: Optional[float] = None,
    render_h: Optional[float] = None,
) -> Tuple[int, int, int, int]:
    """
    Convert percentage bbox to pixel coordinates.
    If render_w/render_h are provided, accounts for display scaling (frontend canvas size).
    Returns clamped (x_px, y_px, w_px, h_px).
    """
    if render_w and render_h:
        disp_x = (bx / 100.0) * render_w
        disp_y = (by / 100.0) * render_h
        disp_w = (bw / 100.0) * render_w
        disp_h = (bh / 100.0) * render_h
        scale_x = orig_w / render_w
        scale_y = orig_h / render_h
        x_px = int(disp_x * scale_x)
        y_px = int(disp_y * scale_y)
        w_px = int(disp_w * scale_x)
        h_px = int(disp_h * scale_y)
    else:
        x_px = int((bx / 100.0) * orig_w)
        y_px = int((by / 100.0) * orig_h)
        w_px = int((bw / 100.0) * orig_w)
        h_px = int((bh / 100.0) * orig_h)

    x_px = max(0, min(x_px, orig_w - 1))
    y_px = max(0, min(y_px, orig_h - 1))
    w_px = max(1, min(w_px, orig_w - x_px))
    h_px = max(1, min(h_px, orig_h - y_px))
    return x_px, y_px, w_px, h_px
