"""Tracker creation utilities."""

import cv2


def create_tracker():
    """Create and return a tracker instance.

    Preference order:
      1. CSRT  — best accuracy (OpenCV < 4.13 or contrib)
      2. MIL   — available in all recent OpenCV builds (no extra models needed)
    """
    # CSRT (older OpenCV / contrib builds)
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT"):
        return cv2.TrackerCSRT.create()
    # cv2.legacy (transitional builds)
    legacy = getattr(cv2, "legacy", None)
    if legacy:
        if hasattr(legacy, "TrackerCSRT_create"):
            return legacy.TrackerCSRT_create()
        if hasattr(legacy, "TrackerMIL_create"):
            return legacy.TrackerMIL_create()
    # OpenCV 4.13+ — CSRT removed, use MIL
    return cv2.TrackerMIL_create()
