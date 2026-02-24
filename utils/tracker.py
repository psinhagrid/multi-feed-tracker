"""Tracker creation utilities."""

import cv2


def create_tracker():
    """Create and return a CSRT tracker instance."""
    return cv2.TrackerCSRT.create()
