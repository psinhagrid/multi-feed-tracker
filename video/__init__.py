"""Video processing and multi-object tracking module."""

from .video_processor import VideoProcessor
from .tracker import PersonTracker

__all__ = ['VideoProcessor', 'PersonTracker']
