"""Video and file upload utilities."""

import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any

import cv2
from fastapi import UploadFile


def save_uploaded_video(file: UploadFile) -> Path:
    """Save uploaded video to temp directory. Returns path to saved file."""
    temp_dir = Path(tempfile.gettempdir()) / "visionai"
    temp_dir.mkdir(exist_ok=True)
    video_id = str(uuid.uuid4())
    video_path = temp_dir / f"{video_id}.mp4"
    with open(video_path, "wb") as f:
        f.write(file.file.read())
    return video_path


def get_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Extract video metadata (fps, frame_count, width, height, duration)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }
