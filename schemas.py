"""Pydantic models for API request/response."""

from typing import List, Optional

from pydantic import BaseModel


class BoundingBox(BaseModel):
    id: str
    x: float  # percentage
    y: float  # percentage
    width: float  # percentage
    height: float  # percentage
    label: str
    confidence: float
    type: str  # "detection" or "tracking"
    status: Optional[str] = "active"  # "active" or "lost"
    frame_number: Optional[int] = None


class DetectionRequest(BaseModel):
    session_id: str
    query: str
    threshold: float = 0.5


class TrackingRequest(BaseModel):
    session_id: str
    bbox: dict  # {x, y, width, height} in percentages


class DetectionResponse(BaseModel):
    boxes: List[BoundingBox]
    fps: float
    frame_count: int
