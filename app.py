"""
FastAPI Backend for VisionAI Studio
Connects React frontend with Python object detection and tracking
"""

import asyncio
import io
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from dotenv import load_dotenv

# Load .env BEFORE importing re_id (needs FAST_REID_PATH for sys.path)
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

import config
from detection import ObjectDetector
from llm.image_describer import describe_image
from schemas import BoundingBox, DetectionRequest, DetectionResponse, TrackingRequest

# Re-ID (optional - fast-reid must be installed)
try:
    from re_id.embedding_extractor import get_reid_extractor
    from re_id.matcher import is_match
    REID_AVAILABLE = True
except ImportError:
    REID_AVAILABLE = False
    get_reid_extractor = None
    is_match = None
from utils import (
    create_tracker,
    get_device,
    get_video_metadata,
    label_roi_from_frame,
    pct_to_px,
    percentage_to_pixels,
    pixels_to_percentage,
    save_roi_debug,
    save_uploaded_video,
)

app = FastAPI(title="VisionAI Studio API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://localhost:8081",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8081",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
device = None
detector = None
active_sessions: Dict[str, Dict[str, Any]] = {}
reid_sessions: Dict[str, Dict[str, Any]] = {}
reid_extractor = None


def get_reid_extractor_cached():
    """Lazy load Re-ID extractor on first use"""
    global reid_extractor
    if reid_extractor is None and REID_AVAILABLE and get_reid_extractor:
        reid_extractor = get_reid_extractor()
    return reid_extractor


def get_detector():
    """Lazy load detector on first use"""
    global device, detector
    if detector is None:
        device = get_device()
        detector = ObjectDetector(device=device)
        print(f"Detector initialized on {device}")
    return detector


@app.get("/api/track-label/{session_id}")
async def get_track_label(session_id: str):
    """Return labels for all trackers in this session"""
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    trackers = session.get("trackers", [])
    result = []
    for t in trackers:
        result.append({
            "id": t.get("id", "T"),
            "label": t.get("label", "tracked object"),
            "status": t.get("status", "active")
        })
    # fallback if no trackers present
    if not result and "llm_label" in session:
        result.append({"id": "T1", "label": session["llm_label"], "status": "active"})
    return {"trackers": result}


@app.get("/api/track-snapshot/{session_id}")
async def track_snapshot(session_id: str):
    """Return the latest cached tracking frame (JPEG)."""
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    frame = session.get("last_frame")
    if frame is None:
        # Attempt to grab first frame as fallback
        video_path = session["video_path"]
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise HTTPException(status_code=400, detail="No frame available")

    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        raise HTTPException(status_code=500, detail="Could not encode snapshot")

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "VisionAI Studio API",
        "device": str(device) if device else "not_initialized",
        "version": "1.0.0"
    }


@app.get("/api/health")
async def health_check():
    """Simple health check for frontend status indicator"""
    return {"status": "online"}


@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify API is accessible"""
    return {
        "status": "ok",
        "message": "Backend API is working",
        "active_sessions": len(active_sessions)
    }


@app.get("/api/video-stream/{session_id}")
async def stream_video_with_boxes(
    session_id: str, 
    query: str, 
    detection_interval: int = 15,
    threshold: float = 0.5
):
    """
    Stream video with bounding boxes drawn on frames
    Returns video stream as multipart JPEG frames
    """
    from fastapi.responses import StreamingResponse
    import io
    
    def generate_frames():
        import time
        
        session = active_sessions.get(session_id)
        if not session:
            return
        
        video_path = session["video_path"]
        
        # Get detector (lazy load)
        det = get_detector()
        
        # Loop the video indefinitely
        while True:
            # Check if session still exists and video file is valid
            if session_id not in active_sessions:
                return
            
            if not Path(video_path).exists():
                return
            
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                cap.release()
                return
            
            # Get video FPS for proper timing
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0  # Default to 30 FPS
            
            frame_idx = 0
            last_boxes = []
            trackers = []  # List of (tracker, label, score) tuples
            
            while cap.isOpened():
                frame_start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection every N frames
                if frame_idx % detection_interval == 0:
                    h, w = frame.shape[:2]
                    new_width = config.RESIZE_WIDTH
                    new_height = int(h * (config.RESIZE_WIDTH / w))
                    resized = cv2.resize(frame, (new_width, new_height))
                    
                    pil_image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                    results, _ = det.detect(pil_image, [query], threshold=threshold)
                    
                    # Store boxes scaled to original size and reinit trackers
                    last_boxes = []
                    trackers = []
                    for box, score in zip(results['boxes'], results['scores']):
                        if score >= threshold:
                            x1, y1, x2, y2 = box.tolist()
                            # Scale back to original
                            scale_x = w / new_width
                            scale_y = h / new_height
                            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                            
                            # Initialize CSRT tracker for this detection
                            tracker = create_tracker()
                            bbox = (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h)
                            tracker.init(frame, bbox)
                            trackers.append((tracker, query, score))
                            last_boxes.append(((x1, y1, x2, y2), score))
                else:
                    # Track existing objects in between detections
                    last_boxes = []
                    updated_trackers = []
                    for tracker, label, score in trackers:
                        success, bbox = tracker.update(frame)
                        if success:
                            x, y, w_box, h_box = [int(v) for v in bbox]
                            last_boxes.append(((x, y, x + w_box, y + h_box), score))
                            updated_trackers.append((tracker, label, score))
                    trackers = updated_trackers
                
                # Draw boxes on frame with more prominent style
                for (x1, y1, x2, y2), score in last_boxes:
                    # Draw thicker green rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Draw label with background for better visibility
                    label = f"{query}: {score:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    
                    # Draw black background for text
                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_height - baseline - 5),
                        (x1 + text_width, y1),
                        (0, 0, 0),
                        -1  # Filled rectangle
                    )
                    
                    # Draw text in bright green
                    cv2.putText(
                        frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
            
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                # Yield frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                frame_idx += 1
                
                # Control frame rate - sleep to maintain proper FPS with extra buffer
                elapsed = time.time() - frame_start_time
                # Add 50% extra delay to slow down playback
                target_delay = frame_delay * 1.5
                sleep_time = max(target_delay - elapsed, 0.001)
                time.sleep(sleep_time)
            
            cap.release()
            # Loop will restart video automatically
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload video file and create a session
    Returns session_id and video metadata
    """
    try:
        # Validate file type
        if not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Save video
        video_path = save_uploaded_video(file)
        try:
            metadata = get_video_metadata(video_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Create session
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            "video_path": video_path,
            "fps": metadata["fps"],
            "frame_count": metadata["frame_count"],
            "width": metadata["width"],
            "height": metadata["height"],
            "duration": metadata["duration"],
            "tracker": None,
            "mode": None,
            "trackers": [],
            "track_labels": {},
            "last_frame": None,
            "original_w": None,
            "original_h": None,
        }

        return {
            "session_id": session_id,
            "metadata": {
                "fps": metadata["fps"],
                "frame_count": metadata["frame_count"],
                "width": metadata["width"],
                "height": metadata["height"],
                "duration": metadata["duration"],
            },
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect", response_model=DetectionResponse)
async def detect_objects(request: DetectionRequest):
    """
    Run object detection on video using Grounding DINO
    Returns bounding boxes for all frames
    """
    try:
        session = active_sessions.get(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        video_path = session["video_path"]
        cap = cv2.VideoCapture(str(video_path))
        
        boxes = []
        frame_idx = 0
        detection_count = 0
        
        # Process every Nth frame for detection
        detection_interval = config.DETECTION_FRAME_INTERVAL
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection on every Nth frame
            if frame_idx % detection_interval == 0:
                # Resize for faster detection
                h, w = frame.shape[:2]
                new_width = config.RESIZE_WIDTH
                new_height = int(h * (config.RESIZE_WIDTH / w))
                resized = cv2.resize(frame, (new_width, new_height))
                
                # Convert to PIL
                pil_image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                
                # Get detector (lazy load)
                det = get_detector()
                
                # Detect
                results, _ = det.detect(pil_image, [request.query], threshold=request.threshold)
                
                # Convert to response format
                for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
                    if score >= request.threshold:
                        x1, y1, x2, y2 = box.tolist()
                        
                        # Scale back to original size
                        scale_x = w / new_width
                        scale_y = h / new_height
                        x1, x2 = x1 * scale_x, x2 * scale_x
                        y1, y2 = y1 * scale_y, y2 * scale_y
                        
                        # Convert to percentage
                        bbox_pct = pixels_to_percentage(
                            int(x1), int(y1), 
                            int(x2 - x1), int(y2 - y1),
                            w, h
                        )
                        
                        boxes.append(BoundingBox(
                            id=f"D{detection_count}",
                            x=bbox_pct["x"],
                            y=bbox_pct["y"],
                            width=bbox_pct["width"],
                            height=bbox_pct["height"],
                            label=request.query,
                            confidence=float(score),
                            type="detection",
                            frame_number=frame_idx  # Add frame number
                        ))
                        detection_count += 1
            
            frame_idx += 1
        
        cap.release()
        session["mode"] = "identify"
        
        return DetectionResponse(
            boxes=boxes,
            fps=session["fps"],
            frame_count=session["frame_count"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/track-stream/{session_id}")
async def stream_video_with_tracking(
    session_id: str,
    bbox_x: Optional[float] = None,
    bbox_y: Optional[float] = None, 
    bbox_width: Optional[float] = None,
    bbox_height: Optional[float] = None,
    render_w: Optional[float] = None,
    render_h: Optional[float] = None,
    frame_time: Optional[float] = None
):
    """
    Stream video with CSRT tracking - draws boxes on backend and streams
    Gets label from LLM first, then tracks with that label
    """
    def generate_tracking_frames():
        import time
        session = active_sessions.get(session_id)
        if not session:
            return

        video_path = session["video_path"]
        fps_session = session.get("fps") or 30.0

        # Only initialize first tracker if trackers list is empty
        if not session.get("trackers"):
            # First tracker requires bbox parameters
            if bbox_x is None or bbox_y is None or bbox_width is None or bbox_height is None:
                print("Error: bbox parameters required for first tracker initialization")
                return
                
            session["trackers"] = []
            session["track_labels"] = {}

            # Get first frame (with optional seek) to initialize
            cap_temp = cv2.VideoCapture(str(video_path))
            if frame_time is not None and frame_time >= 0:
                target_frame = int(frame_time * fps_session)
                cap_temp.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, first_frame = cap_temp.read()
            cap_temp.release()
            if not ret:
                return

            orig_h, orig_w = first_frame.shape[:2]
            session["original_w"] = orig_w
            session["original_h"] = orig_h
            session["last_frame"] = first_frame.copy()

            # Create first tracker
            x_px, y_px, w_px, h_px = pct_to_px(
                bbox_x, bbox_y, bbox_width, bbox_height, orig_w, orig_h, render_w, render_h
            )
            save_roi_debug(first_frame, (x_px, y_px, w_px, h_px), session_id, "T1")
            
            llm_label = label_roi_from_frame(first_frame, (x_px, y_px, w_px, h_px))
            tracker = create_tracker()
            tracker.init(first_frame, (x_px, y_px, w_px, h_px))
            session["trackers"].append({
                "id": "T1",
                "tracker": tracker,
                "label": llm_label,
                "status": "active",
                "fail_count": 0,
            })
            session["track_labels"]["T1"] = llm_label

        # Loop the video indefinitely
        while True:
            # Check if session still exists and video file is valid
            if session_id not in active_sessions:
                return
            
            if not Path(video_path).exists():
                return
            
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                cap.release()
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps > 0 else fps_session
            frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0

            # Resume from stored position if available, otherwise use frame_time parameter
            resume_frame_time = session.get("current_frame_time")
            if resume_frame_time is not None and resume_frame_time >= 0:
                target_frame = int(resume_frame_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                print(f"Resuming from frame {target_frame} (time: {resume_frame_time:.2f}s)")
            elif frame_time is not None and frame_time >= 0:
                target_frame = int(frame_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            while cap.isOpened():
                frame_start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                # Store current frame number for resume
                current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                session["current_frame"] = current_frame_num
                session["current_frame_time"] = current_frame_num / fps if fps > 0 else 0
                
                # Update all trackers
                for t in list(session["trackers"]):
                    success, bbox = t["tracker"].update(frame)
                    if success:
                        x, y, w_box, h_box = [int(v) for v in bbox]
                        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 3)
                        (text_width, text_height), baseline = cv2.getTextSize(t["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x, y - text_height - baseline - 5), (x + text_width, y), (0, 0, 0), -1)
                        cv2.putText(frame, t["label"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        t["status"] = "active"
                        t["fail_count"] = 0
                        session["track_labels"][t["id"]] = t["label"]
                    else:
                        t["fail_count"] += 1
                        t["status"] = "lost"
                        cv2.putText(frame, f"{t['id']} LOST", (50, 50 + 20 * int(t['id'][1:] or 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if t["fail_count"] > 30:
                            session["trackers"].remove(t)

                # Save frame with boxes drawn for snapshot
                session["last_frame"] = frame.copy()

                # Encode and yield frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                # Control frame rate
                elapsed = time.time() - frame_start_time
                target_delay = frame_delay * 1.5
                sleep_time = max(target_delay - elapsed, 0.001)
                time.sleep(sleep_time)

            cap.release()
            # Reset resume position so next loop plays from the beginning
            session["current_frame_time"] = None
    
    return StreamingResponse(
        generate_tracking_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/track-debug/{session_id}")
async def debug_track_frame(
    session_id: str,
    bbox_x: float,
    bbox_y: float,
    bbox_width: float,
    bbox_height: float,
    render_w: Optional[float] = None,
    render_h: Optional[float] = None,
    frame_time: Optional[float] = None
):
    """
    Returns the FIRST frame with the provided bbox drawn (no streaming).
    Useful to visually verify that frontend-sent percentages map to backend pixels.
    """
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    video_path = session["video_path"]
    fps_session = session.get("fps") or 30.0
    cap = cv2.VideoCapture(str(video_path))
    if frame_time is not None and frame_time >= 0:
        target_frame = int(frame_time * fps_session)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=400, detail="Could not read first frame")

    original_h, original_w = frame.shape[:2]
    x_px, y_px, w_px, h_px = pct_to_px(
        bbox_x, bbox_y, bbox_width, bbox_height,
        original_w, original_h, render_w, render_h
    )

    # Draw box for debugging
    cv2.rectangle(frame, (x_px, y_px), (x_px + w_px, y_px + h_px), (255, 0, 255), 3)
    label = f"debug: {bbox_x:.1f}%,{bbox_y:.1f}% {bbox_width:.1f}x{bbox_height:.1f}%"
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x_px, max(0, y_px - text_height - baseline - 5)), (x_px + text_width, y_px), (0, 0, 0), -1)
    cv2.putText(frame, label, (x_px, max(5 + text_height, y_px - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        raise HTTPException(status_code=500, detail="Could not encode debug frame")

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )


@app.post("/api/track-add/{session_id}")
async def add_tracker(
    session_id: str,
    bbox_x: float,
    bbox_y: float,
    bbox_width: float,
    bbox_height: float,
    render_w: Optional[float] = None,
    render_h: Optional[float] = None,
    frame_time: Optional[float] = None
):
    """
    Add an additional CSRT tracker during an active tracking session.
    Uses the latest cached frame (or seeks) to init tracker and run LLM label.
    """
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    video_path = session["video_path"]
    fps_session = session.get("fps") or 30.0
    orig_w = session.get("original_w")
    orig_h = session.get("original_h")

    # Get a frame to initialize tracker
    frame = session.get("last_frame")
    if frame is None:
        cap = cv2.VideoCapture(str(video_path))
        if frame_time is not None and frame_time >= 0:
            target_frame = int(frame_time * fps_session)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise HTTPException(status_code=400, detail="Could not read frame to add tracker")

    if orig_w is None or orig_h is None:
        orig_h, orig_w = frame.shape[:2]
        session["original_w"] = orig_w
        session["original_h"] = orig_h

    x_px, y_px, w_px, h_px = pct_to_px(
        bbox_x, bbox_y, bbox_width, bbox_height, orig_w, orig_h, render_w, render_h
    )
    tracker_id = f"T{len(session.get('trackers', [])) + 1}"
    save_roi_debug(frame, (x_px, y_px, w_px, h_px), session_id, tracker_id)
    llm_label = label_roi_from_frame(frame, (x_px, y_px, w_px, h_px))

    tracker = create_tracker()
    tracker.init(frame, (x_px, y_px, w_px, h_px))

    session.setdefault("trackers", []).append({
        "id": tracker_id,
        "tracker": tracker,
        "label": llm_label,
        "status": "active",
        "fail_count": 0,
    })
    session.setdefault("track_labels", {})[tracker_id] = llm_label

    return {
        "id": tracker_id,
        "label": llm_label,
        "status": "active"
    }


@app.post("/api/track")
async def start_tracking(request: TrackingRequest):
    """
    Initialize CSRT tracker with user-drawn bounding box
    Returns tracking session info (deprecated - use track-stream instead)
    """
    try:
        session = active_sessions.get(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        video_path = session["video_path"]
        cap = cv2.VideoCapture(str(video_path))
        
        # Get first frame
        ret, frame = cap.read()
        if not ret:
            raise HTTPException(status_code=500, detail="Could not read video frame")
        
        h, w = frame.shape[:2]
        
        # Convert percentage bbox to pixels
        bbox_pixels = percentage_to_pixels(request.bbox, w, h)
        
        # Initialize CSRT tracker
        tracker = create_tracker()
        tracker.init(frame, bbox_pixels)
        
        cap.release()
        
        # Store tracker in session
        session["tracker"] = tracker
        session["mode"] = "track"
        session["initial_bbox"] = bbox_pixels
        
        return {
            "session_id": request.session_id,
            "status": "tracking_initialized",
            "bbox": request.bbox
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/track/{session_id}/stream")
async def stream_tracking(session_id: str):
    """
    Stream tracking results frame-by-frame
    """
    session = active_sessions.get(session_id)
    if not session or not session.get("tracker"):
        raise HTTPException(status_code=404, detail="Tracking session not found")
    
    async def generate_tracking_updates():
        video_path = session["video_path"]
        tracker = session["tracker"]
        
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w = frame.shape[:2]
            
            # Update tracker
            success, bbox_pixels = tracker.update(frame)
            
            if success:
                x, y, w_box, h_box = [int(v) for v in bbox_pixels]
                bbox_pct = pixels_to_percentage(x, y, w_box, h_box, w, h)
                
                result = {
                    "frame": frame_idx,
                    "bbox": bbox_pct,
                    "status": "active",
                    "confidence": 0.95  # CSRT doesn't provide confidence
                }
            else:
                result = {
                    "frame": frame_idx,
                    "status": "lost"
                }
            
            yield f"data: {json.dumps(result)}\n\n"
            frame_idx += 1
            await asyncio.sleep(1 / session["fps"])  # Real-time playback
        
        cap.release()
    
    return StreamingResponse(
        generate_tracking_updates(),
        media_type="text/event-stream"
    )


@app.post("/api/label-roi")
async def label_roi_with_llm(session_id: str, file: UploadFile = File(...)):
    """
    Send cropped ROI to LLM for labeling
    Returns descriptive label
    """
    try:
        session = active_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Save crop temporarily
        temp_dir = Path(tempfile.gettempdir()) / "visionai" / "crops"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        crop_path = temp_dir / f"{uuid.uuid4()}.jpg"
        with open(crop_path, "wb") as f:
            f.write(file.file.read())
        
        # Get LLM label
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
        
        label = describe_image(str(crop_path), api_key)
        
        # Cleanup
        crop_path.unlink()
        
        return {
            "label": label,
            "session_id": session_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Re-ID Mode Endpoints ==============


@app.post("/api/reid-upload-video1")
async def reid_upload_video1(file: UploadFile = File(...)):
    """
    Re-ID Step 1: Upload first video (source - person to find).
    Returns reid_session_id and video metadata.
    """
    try:
        if not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="File must be a video")

        video_path = save_uploaded_video(file)
        metadata = get_video_metadata(video_path)

        session_id = str(uuid.uuid4())
        reid_sessions[session_id] = {
            "video1_path": video_path,
            "video2_path": None,
            "reference_embedding": None,
            "reference_frame": None,
            "step": "set_reference",  # set_reference | upload_v2 | streaming
            "fps": metadata["fps"],
            "frame_count": metadata["frame_count"],
            "width": metadata["width"],
            "height": metadata["height"],
            "duration": metadata["duration"],
            "original_w": None,
            "original_h": None,
            "current_frame_time": None,
        }
        return {
            "session_id": session_id,
            "metadata": {
                "fps": metadata["fps"],
                "frame_count": metadata["frame_count"],
                "width": metadata["width"],
                "height": metadata["height"],
                "duration": metadata["duration"],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reid-set-reference")
async def reid_set_reference(
    session_id: str,
    bbox_x: float,
    bbox_y: float,
    bbox_width: float,
    bbox_height: float,
    render_w: Optional[float] = None,
    render_h: Optional[float] = None,
    frame_time: Optional[float] = None,
):
    """
    Re-ID Step 2: Set reference person from video 1 using drawn box.
    Extracts embedding and stores for matching in video 2.
    """
    session = reid_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Re-ID session not found")

    ext = get_reid_extractor_cached()
    if not ext:
        raise HTTPException(
            status_code=503,
            detail="Re-ID not available. Install fast-reid and set REID_CONFIG_PATH, REID_WEIGHTS_PATH.",
        )

    video_path = session["video1_path"]
    fps_session = session.get("fps") or 30.0
    cap = cv2.VideoCapture(str(video_path))
    if frame_time is not None and frame_time >= 0:
        target_frame = int(frame_time * fps_session)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=400, detail="Could not read frame from video 1")

    orig_h, orig_w = frame.shape[:2]
    session["original_w"] = orig_w
    session["original_h"] = orig_h

    x_px, y_px, w_px, h_px = pct_to_px(
        bbox_x, bbox_y, bbox_width, bbox_height, orig_w, orig_h, render_w, render_h
    )
    bbox = (x_px, y_px, w_px, h_px)

    try:
        embedding = ext.extract_roi(frame, bbox)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract embedding: {e}")

    # Ask LLM to label the reference crop (same as Track mode)
    try:
        ref_label = label_roi_from_frame(frame, bbox)
    except Exception:
        ref_label = "person"

    # Store as list for JSON compatibility; we use numpy when matching
    session["reference_embedding"] = embedding.tolist()
    session["reference_frame"] = frame  # keep for display
    session["reference_label"] = ref_label
    session["step"] = "upload_v2"

    print(f"[Re-ID] Reference set. LLM label: '{ref_label}'", flush=True)
    return {"status": "reference_set", "embedding_shape": list(embedding.shape), "label": ref_label}


@app.post("/api/reid-upload-video2")
async def reid_upload_video2(session_id: str, file: UploadFile = File(...)):
    """
    Re-ID Step 3: Upload second video (search - find the person here).
    """
    session = reid_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Re-ID session not found")
    if session.get("reference_embedding") is None:
        raise HTTPException(status_code=400, detail="Set reference first (draw box on video 1)")

    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    video_path = save_uploaded_video(file)
    metadata = get_video_metadata(video_path)

    session["video2_path"] = video_path
    session["step"] = "streaming"
    session["v2_fps"] = metadata["fps"]
    session["v2_frame_count"] = metadata["frame_count"]
    session["v2_width"] = metadata["width"]
    session["v2_height"] = metadata["height"]
    session["v2_duration"] = metadata["duration"]

    return {
        "status": "video2_ready",
        "metadata": {
            "fps": metadata["fps"],
            "frame_count": metadata["frame_count"],
            "width": metadata["width"],
            "height": metadata["height"],
            "duration": metadata["duration"],
        },
    }


@app.get("/api/reid-stream/{session_id}")
async def reid_stream(
    session_id: str,
    detection_interval: int = 15,
    threshold: float = 0.5,
):
    """
    Stream video 2 with re-ID matching: green box = match, gray = no match.
    """
    session = reid_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Re-ID session not found")

    ref_emb = session.get("reference_embedding")
    video2_path = session.get("video2_path")
    if not ref_emb or not video2_path or not Path(video2_path).exists():
        raise HTTPException(status_code=400, detail="Complete steps 1–3: set reference and upload video 2")

    ext = get_reid_extractor_cached()
    det = get_detector()
    if not ext:
        raise HTTPException(status_code=503, detail="Re-ID model not available")

    import numpy as np
    ref_emb_np = np.array(ref_emb, dtype=np.float32)
    match_thresh = getattr(config, "REID_MATCH_THRESHOLD", 0.5)
    ref_label = session.get("reference_label", "person")

    def generate_frames():
        import time
        import threading

        fps = 30.0
        frame_delay = 1.0 / fps
        frame_idx = 0
        # trackers: list of (tracker, matched, sim, last_box)
        trackers = []
        last_boxes = []

        # ── Background detection + Re-ID thread ──────────────────────────────
        # Detection + Re-ID is expensive (1-5s on CPU). Run it in a background
        # thread so the main loop always runs at full FPS via CSRT updates.
        det_lock = threading.Lock()
        det_state = {
            "running": False,
            "new_results": None,    # list of (box_xyxy, matched, sim)
            "detection_frame": None, # the EXACT frame detection ran on (use for tracker.init)
        }

        def run_detection_reid(frame_copy):
            """Run detect + Re-ID on a copy of the frame; store result in det_state."""
            new_results = []
            t0 = time.time()
            try:
                h, w = frame_copy.shape[:2]
                nw = config.RESIZE_WIDTH
                nh = int(h * (nw / w))
                resized = cv2.resize(frame_copy, (nw, nh))
                pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

                t_det = time.time()
                results, _ = det.detect(pil_img, ["person"], threshold=threshold)
                detections = [(b, s) for b, s in zip(results["boxes"], results["scores"]) if s >= threshold]
                detections = sorted(detections, key=lambda x: -x[1])[:10]
                print(f"[Re-ID] DINO detect: {len(detections)} people in {time.time()-t_det:.2f}s", flush=True)

                scale_x, scale_y = w / nw, h / nh
                best_box = None
                best_sim = -1.0
                all_sims = []
                for i, (box, score) in enumerate(detections):
                    x1, y1, x2, y2 = box.tolist()
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    roi = frame_copy[y1:y2, x1:x2]
                    if roi.size == 0:
                        print(f"  person {i+1}: empty ROI, skipping")
                        continue
                    t_reid = time.time()
                    try:
                        query_emb = ext.extract(roi)
                        matched, sim = is_match(ref_emb_np, query_emb, match_thresh)
                    except Exception as re:
                        matched, sim = False, 0.0
                        print(f"  person {i+1}: Re-ID extract FAILED: {re}")
                    all_sims.append(sim)
                    print(f"  person {i+1}: sim={sim:.3f} matched={matched} reid_time={time.time()-t_reid:.2f}s box=({x1},{y1},{x2},{y2})")
                    if sim > best_sim:
                        best_sim = sim
                        best_box = ((x1, y1, x2, y2), matched, sim)

                if best_box is not None and best_sim >= 0.2:
                    new_results = [best_box]
                    print(f"[Re-ID] Best match: sim={best_sim:.3f} matched={best_box[1]} (threshold={match_thresh})", flush=True)
                else:
                    print(f"[Re-ID] No match found (best_sim={best_sim:.3f} < 0.2 or no detections). All sims: {[f'{s:.2f}' for s in all_sims]}", flush=True)

                print(f"[Re-ID] Total detection+reid time: {time.time()-t0:.2f}s", flush=True)
            except Exception as e:
                import traceback
                print(f"[Re-ID] Detection thread EXCEPTION: {e}", flush=True)
                traceback.print_exc()
            finally:
                with det_lock:
                    det_state["new_results"] = new_results
                    det_state["detection_frame"] = frame_copy  # keep so tracker.init uses the right frame
                    det_state["running"] = False

        # ─────────────────────────────────────────────────────────────────────

        print(f"[Re-ID] Stream started, interval={detection_interval}", flush=True)

        # ── Warm-up: run first detection synchronously so box shows from frame 1 ──
        # This causes a 1-2s pause before streaming starts, but avoids the
        # "no box for first N frames" gap that the async approach produces.
        print(f"[Re-ID] Running initial detection (warm-up)...", flush=True)
        cap_init = cv2.VideoCapture(str(video2_path))
        ret_init, first_frame = cap_init.read()
        cap_init.release()
        if ret_init:
            run_detection_reid(first_frame)   # blocking call (not in thread)
            new_results = det_state.get("new_results")
            init_frame_warm = det_state.get("detection_frame")
            if new_results:
                det_state["new_results"] = None
                det_state["detection_frame"] = None
                for (x1, y1, x2, y2), matched, sim in new_results:
                    tr = create_tracker()
                    tr.init(init_frame_warm, (x1, y1, x2 - x1, y2 - y1))
                    trackers.append((tr, matched, sim, (x1, y1, x2, y2), 0))
                    last_boxes.append(((x1, y1, x2, y2), matched, sim))
                print(f"[Re-ID] Warm-up done: {len(trackers)} tracker(s) ready", flush=True)
        # ─────────────────────────────────────────────────────────────────────────

        while session_id in reid_sessions and Path(video2_path).exists():
            cap = cv2.VideoCapture(str(video2_path))
            if not cap.isOpened():
                print("[Re-ID] ERROR: Could not open video")
                break
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_delay = 1.0 / fps
            if frame_idx == 0:
                print(f"[Re-ID] Video: {total_frames} frames, {fps} fps", flush=True)

            while cap.isOpened():
                frame_start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                # ── Schedule background detection (non-blocking) ──────────────
                if frame_idx % detection_interval == 0:
                    with det_lock:
                        already_running = det_state["running"]
                    if not already_running:
                        with det_lock:
                            det_state["running"] = True
                        print(f"[Re-ID] Scheduling detection thread at frame {frame_idx}", flush=True)
                        t = threading.Thread(
                            target=run_detection_reid,
                            args=(frame.copy(),),
                            daemon=True,
                        )
                        t.start()
                    else:
                        print(f"[Re-ID] Frame {frame_idx}: skipping detection, previous still running", flush=True)

                # ── Apply new detection results if ready ──────────────────────
                with det_lock:
                    new_results = det_state["new_results"]
                    init_frame = det_state["detection_frame"]
                    if new_results is not None:
                        det_state["new_results"] = None
                        det_state["detection_frame"] = None

                if new_results is not None:
                    # Reinit trackers — MUST use the frame detection actually ran on,
                    # not the current frame (person may have moved since detection ran).
                    print(f"[Re-ID] Frame {frame_idx}: applying new detection ({len(new_results)} results), reinit trackers on detection frame", flush=True)
                    trackers = []
                    last_boxes = []
                    for (x1, y1, x2, y2), matched, sim in new_results:
                        tr = create_tracker()
                        tr.init(init_frame, (x1, y1, x2 - x1, y2 - y1))
                        # (tracker, matched, sim, last_box, fail_count)
                        trackers.append((tr, matched, sim, (x1, y1, x2, y2), 0))
                        last_boxes.append(((x1, y1, x2, y2), matched, sim))
                else:
                    # CSRT update every frame — instant, no stalling
                    last_boxes = []
                    updated = []
                    for tr, matched, sim, last_box, fail_count in trackers:
                        t_track = time.time()
                        ok, bbox = tr.update(frame)
                        track_ms = (time.time() - t_track) * 1000
                        if ok:
                            x, y, bw, bh = [int(v) for v in bbox]
                            new_box = (x, y, x + bw, y + bh)
                            last_boxes.append((new_box, matched, sim))
                            updated.append((tr, matched, sim, new_box, 0))
                            if frame_idx % 30 == 0:
                                print(f"[Re-ID] Frame {frame_idx}: tracker OK box={new_box} ({track_ms:.1f}ms)", flush=True)
                        else:
                            fail_count += 1
                            print(f"[Re-ID] Frame {frame_idx}: tracker FAILED fail_count={fail_count} ({track_ms:.1f}ms)", flush=True)
                            if fail_count <= 45:
                                # Keep last known box for ~1.5s before giving up
                                last_boxes.append((last_box, matched, sim))
                                updated.append((tr, matched, sim, last_box, fail_count))
                            else:
                                print(f"[Re-ID] Frame {frame_idx}: tracker DROPPED after {fail_count} failures", flush=True)
                    trackers = updated

                # Frame timing
                if frame_idx % 30 == 0:
                    print(f"[Re-ID] Frame {frame_idx}: {len(trackers)} trackers, {len(last_boxes)} boxes drawn", flush=True)

                # Draw boxes — use LLM label like Track mode, with similarity %
                for (x1, y1, x2, y2), matched, sim in last_boxes:
                    color = (0, 255, 0) if matched else (0, 165, 255)
                    box_label = f"{ref_label} {int(sim * 100)}%"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    (tw, th), _ = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 0, 0), -1)
                    cv2.putText(frame, box_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                ret2, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

                frame_idx += 1
                elapsed = time.time() - frame_start_time
                time.sleep(max(0.001, frame_delay * 1.5 - elapsed))

            cap.release()
            frame_idx = 0
            # Keep trackers and last_boxes across loop so box stays visible at restart

        print(f"[Re-ID] Stream ended", flush=True)

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/reid-label/{session_id}")
async def reid_label(session_id: str):
    """Return the LLM label for the reference person."""
    session = reid_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Re-ID session not found")
    return {"label": session.get("reference_label", "person")}


@app.get("/api/reid-snapshot/{session_id}")
async def reid_snapshot(session_id: str, source: str = "video1"):
    """Return a snapshot frame for display (video1 or video2). source: 'video1' | 'video2'"""
    session = reid_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Re-ID session not found")

    if source == "video1":
        frame = session.get("reference_frame")
        if frame is None:
            cap = cv2.VideoCapture(str(session["video1_path"]))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise HTTPException(status_code=400, detail="No frame available")
    else:
        path = session.get("video2_path")
        if not path:
            raise HTTPException(status_code=400, detail="Video 2 not uploaded yet")
        cap = cv2.VideoCapture(str(path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise HTTPException(status_code=400, detail="No frame available")

    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        raise HTTPException(status_code=500, detail="Could not encode snapshot")
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


@app.delete("/api/reid-session/{session_id}")
async def delete_reid_session(session_id: str):
    """Clean up Re-ID session and temp videos."""
    session = reid_sessions.get(session_id)
    if session:
        for key in ("video1_path", "video2_path"):
            p = session.get(key)
            if p and Path(p).exists():
                Path(p).unlink()
        del reid_sessions[session_id]
        return {"status": "session_deleted", "session_id": session_id}
    return {"status": "session_already_deleted", "session_id": session_id}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """
    Clean up session and delete temporary video
    """
    session = active_sessions.get(session_id)
    if session:
        # Delete video file
        video_path = session["video_path"]
        if video_path.exists():
            video_path.unlink()
        
        # Remove from active sessions
        del active_sessions[session_id]
        
        return {"status": "session_deleted", "session_id": session_id}
    
    # Return success even if session doesn't exist (idempotent delete)
    return {"status": "session_already_deleted", "session_id": session_id}


@app.websocket("/ws/track/{session_id}")
async def websocket_tracking(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time tracking updates
    """
    await websocket.accept()
    
    session = active_sessions.get(session_id)
    if not session:
        await websocket.close(code=1008, reason="Session not found")
        return
    
    try:
        video_path = session["video_path"]
        tracker = session.get("tracker")
        
        if not tracker:
            await websocket.close(code=1008, reason="Tracker not initialized")
            return
        
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w = frame.shape[:2]
            success, bbox_pixels = tracker.update(frame)
            
            if success:
                x, y, w_box, h_box = [int(v) for v in bbox_pixels]
                bbox_pct = pixels_to_percentage(x, y, w_box, h_box, w, h)
                
                await websocket.send_json({
                    "frame": frame_idx,
                    "bbox": bbox_pct,
                    "status": "active"
                })
            else:
                await websocket.send_json({
                    "frame": frame_idx,
                    "status": "lost"
                })
            
            frame_idx += 1
            await asyncio.sleep(1 / session["fps"])
        
        cap.release()
        await websocket.close()
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("VisionAI Studio API")
    print("=" * 70)
    print("Backend:  http://localhost:8080")
    print("API Docs: http://localhost:8080/docs")
    print("Health:   http://localhost:8080/api/health")
    print("Frontend: http://localhost:8000")
    print("=" * 70)
    print("Detector will be initialized on first request (lazy loading)")
    print("=" * 70)
    print("")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
