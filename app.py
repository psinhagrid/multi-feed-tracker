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
yolo_seg_model = None   # yolov8n-seg.pt — Re-ID detection + segmentation


def get_yolo_seg():
    """Lazy-load YOLOv8 segmentation model (auto-downloads on first use)."""
    global yolo_seg_model
    if yolo_seg_model is None:
        from ultralytics import YOLO
        model_name = getattr(config, "YOLO_SEG_MODEL", "yolov8n-seg.pt")
        yolo_seg_model = YOLO(model_name)
        print(f"[Re-ID] YOLO seg model loaded: {model_name}", flush=True)
    return yolo_seg_model


def clothing_hue_histogram(crop: "np.ndarray") -> "np.ndarray":
    """
    Extract a normalised hue histogram from the torso region of a masked crop.
    Only counts foreground pixels (V>25, S>40) so black YOLO-mask background
    and achromatic clothing are handled gracefully.
    Returns a 18-bin float32 array, or None if too few colourful pixels.
    """
    import numpy as np
    import cv2 as _cv2
    h = crop.shape[0]
    torso = crop[h // 4: 3 * h // 4, :]
    if torso.size == 0:
        torso = crop
    hsv = _cv2.cvtColor(torso, _cv2.COLOR_BGR2HSV)
    fg = (hsv[:, :, 2] > 25) & (hsv[:, :, 1] > 40)
    hues = hsv[:, :, 0][fg]
    if len(hues) < 30:
        return None
    hist, _ = np.histogram(hues, bins=18, range=(0, 180))
    hist = hist.astype(np.float32)
    s = hist.sum()
    return hist / s if s > 0 else None


def hue_similarity(hist_a: "np.ndarray", hist_b: "np.ndarray") -> float:
    """Bhattacharyya coefficient between two hue histograms (0=different, 1=identical)."""
    import numpy as np
    if hist_a is None or hist_b is None:
        return 1.0   # unknown → don't penalise
    return float(np.clip(np.sum(np.sqrt(hist_a * hist_b)), 0.0, 1.0))


def apply_yolo_seg_mask(yolo, image: "np.ndarray", conf: float = 0.3) -> "np.ndarray":
    """
    Run yolov8n-seg on `image` (BGR), find the highest-confidence person,
    apply its segmentation mask, and return the masked image.
    Falls back to the raw image if no person is detected.
    """
    import numpy as np
    results = yolo(image, classes=[0], conf=conf, verbose=False)
    result = results[0]
    if result.masks is not None and len(result.masks) > 0:
        fh, fw = image.shape[:2]
        best_idx = int(result.boxes.conf.argmax())
        mask_np = result.masks.data[best_idx].cpu().numpy()
        mask_resized = cv2.resize(mask_np, (fw, fh), interpolation=cv2.INTER_NEAREST)
        fg = (mask_resized > 0.5).astype(np.uint8)
        return image * fg[:, :, np.newaxis]
    return image


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

        # Clear any stale resume position from a previous (possibly interrupted) stream.
        # This prevents "Resuming from frame N" on every new stream request.
        session["current_frame_time"] = None

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

            # On first loop, seek to the frame where the user drew the box.
            # Subsequent loops always start from the beginning (current_frame_time is None).
            resume_frame_time = session.get("current_frame_time")
            if resume_frame_time is not None and resume_frame_time >= 0:
                target_frame = int(resume_frame_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
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
        # Crop the drawn box, apply YOLO-seg mask to remove background, then embed
        x_c, y_c, w_c, h_c = [int(v) for v in bbox]
        ref_crop = frame[y_c:y_c + h_c, x_c:x_c + w_c]
        if ref_crop.size == 0:
            raise ValueError("Empty crop from drawn box")
        yolo = get_yolo_seg()
        ref_crop_masked = apply_yolo_seg_mask(yolo, ref_crop, conf=0.2)
        embedding = ext.extract(ref_crop_masked)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract embedding: {e}")

    # Ask LLM to label the reference crop (same as Track mode)
    try:
        ref_label = label_roi_from_frame(frame, bbox)
    except Exception:
        ref_label = "person"

    # Store as list for JSON compatibility; we use numpy when matching
    session["reference_embedding"] = embedding.tolist()
    session["reference_crop"] = ref_crop_masked
    # Pre-compute hue histogram for clothing-colour veto during tracking
    ref_hue_hist = clothing_hue_histogram(ref_crop_masked)
    session["reference_hue_hist"] = ref_hue_hist.tolist() if ref_hue_hist is not None else None
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
    detection_interval: int = 5,
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
    if not ext:
        raise HTTPException(status_code=503, detail="Re-ID model not available")

    import numpy as np
    match_thresh = getattr(config, "REID_MATCH_THRESHOLD", 0.5)
    ref_label = session.get("reference_label", "person")

    _ref_hue_raw = session.get("reference_hue_hist")
    ref_hue_hist = np.array(_ref_hue_raw, dtype=np.float32) if _ref_hue_raw else None

    # Wrap in a dict so inner closures can mutate it without triggering
    # Python's "local variable referenced before assignment" error.
    ref_state = {
        "emb":         np.array(ref_emb, dtype=np.float32),
        "adapted":     False,
        "adapt_count": 0,       # consecutive qualifying confirmations before adapting
        "hue_hist":    ref_hue_hist,
    }

    def generate_frames():
        import time
        import threading

        fps = 30.0
        frame_delay = 1.0 / fps
        frame_idx = 0
        # trackers: list of (tracker, matched, sim, last_box)
        trackers = []
        last_boxes = []
        # ref_state["adapted"] tracks whether we've updated to a video-2 embedding.

        # ── Background detection + Re-ID thread ──────────────────────────────
        # Detection + Re-ID is expensive (1-5s on CPU). Run it in a background
        # thread so the main loop always runs at full FPS via CSRT updates.
        det_lock = threading.Lock()
        det_state = {
            "running": False,
            "new_results": None,     # list of (box_xyxy, matched, sim)
            "detection_frame": None, # the EXACT frame detection ran on (use for tracker.init)
            "all_candidates": [],    # every detected person's sim — drawn on-screen for debugging
            "chosen_emb": None,      # embedding of the chosen person (for reference adaptation)
        }
        # All candidates from last detection — drawn as dim boxes so user can
        # see every person's similarity score on-screen
        last_candidates = []

        def _iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix = max(0, min(ax2, bx2) - max(ax1, bx1))
            iy = max(0, min(ay2, by2) - max(ay1, by1))
            inter = ix * iy
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            return inter / max(1, union)

        def run_detection_reid(frame_copy):
            """Run YOLO-seg detect + Re-ID on a frame copy; store best match in det_state."""
            new_results = []
            t0 = time.time()
            try:

                # ── YOLO-seg: detect all persons + get per-person masks in one pass ──
                fh, fw = frame_copy.shape[:2]
                yolo = get_yolo_seg()
                t_det = time.time()
                yolo_results = yolo(frame_copy, classes=[0], conf=threshold,
                                    verbose=False, imgsz=640)
                yolo_result = yolo_results[0]
                n_det = len(yolo_result.boxes) if yolo_result.boxes is not None else 0
                print(f"[Re-ID] YOLO detect: {n_det} people in {time.time()-t_det:.2f}s", flush=True)

                candidates = []  # list of (box_xyxy, matched, sim)
                for i, box in enumerate(yolo_result.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    # Clamp to frame bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(fw, x2), min(fh, y2)
                    box_w, box_h = x2 - x1, y2 - y1

                    # Skip merged detections (too wide relative to height)
                    if box_h > 0 and (box_w / box_h) > 0.65:
                        print(f"  person {i+1}: skipping wide box (ratio={box_w/box_h:.2f})")
                        continue

                    # Build segmentation-masked crop using YOLO-seg mask
                    crop = frame_copy[y1:y2, x1:x2].copy()
                    if crop.size == 0:
                        continue
                    if yolo_result.masks is not None and i < len(yolo_result.masks):
                        mask_np = yolo_result.masks.data[i].cpu().numpy()
                        mask_full = cv2.resize(mask_np, (fw, fh),
                                               interpolation=cv2.INTER_NEAREST)
                        fg = (mask_full[y1:y2, x1:x2] > 0.5).astype(np.uint8)
                        crop = crop * fg[:, :, np.newaxis]

                    t_reid = time.time()
                    try:
                        query_emb = ext.extract(crop)   # L2-normalised, crop already masked
                        _, sim = is_match(ref_state["emb"], query_emb, match_thresh)

                        # Colour veto: if clothing hue is clearly wrong, reduce score.
                        # We don't blend 50/50 — we just penalise obvious mismatches.
                        # Threshold 0.25: below this the hue distributions barely overlap.
                        HUE_VETO_THRESH = 0.25
                        cand_hist = clothing_hue_histogram(crop)
                        hue_sim   = hue_similarity(ref_state["hue_hist"], cand_hist)
                        if hue_sim < HUE_VETO_THRESH:
                            sim = sim * 0.5   # heavy penalty for clearly wrong colour
                        matched = sim >= match_thresh
                    except Exception as re:
                        query_emb, matched, sim, hue_sim = None, False, 0.0, 0.0
                        print(f"  person {i+1}: Re-ID extract FAILED: {re}")
                    candidates.append(((x1, y1, x2, y2), matched, sim, query_emb))
                    print(f"  person {i+1}: sim={sim:.3f} hue={hue_sim:.3f} "
                          f"reid_time={time.time()-t_reid:.2f}s box=({x1},{y1},{x2},{y2})")

                chosen_emb_out = None
                if candidates:
                    chosen = max(candidates, key=lambda c: c[2])
                    if chosen[2] >= 0.2:
                        # Strip emb from new_results (main loop only needs box/matched/sim)
                        new_results = [(chosen[0], chosen[1], chosen[2])]
                        chosen_emb_out = chosen[3]
                        print(f"[Re-ID] ★ Chosen: sim={chosen[2]:.3f} box={chosen[0]} "
                              f"| all: {[(round(c[2],3), c[0]) for c in candidates]}", flush=True)
                    else:
                        print(f"[Re-ID] No match (best={chosen[2]:.3f} < 0.2) "
                              f"| all: {[round(c[2],3) for c in candidates]}", flush=True)

                # Store candidates (without emb arrays) for on-screen debug boxes
                with det_lock:
                    det_state["all_candidates"] = [(c[0], c[1], c[2]) for c in candidates]

                # Persist latest scores to session so the UI can poll them.
                # cx = horizontal centre of the bounding box, used by the UI
                # to label people by position (left → right).
                if session_id in reid_sessions:
                    reid_sessions[session_id]["latest_candidates"] = [
                        {
                            "sim": round(float(c[2]), 3),
                            "matched": bool(c[1]),
                            "cx": int((c[0][0] + c[0][2]) / 2),
                        }
                        for c in candidates
                    ]

                print(f"[Re-ID] Total detection+reid time: {time.time()-t0:.2f}s", flush=True)
            except Exception as e:
                import traceback
                print(f"[Re-ID] Detection thread EXCEPTION: {e}", flush=True)
                traceback.print_exc()
            finally:
                with det_lock:
                    det_state["new_results"] = new_results
                    det_state["detection_frame"] = frame_copy
                    det_state["chosen_emb"] = chosen_emb_out
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
                    pass  # ref_state["adapted"] stays False until first confident detection
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

                if new_results:  # truthy: non-empty list ([] means no match found)
                    (nx1, ny1, nx2, ny2), n_matched, n_sim = new_results[0]
                    chosen_emb = det_state.get("chosen_emb")
                    all_cands  = det_state.get("all_candidates", [])

                    # ── Reference adaptation (one-shot) ──────────────────────────
                    # The reference was extracted from video 1. Due to cross-camera
                    # domain shift all video-2 similarities cluster at 0.93–0.97,
                    # making discrimination hard. Once we see one confident,
                    # unambiguous detection we update ref_emb_np to the video-2
                    # embedding so subsequent comparisons are within the same domain
                    # and give clear gaps (0.97 correct vs 0.82 others).
                    if not ref_state["adapted"] and chosen_emb is not None and len(all_cands) == 1:
                        # Guard: the solo detection must be near the person we are currently
                        # tracking. A solo detection that is far from the tracker is a
                        # different person — adapting on it permanently corrupts the reference.
                        # (No tracker yet at warm-up → allow unconditionally.)
                        _solo_near_tracker = True
                        if last_boxes:
                            _lx1, _ly1, _lx2, _ly2 = last_boxes[0][0]
                            _solo_iou = _iou((_lx1, _ly1, _lx2, _ly2), (nx1, ny1, nx2, ny2))
                            _solo_near_tracker = _solo_iou > 0.20
                            if not _solo_near_tracker:
                                ref_state["adapt_count"] = 0  # reset streak — different person
                                print(f"[Re-ID] Skip adaptation: solo detection far from "
                                      f"tracker (IoU={_solo_iou:.2f}) — different person",
                                      flush=True)

                        if _solo_near_tracker:
                            # Require 3 consecutive qualifying detections of OUR tracked person
                            # before locking in. The hue histogram is anchored to video-1 and
                            # is never overwritten.
                            _adapt_crop = init_frame[ny1:ny2, nx1:nx2] if init_frame is not None else None
                            _adapt_hist = clothing_hue_histogram(_adapt_crop) if (
                                _adapt_crop is not None and _adapt_crop.size > 0) else None
                            _adapt_hue  = hue_similarity(ref_state["hue_hist"], _adapt_hist)

                            if n_sim >= 0.80 and _adapt_hue >= 0.35:
                                ref_state["adapt_count"] += 1
                                if ref_state["adapt_count"] >= 3:
                                    ref_state["emb"] = chosen_emb
                                    ref_state["adapted"] = True
                                    print(f"[Re-ID] ✓ Reference adapted to video-2 domain "
                                          f"(sim={n_sim:.3f} hue={_adapt_hue:.3f} count=3)", flush=True)
                                else:
                                    print(f"[Re-ID] Adaptation candidate "
                                          f"{ref_state['adapt_count']}/3 "
                                          f"(sim={n_sim:.3f} hue={_adapt_hue:.3f})", flush=True)
                            else:
                                ref_state["adapt_count"] = 0  # mismatch breaks the streak
                                print(f"[Re-ID] Skip adaptation: sim={n_sim:.3f} "
                                      f"hue={_adapt_hue:.3f} (wrong colour or low confidence)",
                                      flush=True)

                    # ── Simple gap-based stability ────────────────────────────────
                    # Require the winner to be clearly ahead of the second-best.
                    # Only applies once we're tracking someone (last_boxes set).
                    SWITCH_THRESHOLD = 0.05
                    if last_boxes and len(all_cands) >= 2:
                        sorted_cands = sorted(all_cands, key=lambda c: c[2], reverse=True)
                        best_sim   = sorted_cands[0][2]
                        second_sim = sorted_cands[1][2]
                        if best_sim - second_sim < SWITCH_THRESHOLD:
                            # Scores too close — find which candidate overlaps current
                            # tracked box and keep that one instead of switching
                            last_box_coords = last_boxes[0][0]
                            best_match = max(all_cands,
                                             key=lambda c: _iou(last_box_coords, c[0]))
                            if _iou(last_box_coords, best_match[0]) > 0.25:
                                nx1,ny1,nx2,ny2 = best_match[0]
                                n_matched = best_match[1]
                                n_sim     = best_match[2]
                                print(f"[Re-ID] Gap too small ({best_sim:.3f}-{second_sim:.3f}"
                                      f"={best_sim-second_sim:.3f}), keeping current box",
                                      flush=True)

                    # Smart reinit logic:
                    #
                    # Case A — detection agrees with tracker (IoU > 0.30):
                    #   Keep the running tracker, just update the sim/matched metadata.
                    #
                    # Case B — detection is FAR from a healthy tracker (IoU < 0.30)
                    #   AND the new sim is not meaningfully higher than what we already track:
                    #   This is a missed-detection — YOLO failed to find the correct person
                    #   and is returning someone else. Keep tracking, discard the result.
                    #   Example from logs: tracker on person at x=252 (sim=0.968), YOLO
                    #   only detects person at x=670 (sim=0.949). 0.949 < 0.968+0.03 → ignore.
                    #
                    # Case C — detection is far from tracker AND the new sim is clearly
                    #   higher (> current + 0.03): the tracker has drifted to the wrong
                    #   person and the correct person was just found. Reinit on current frame.
                    #   Example: tracker on wrong person (sim=0.930), correct person detected
                    #   at sim=0.993. 0.993 > 0.930+0.03 → reinit.
                    should_reinit = True
                    if trackers and last_boxes:
                        lx1, ly1, lx2, ly2 = last_boxes[0][0]
                        current_fail_count  = trackers[0][4]
                        current_tracked_sim = trackers[0][2]
                        det_iou = _iou((lx1, ly1, lx2, ly2), (nx1, ny1, nx2, ny2))

                        if det_iou > 0.30:
                            # Case A: detection and tracker agree — keep the CSRT box
                            # as the drawn box. Using the YOLO detection box here would
                            # cause size fluctuation because YOLO and CSRT produce
                            # differently-sized bounding boxes (YOLO = full body,
                            # CSRT = learned region). Only sim/matched metadata updates.
                            existing_tr = trackers[0][0]
                            trackers   = [(existing_tr, n_matched, n_sim, (nx1, ny1, nx2, ny2), 0)]
                            last_boxes = [((lx1, ly1, lx2, ly2), n_matched, n_sim)]
                            should_reinit = False
                            print(f"[Re-ID] Frame {frame_idx}: tracker agrees with detection "
                                  f"(IoU={det_iou:.2f}), keeping tracker sim={n_sim:.3f}", flush=True)
                        elif current_fail_count == 0 and n_sim <= current_tracked_sim + 0.03:
                            # Case B: missed detection — YOLO lost the correct person
                            last_boxes = [((lx1, ly1, lx2, ly2), trackers[0][1], current_tracked_sim)]
                            should_reinit = False
                            print(f"[Re-ID] Frame {frame_idx}: missed detection "
                                  f"(IoU={det_iou:.2f}, new_sim={n_sim:.3f} vs tracked={current_tracked_sim:.3f})"
                                  f" — keeping healthy tracker", flush=True)
                        # else Case C: fall through → reinit below

                    if should_reinit:
                        print(f"[Re-ID] Frame {frame_idx}: reinit tracker on current frame "
                              f"sim={n_sim:.3f} box=({nx1},{ny1},{nx2},{ny2})", flush=True)
                        tr = create_tracker()
                        tr.init(frame, (nx1, ny1, nx2 - nx1, ny2 - ny1))
                        trackers   = [(tr, n_matched, n_sim, (nx1, ny1, nx2, ny2), 0)]
                        last_boxes = [((nx1, ny1, nx2, ny2), n_matched, n_sim)]
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
                            # Clip to frame bounds — CSRT can drift outside the frame
                            fh, fw = frame.shape[:2]
                            x  = max(0, min(x, fw - 1))
                            y  = max(0, min(y, fh - 1))
                            bw = max(1, min(bw, fw - x))
                            bh = max(1, min(bh, fh - y))
                            new_box = (x, y, x + bw, y + bh)

                            # Drift guard: if the tracker box has moved too far from the
                            # last confirmed detection box, it has likely drifted to a
                            # different person. Discard the update and hold last_box.
                            lx1, ly1, lx2, ly2 = last_box
                            last_cx, last_cy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
                            new_cx, new_cy = (x + x + bw) / 2.0, (y + y + bh) / 2.0
                            center_drift = ((last_cx - new_cx) ** 2 + (last_cy - new_cy) ** 2) ** 0.5
                            last_area = max(1, (lx2 - lx1) * (ly2 - ly1))
                            new_area = bw * bh
                            area_ratio = new_area / last_area

                            if center_drift > 250 or area_ratio > 2.0 or area_ratio < 0.3:
                                # Tracker drifted — hold position, wait for next detection
                                print(f"[Re-ID] Frame {frame_idx}: tracker DRIFT detected "
                                      f"(center_drift={center_drift:.0f}px, area_ratio={area_ratio:.2f}) — holding last box", flush=True)
                                last_boxes.append((last_box, matched, sim))
                                updated.append((tr, matched, sim, last_box, fail_count + 1))
                            else:
                                last_boxes.append((new_box, matched, sim))
                                updated.append((tr, matched, sim, new_box, 0))
                                if frame_idx % 30 == 0:
                                    print(f"[Re-ID] Frame {frame_idx}: tracker OK box={new_box} drift={center_drift:.0f}px ({track_ms:.1f}ms)", flush=True)
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
            # Reset tracker at loop boundary. The CSRT internal state from the
            # last frame of the video is invalid when the video restarts from
            # frame 0 — the person is at a completely different position.
            # Detection fires at frame 0 of the new loop and reinits cleanly.
            trackers = []
            last_boxes = []

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


@app.get("/api/reid-reference-crop/{session_id}")
async def reid_reference_crop(session_id: str):
    """Return the masked reference person crop as a JPEG for display in the UI."""
    session = reid_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Re-ID session not found")
    crop = session.get("reference_crop")
    if crop is None:
        raise HTTPException(status_code=404, detail="Reference crop not set yet")
    ret, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ret:
        raise HTTPException(status_code=500, detail="Could not encode crop image")
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


@app.get("/api/reid-candidates/{session_id}")
async def reid_candidates(session_id: str):
    """Return the similarity scores of all people detected in the latest Re-ID frame."""
    session = reid_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Re-ID session not found")
    return {"candidates": session.get("latest_candidates", [])}


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
