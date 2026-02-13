"""
FastAPI Backend for VisionAI Studio
Connects React frontend with Python object detection and tracking
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from pathlib import Path
import tempfile
import uuid
import asyncio
import json
import io
from PIL import Image
import os
from dotenv import load_dotenv

# Import your detection and tracking modules
from detection import ObjectDetector
from utils import get_device
from llm.image_describer import describe_image
import config

# Load environment variables
load_dotenv()

app = FastAPI(title="VisionAI Studio API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
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


def get_detector():
    """Lazy load detector on first use"""
    global device, detector
    if detector is None:
        device = get_device()
        detector = ObjectDetector(device=device)
        print(f"Detector initialized on {device}")
    return detector


# Pydantic models
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
    frame_number: Optional[int] = None  # Which frame this box belongs to


class DetectionRequest(BaseModel):
    session_id: str
    query: str
    threshold: float = 0.5


class TrackingRequest(BaseModel):
    session_id: str
    bbox: Dict[str, float]  # {x, y, width, height} in percentages


class DetectionResponse(BaseModel):
    boxes: List[BoundingBox]
    fps: float
    frame_count: int


# Helper functions
def save_uploaded_video(file: UploadFile) -> Path:
    """Save uploaded video to temp directory"""
    temp_dir = Path(tempfile.gettempdir()) / "visionai"
    temp_dir.mkdir(exist_ok=True)
    
    video_id = str(uuid.uuid4())
    video_path = temp_dir / f"{video_id}.mp4"
    
    with open(video_path, "wb") as f:
        f.write(file.file.read())
    
    return video_path


def percentage_to_pixels(bbox: Dict[str, float], frame_width: int, frame_height: int) -> tuple:
    """Convert percentage coordinates to pixel coordinates"""
    x = int(bbox["x"] * frame_width / 100)
    y = int(bbox["y"] * frame_height / 100)
    w = int(bbox["width"] * frame_width / 100)
    h = int(bbox["height"] * frame_height / 100)
    return (x, y, w, h)


def pixels_to_percentage(x: int, y: int, w: int, h: int, frame_width: int, frame_height: int) -> Dict[str, float]:
    """Convert pixel coordinates to percentage coordinates"""
    return {
        "x": (x / frame_width) * 100,
        "y": (y / frame_height) * 100,
        "width": (w / frame_width) * 100,
        "height": (h / frame_height) * 100,
    }


@app.get("/api/track-label/{session_id}")
async def get_track_label(session_id: str):
    """Return last computed tracking label for this session"""
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    label = session.get("llm_label", "tracked object")
    return {"label": label}


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
            cap = cv2.VideoCapture(str(video_path))
            
            # Get video FPS for proper timing
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0  # Default to 30 FPS
            
            frame_idx = 0
            last_boxes = []
            
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
                    
                    # Store boxes scaled to original size
                    last_boxes = []
                    for box, score in zip(results['boxes'], results['scores']):
                        if score >= threshold:
                            x1, y1, x2, y2 = box.tolist()
                            # Scale back to original
                            scale_x = w / new_width
                            scale_y = h / new_height
                            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                            last_boxes.append(((x1, y1, x2, y2), score))
                
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
        
        # Get video metadata
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Create session
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            "video_path": video_path,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "tracker": None,
            "mode": None
        }
        
        return {
            "session_id": session_id,
            "metadata": {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": duration
            }
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
    bbox_x: float,
    bbox_y: float, 
    bbox_width: float,
    bbox_height: float,
    render_w: Optional[float] = None,
    render_h: Optional[float] = None,
    frame_time: Optional[float] = None
):
    """
    Stream video with CSRT tracking - draws boxes on backend and streams
    Gets label from LLM first, then tracks with that label
    """
    from fastapi.responses import StreamingResponse
    from llm.image_describer import describe_image
    import io
    
    def generate_tracking_frames():
        import time
        
        session = active_sessions.get(session_id)
        if not session:
            return
        
        video_path = session["video_path"]
        fps_session = session.get("fps") or 30.0
        
        # Get first frame to get LLM label (seek to requested frame_time if provided)
        cap_temp = cv2.VideoCapture(str(video_path))
        if frame_time is not None and frame_time >= 0:
            target_frame = int(frame_time * fps_session)
            cap_temp.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, first_frame = cap_temp.read()
        cap_temp.release()
        
        if not ret:
            return
        
        original_h, original_w = first_frame.shape[:2]
        
        # Convert percentage coordinates to pixels; if render size provided, scale to original
        if render_w and render_h:
            disp_x = (bbox_x / 100.0) * render_w
            disp_y = (bbox_y / 100.0) * render_h
            disp_w = (bbox_width / 100.0) * render_w
            disp_h = (bbox_height / 100.0) * render_h

            scale_x = original_w / render_w
            scale_y = original_h / render_h

            x_px = int(disp_x * scale_x)
            y_px = int(disp_y * scale_y)
            w_px = int(disp_w * scale_x)
            h_px = int(disp_h * scale_y)
        else:
            x_px = int((bbox_x / 100) * original_w)
            y_px = int((bbox_y / 100) * original_h)
            w_px = int((bbox_width / 100) * original_w)
            h_px = int((bbox_height / 100) * original_h)
        
        print(f"Video dimensions: {original_w}x{original_h}")
        print(f"Box percentages: x={bbox_x:.1f}%, y={bbox_y:.1f}%, w={bbox_width:.1f}%, h={bbox_height:.1f}%")
        print(f"Box pixels: x={x_px}, y={y_px}, w={w_px}, h={h_px}")
        
        # Crop ROI and get LLM label (with bounds checking)
        y_end = min(y_px + h_px, original_h)
        x_end = min(x_px + w_px, original_w)
        roi = first_frame[y_px:y_end, x_px:x_end]
        
        try:
            # Save ROI temporarily
            import tempfile
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            api_key = os.getenv("ANTHROPIC_API_KEY")
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, roi)
                tmp_path = tmp.name
            
            # Get descriptive label from LLM
            llm_label = describe_image(tmp_path, api_key)
            print(f"LLM Label: {llm_label}")
            # store label for frontend retrieval
            session["llm_label"] = llm_label
            
            # Clean up temp file
            os.unlink(tmp_path)
        except Exception as e:
            print(f"LLM labeling failed: {e}")
            llm_label = "tracked object"
            session["llm_label"] = llm_label
        
        # Loop the video indefinitely
        while True:
            cap = cv2.VideoCapture(str(video_path))
            
            # Get video FPS for proper timing
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps > 0 else fps_session
            frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0
            
            # Seek to requested frame_time if provided
            if frame_time is not None and frame_time >= 0:
                target_frame = int(frame_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            # Get first frame to initialize tracker
            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                continue
            
            # Use the SAME dimensions as the original frame; scale with render metrics if provided
            if render_w and render_h:
                disp_x = (bbox_x / 100.0) * render_w
                disp_y = (bbox_y / 100.0) * render_h
                disp_w = (bbox_width / 100.0) * render_w
                disp_h = (bbox_height / 100.0) * render_h

                scale_x = original_w / render_w
                scale_y = original_h / render_h

                x_px_loop = int(disp_x * scale_x)
                y_px_loop = int(disp_y * scale_y)
                w_px_loop = int(disp_w * scale_x)
                h_px_loop = int(disp_h * scale_y)
            else:
                x_px_loop = int((bbox_x / 100) * original_w)
                y_px_loop = int((bbox_y / 100) * original_h)
                w_px_loop = int((bbox_width / 100) * original_w)
                h_px_loop = int((bbox_height / 100) * original_h)
            
            # Initialize CSRT tracker
            tracker = cv2.TrackerCSRT.create()
            tracker.init(first_frame, (x_px_loop, y_px_loop, w_px_loop, h_px_loop))
            
            frame_idx = 0
            
            # Process first frame with LLM label
            cv2.rectangle(first_frame, (x_px, y_px), (x_px + w_px, y_px + h_px), (0, 255, 0), 3)
            (text_width, text_height), baseline = cv2.getTextSize(llm_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(first_frame, (x_px, y_px - text_height - baseline - 5), (x_px + text_width, y_px), (0, 0, 0), -1)
            cv2.putText(first_frame, llm_label, (x_px, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', first_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            while cap.isOpened():
                frame_start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update tracker
                success, bbox = tracker.update(frame)
                
                if success:
                    # Draw tracking box with LLM label
                    x, y, w_box, h_box = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 3)
                    (text_width, text_height), baseline = cv2.getTextSize(llm_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x, y - text_height - baseline - 5), (x + text_width, y), (0, 0, 0), -1)
                    cv2.putText(frame, llm_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Lost tracking
                    cv2.putText(frame, "LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                
                # Encode and yield frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                frame_idx += 1
                
                # Control frame rate
                elapsed = time.time() - frame_start_time
                target_delay = frame_delay * 1.5
                sleep_time = max(target_delay - elapsed, 0.001)
                time.sleep(sleep_time)
            
            cap.release()
    
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

    if render_w and render_h:
        disp_x = (bbox_x / 100.0) * render_w
        disp_y = (bbox_y / 100.0) * render_h
        disp_w = (bbox_width / 100.0) * render_w
        disp_h = (bbox_height / 100.0) * render_h

        scale_x = original_w / render_w
        scale_y = original_h / render_h

        x_px = int(disp_x * scale_x)
        y_px = int(disp_y * scale_y)
        w_px = int(disp_w * scale_x)
        h_px = int(disp_h * scale_y)
    else:
        x_px = int((bbox_x / 100) * original_w)
        y_px = int((bbox_y / 100) * original_h)
        w_px = int((bbox_width / 100) * original_w)
        h_px = int((bbox_height / 100) * original_h)

    # Clamp ROI within frame
    x_px = max(0, min(x_px, original_w - 1))
    y_px = max(0, min(y_px, original_h - 1))
    w_px = max(1, min(w_px, original_w - x_px))
    h_px = max(1, min(h_px, original_h - y_px))

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
        tracker = cv2.TrackerCSRT.create()
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
    
    raise HTTPException(status_code=404, detail="Session not found")


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
    print("Frontend: http://localhost:5173")
    print("=" * 70)
    print("Detector will be initialized on first request (lazy loading)")
    print("=" * 70)
    print("")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
