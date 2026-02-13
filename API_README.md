# VisionAI Studio - API Documentation

FastAPI backend that connects your Python object detection/tracking system with the React frontend.

## Architecture

```
React Frontend (Port 5173)
    ↕ HTTP/WebSocket
FastAPI Backend (Port 8000)
    ↕ Python
Detection (Grounding DINO) + Tracking (CSRT) + LLM (Claude)
```

## Setup

### 1. Install Backend Dependencies

```bash
pip install fastapi uvicorn python-multipart websockets
# or
pip install -r requirements.txt
```

### 2. Start Backend Server

```bash
python app.py
```

Backend will run on: `http://localhost:8000`

### 3. Start Frontend

```bash
cd frontend/vision-explorer
npm install
npm run dev
```

Frontend will run on: `http://localhost:5173`

## API Endpoints

### Health Check
```
GET /
Returns: { status, service, device, version }
```

### Upload Video
```
POST /api/upload-video
Body: multipart/form-data with video file
Returns: {
  session_id: string,
  metadata: { fps, frame_count, width, height, duration }
}
```

### Object Detection (Identify Mode)
```
POST /api/detect
Body: {
  session_id: string,
  query: string,  // e.g., "person in yellow"
  threshold: float  // 0.0 - 1.0
}
Returns: {
  boxes: [{ id, x, y, width, height, label, confidence, type }],
  fps: float,
  frame_count: int
}
```

### Start Tracking (Track Mode)
```
POST /api/track
Body: {
  session_id: string,
  bbox: { x, y, width, height }  // percentages
}
Returns: {
  session_id: string,
  status: "tracking_initialized",
  bbox: { x, y, width, height }
}
```

### Stream Tracking Results
```
GET /api/track/{session_id}/stream
Returns: Server-Sent Events stream
Event data: {
  frame: int,
  bbox: { x, y, width, height },
  status: "active" | "lost",
  confidence: float
}
```

### Label ROI with LLM
```
POST /api/label-roi?session_id={session_id}
Body: multipart/form-data with cropped image
Returns: {
  label: string,  // e.g., "person in yellow"
  session_id: string
}
```

### Delete Session
```
DELETE /api/session/{session_id}
Returns: { status: "session_deleted", session_id }
```

### WebSocket Tracking (Alternative to SSE)
```
WS /ws/track/{session_id}
Sends: {
  frame: int,
  bbox: { x, y, width, height },
  status: "active" | "lost"
}
```

## Frontend Integration

### Example: Upload Video

```typescript
const formData = new FormData();
formData.append('file', videoFile);

const response = await fetch('http://localhost:8000/api/upload-video', {
  method: 'POST',
  body: formData
});

const { session_id, metadata } = await response.json();
```

### Example: Detect Objects

```typescript
const response = await fetch('http://localhost:8000/api/detect', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: sessionId,
    query: "person in yellow",
    threshold: 0.5
  })
});

const { boxes, fps, frame_count } = await response.json();
```

### Example: Track Object

```typescript
// 1. Initialize tracking
await fetch('http://localhost:8000/api/track', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: sessionId,
    bbox: { x: 15, y: 20, width: 12, height: 28 }
  })
});

// 2. Stream tracking results
const eventSource = new EventSource(
  `http://localhost:8000/api/track/${sessionId}/stream`
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Frame:', data.frame, 'Status:', data.status);
  if (data.bbox) {
    updateBoundingBox(data.bbox);
  }
};
```

## Coordinate System

All bounding box coordinates are in **percentages** (0-100):
- `x`: Left edge (% of video width)
- `y`: Top edge (% of video height)
- `width`: Box width (% of video width)
- `height`: Box height (% of video height)

This ensures consistent display across different video resolutions.

## Models Used

1. **Grounding DINO** (Identify mode)
   - Zero-shot object detection
   - Text-based queries
   - Threshold: configurable (default 0.5)

2. **CSRT Tracker** (Track mode)
   - OpenCV visual tracker
   - No detection model needed
   - Adapts box size automatically

3. **Claude API** (LLM labeling)
   - Describes objects in ROI
   - Generates tracking-friendly labels
   - Requires ANTHROPIC_API_KEY in .env

## Configuration

Edit `config.py` to adjust:
- `DETECTION_THRESHOLD`: Min confidence for detections
- `DETECTION_FRAME_INTERVAL`: Frames between detections
- `RESIZE_WIDTH`: Frame width for faster processing

## Error Handling

All endpoints return standard HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid file, parameters)
- `404`: Session not found
- `500`: Server error

Error response format:
```json
{
  "detail": "Error message"
}
```

## Session Management

- Sessions are created on video upload
- Each session has unique `session_id`
- Sessions store video path, tracker state, metadata
- Clean up sessions with `DELETE /api/session/{session_id}`
- Videos stored in: `/tmp/visionai/`

## Performance Tips

1. **Detection**: Runs on every Nth frame (config: `DETECTION_FRAME_INTERVAL`)
2. **Tracking**: Processes every frame (faster than detection)
3. **Frame Resize**: Configurable via `RESIZE_WIDTH` in config
4. **GPU**: Automatically uses MPS (Apple Silicon) or CUDA if available

## Development

Start backend in dev mode with auto-reload:
```bash
uvicorn app:app --reload --port 8000
```

Check API docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Next Steps

To connect the frontend:
1. Update `Index.tsx` to call real API instead of mock data
2. Replace `handleStartDetection` with API call to `/api/detect`
3. Replace `handleDrawBox` with API call to `/api/track`
4. Add video upload handling with `/api/upload-video`
5. Implement SSE/WebSocket for real-time tracking updates
