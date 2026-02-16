# Multi-Feed Tracker - Quick Start Guide

## Prerequisites

- Python 3.8+ with virtual environment activated
- Node.js and npm installed
- Virtual environment created and activated

## First Time Setup

1. **Activate virtual environment:**
```bash
source venv/bin/activate
```

2. **Install Python dependencies (if not already done):**
```bash
pip install -r requirements.txt
```

3. **Install frontend dependencies:**
```bash
cd frontend/vision-explorer
npm install
cd ../..
```

## Running the Application

### Start Everything (Recommended)

```bash
./start_app.sh
```

This will:
- ✅ Check that virtual environment is activated
- ✅ Check and install frontend dependencies if needed
- ✅ Start FastAPI backend on port 8080
- ✅ Start React frontend on port 8000
- ✅ Wait for both services to be ready

**Access the application:**
- Frontend: http://localhost:8000
- Backend API: http://localhost:8080
- API Documentation: http://localhost:8080/docs

### Stop Everything

Option 1: Press `Ctrl+C` in the terminal running `start_app.sh`

Option 2: Run the stop script:
```bash
./stop_app.sh
```

## Manual Running (Alternative)

If you prefer to run services separately:

### Backend Only
```bash
python app.py
```

### Frontend Only
```bash
cd frontend/vision-explorer
npm run dev
```

## Using the Application

### Track Mode (Manual Tracking)
1. Upload a video
2. Click "Track" mode
3. Draw a box around the object you want to track
4. The system uses CSRT tracking to follow the object
5. Pause anytime to add more trackers
6. AI will automatically label tracked objects

### Identify Mode (Auto Detection)
1. Upload a video
2. Click "Identify" mode
3. Adjust confidence threshold (default: 50%)
4. Adjust detection interval (default: 30 frames)
5. System automatically detects and tracks objects
6. Uses detect-then-track cycle: detection every N frames, CSRT tracking between detections

## Configuration

### Backend Port (8080)
Edit `app.py` if you need to change the backend port.

### Frontend Port (8000)
Edit `frontend/vision-explorer/vite.config.ts` to change the frontend port.

### API URL
If you change backend port, update `frontend/vision-explorer/.env`:
```
VITE_API_URL=http://localhost:YOUR_PORT
```

## Troubleshooting

### Backend fails to start
- Check `backend.log` for detailed error messages
- Ensure port 8080 is not already in use: `lsof -ti:8080`
- Make sure virtual environment is activated

### Frontend fails to start
- Ensure port 8000 is not already in use: `lsof -ti:8000`
- Delete `frontend/vision-explorer/node_modules` and run `npm install` again
- Check Node.js version: `node --version` (should be 16+)

### CORS errors
- Make sure both services are running
- Clear browser cache
- Check that frontend is accessing `http://localhost:8080` (not 127.0.0.1)

### Video not playing
- Ensure video file is in a supported format (MP4, WebM)
- Check browser console for errors
- Verify video file is not corrupted

## Features

✅ Multi-tracker support (add multiple trackers to one video)
✅ CSRT tracking (robust real-time tracking)
✅ AI labeling with Claude API
✅ Confidence threshold filtering
✅ Detect-then-track cycle for Identify mode
✅ Frame-accurate pause and resume
✅ Smooth drawing experience for all trackers
✅ Debug ROI visualization (saved to `debug_rois/` folder)

## Logs and Debug Files

- `backend.log` - Backend server logs
- `debug_rois/` - ROI images sent to backend for tracking

## Support

For issues or questions, check:
- API Documentation: http://localhost:8080/docs
- Backend logs: `backend.log`
- Frontend console: Browser Developer Tools (F12)
