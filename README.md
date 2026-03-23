# Multi-Feed Object Tracker

A modular object detection and person re-identification system using Grounding DINO and ResNet50.

## Features

- 🎯 **Zero-shot object detection** with Grounding DINO
- 🧠 **Person re-identification** using ResNet50 features
- 💻 **GPU acceleration** (CUDA, MPS, or CPU)
- 📊 **Visual bounding box overlay**
- ⚡ **Performance timing**
- 🔧 **Configurable thresholds**
- 📁 **Clean modular architecture**

## Business Requirements

### Project Overview
Multi-Feed Object Tracker is designed to provide an intelligent video surveillance and analysis solution that enables real-time object detection and person re-identification across multiple video feeds.

### Key Business Objectives
- **Automated Surveillance**: Reduce manual monitoring effort by providing automated object detection and tracking capabilities
- **Person Re-identification**: Enable tracking of individuals across different camera feeds and time periods
- **Real-time Processing**: Provide near real-time detection and tracking for security and monitoring applications
- **Cost-Effective Solution**: Utilize open-source models and efficient algorithms to minimize operational costs
- **Scalability**: Support multiple video feeds simultaneously with modular architecture

### Target Use Cases
1. **Security & Surveillance**: Monitor restricted areas and track persons of interest
2. **Retail Analytics**: Track customer movement patterns and behavior in stores
3. **Traffic Management**: Monitor vehicle and pedestrian traffic in urban environments
4. **Event Management**: Track attendee movement and crowd density at large events
5. **Industrial Safety**: Monitor worker presence in hazardous zones

### Success Criteria
- Accurate object detection with configurable confidence thresholds
- Reliable person re-identification across video frames
- Responsive web interface for easy operation by non-technical users
- Support for common video formats and resolutions
- Minimal hardware requirements with GPU acceleration support

### Constraints
- Must support GPU acceleration (CUDA/MPS) and CPU fallback
- Must provide REST API for integration with external systems
- Must maintain user privacy and data security standards
- Must be deployable on standard server infrastructure

## Project Structure

```
Multi-Feed_Tracker/
├── app.py                    # FastAPI backend server
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── start_app.sh             # Startup script
├── stop_app.sh              # Stop script
├── README.md
├── QUICK_START.md
│
├── detection/                # Object detection module
│   ├── __init__.py
│   ├── detector.py          # Grounding DINO detector
│   └── visualizer.py        # Bounding box visualization
│
├── llm/                      # LLM integration
│   ├── __init__.py
│   └── image_describer.py   # Claude API for object labeling
│
├── re_id/                    # Person Re-Identification (optional)
│   ├── __init__.py
│   ├── embedding_extractor.py  # fast-reid embedding extraction
│   └── matcher.py             # Cosine similarity matching
│
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── device.py            # Device selection (CUDA/MPS/CPU)
│   └── image_loader.py      # Image loading utilities
│
└── frontend/                 # React web interface
    └── vision-explorer/
        ├── src/
        ├── package.json
        └── vite.config.ts
```

## Installation

1. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### Start the Application (Backend + Frontend)

```bash
# Activate virtual environment
source venv/bin/activate

# Start both services
./start_app.sh
```

This will start:
- **Backend** (FastAPI): http://localhost:8080
- **Frontend** (React): http://localhost:8000
- **API Docs**: http://localhost:8080/docs

### Stop the Application

```bash
./stop_app.sh
```

Or press `Ctrl+C` in the terminal running `start_app.sh`.

## Usage

### 1. Web Interface (Recommended)

1. Start the application with `./start_app.sh`
2. Open http://localhost:8000 in your browser
3. Upload a video file
4. Choose mode:
   - **Identify**: Text-based object detection (e.g., "person", "car")
   - **Track**: Draw a box to track a specific object (CSRT + AI labeling)
   - **Re-ID**: Find a person across two videos (upload video 1, draw box, upload video 2)

### 2. Re-ID Mode Setup (Optional)

Re-ID uses [fast-reid](https://github.com/JDAI-CV/fast-reid) for person re-identification. Install it separately:

```bash
# Clone fast-reid
git clone https://github.com/JDAI-CV/fast-reid.git
cd fast-reid
pip install -r requirements.txt
python setup.py develop
cd ..
pip install faiss-cpu opencv-python
```

Download a pretrained model (e.g., from the [fast-reid Model Zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md)) and place it as `fast-reid/model.pth`, or set environment variables:

```bash
export FAST_REID_PATH=/path/to/fast-reid
export REID_WEIGHTS_PATH=/path/to/model.pth   # e.g. bagtricks_R50_market.pth
export REID_CONFIG_PATH=/path/to/fast-reid/configs/Market1501/bagtricks_R50.yml
export REID_DEVICE=cuda   # or cpu, mps
```

### 3. Video Tracking with Object Detection

**Run video tracking pipeline:**
```bash
python starter.py  # Uncomment Demo 1 in starter.py
```

### 4. Programmatic Usage

**Object Detection:**
```python
from detection import ObjectDetector
from utils import get_device

# Initialize
device = get_device()
detector = ObjectDetector(device=device)

# Detect objects in image
results, time = detector.detect(image, ["a person", "a car"])
```

## Configuration

Edit `config.py` to customize:
- Model selection
- Detection thresholds
- Visualization colors and styles
- Default labels
- Re-ID: `REID_CONFIG_PATH`, `REID_WEIGHTS_PATH`, `REID_DEVICE`, `REID_MATCH_THRESHOLD` (env vars)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- transformers
- Pillow
- matplotlib
- requests

## Performance

- **GPU Support:** CUDA (NVIDIA), MPS (Apple Silicon), CPU fallback
- **Inference timing:** Automatically measured and reported
- **Models:**
  - Detection: Grounding DINO Tiny (lightweight and fast)
  - Tracking: CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)

## Features

- ✅ **Re-ID mode**: Find a person across two videos (fast-reid)
- ✅ Multi-object tracking with CSRT
- ✅ AI-powered object labeling with Claude API
- ✅ Detect-then-track cycle for automatic detection mode
- ✅ Confidence threshold filtering
- ✅ Real-time video streaming with bounding boxes
- ✅ Frame-accurate pause and resume
- ✅ Multiple trackers per video

## License

MIT License
