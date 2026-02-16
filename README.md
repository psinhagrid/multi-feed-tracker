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

## Project Structure

```
Multi-Feed_Tracker/
├── config.py                  # Global configuration
├── starter.py                 # Main entry point
├── requirements.txt           # Dependencies
├── README.md
│
├── detection/                 # Object detection module
│   ├── __init__.py
│   ├── detector.py           # Grounding DINO detector
│   └── visualizer.py         # Bounding box visualization
│
├── tracking/                  # Person tracking & ReID module
│   ├── __init__.py
│   └── feature_extractor.py  # ResNet50 feature extraction
│
├── video_processing/          # Video processing pipeline
│   ├── __init__.py
│   └── video_processor.py    # Video tracking with ByteTrack
│
├── tools/                     # Utility tools
│   └── roi_selector.py       # ROI selection and labeling
│
├── llm/                       # LLM integration
│   ├── __init__.py
│   └── image_describer.py    # Claude API for image labeling
│
├── utils/                     # Utilities
│   ├── __init__.py
│   ├── device.py             # Device selection (CUDA/MPS/CPU)
│   └── image_loader.py       # Image loading (URL/local)
│
└── examples/                  # Example scripts
    ├── detection_example.py  # Detection examples
    ├── reid_example.py       # Person ReID examples
    └── reid_workflow.py      # Full ReID workflow
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
4. Choose tracking mode:
   - **Track**: Manual tracker placement with CSRT tracking
   - **Identify**: Automatic detection with AI labeling

### 2. Video Tracking with Object Detection

**Run video tracking pipeline:**
```bash
python starter.py  # Uncomment Demo 1 in starter.py
```

### 3. Programmatic Usage

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

- ✅ Multi-object tracking with CSRT
- ✅ AI-powered object labeling with Claude API
- ✅ Detect-then-track cycle for automatic detection mode
- ✅ Confidence threshold filtering
- ✅ Real-time video streaming with bounding boxes
- ✅ Frame-accurate pause and resume
- ✅ Multiple trackers per video

## License

MIT License
