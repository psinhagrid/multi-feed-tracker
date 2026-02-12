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

## Usage

### 1. Video Tracking with Object Detection

**Run video tracking pipeline:**
```bash
python starter.py  # Uncomment Demo 1 in starter.py
```

### 2. ROI Selection and Labeling

**Select regions and generate AI labels:**
```bash
python starter.py  # Uncomment Demo 2 in starter.py (default)
```

### 3. Person Re-Identification

**Compare two images:**
```bash
python examples/reid_example.py \
    --image1 person1.jpg \
    --image2 person2.jpg
```

**Batch comparison (1 vs many):**
```bash
python examples/reid_example.py \
    --image1 reference.jpg \
    --image2 img1.jpg img2.jpg img3.jpg \
    --batch
```

**Quiet mode (score only):**
```bash
python examples/reid_example.py \
    --image1 person1.jpg \
    --image2 person2.jpg \
    --quiet
```

### 3. Programmatic Usage

**Object Detection:**
```python
from detection import ObjectDetector, DetectionVisualizer
from utils import get_device, load_image

# Initialize
device = get_device()
detector = ObjectDetector(device=device)
visualizer = DetectionVisualizer()

# Detect
image = load_image("path/to/image.jpg")
results, time = detector.detect(image, ["a person", "a car"])

# Visualize
visualizer.draw_boxes(image, results)
```

**Person Re-Identification:**
```python
from tracking import FeatureExtractor

# Initialize
extractor = FeatureExtractor()

# Compare two images
features1 = extractor.extract_features("person1.jpg")
features2 = extractor.extract_features("person2.jpg")

similarity, interpretation = extractor.compute_similarity(
    features1, features2, interpret=True
)

print(f"Similarity: {similarity:.4f}")
print(f"Result: {interpretation}")
```

## Similarity Interpretation

When comparing persons, the similarity score is interpreted as:

| Score Range | Interpretation |
|------------|----------------|
| > 0.8      | ✅ Very likely same person |
| 0.6-0.8    | ⚠️ Possible match |
| < 0.6      | ❌ Probably different person |

**Note:** These thresholds should be tuned based on your specific use case.

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
  - ReID: ResNet50 (2048-dim features)

## Examples

Run the example scripts to see the system in action:

```bash
# Detection examples
python examples/detection_example.py

# Person ReID examples  
python examples/reid_example.py

# Full ReID workflow (detect + track)
python examples/reid_workflow.py
```

## License

MIT License
