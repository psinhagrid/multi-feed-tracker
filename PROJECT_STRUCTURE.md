# Project Structure

## Directory Layout

```
Multi-Feed_Tracker/
│
├── 📄 config.py                    # Global configuration settings
├── 📄 main.py                      # Main CLI for object detection
├── 📄 requirements.txt             # Python dependencies
├── 📄 README.md                    # Project documentation
├── 📄 PROJECT_STRUCTURE.md         # This file
├── 🔒 .gitignore                   # Git ignore patterns
│
├── 📁 detection/                   # Object Detection Module
│   ├── __init__.py                # Module exports
│   ├── detector.py                # ObjectDetector class (Grounding DINO)
│   └── visualizer.py              # DetectionVisualizer class
│
├── 📁 tracking/                    # Person Tracking & ReID Module
│   ├── __init__.py                # Module exports
│   └── feature_extractor.py      # FeatureExtractor class (ResNet50)
│
├── 📁 utils/                       # Utility Functions
│   ├── __init__.py                # Module exports
│   ├── device.py                  # Device selection (CUDA/MPS/CPU)
│   └── image_loader.py            # Image loading (URL/local paths)
│
└── 📁 examples/                    # Example Scripts
    ├── __init__.py                # Module marker
    ├── detection_example.py       # Object detection examples
    ├── reid_example.py            # Person ReID comparison (MAIN)
    └── reid_workflow.py           # Full detect→track→compare workflow
```

## Module Descriptions

### 🎯 Detection Module (`detection/`)

**Purpose:** Object detection using Grounding DINO zero-shot detector

**Files:**
- `detector.py` - `ObjectDetector` class for running inference
- `visualizer.py` - `DetectionVisualizer` class for drawing bounding boxes

**Key Features:**
- Zero-shot detection (no training needed)
- Custom text labels
- Performance timing
- Batch processing support

---

### 🧠 Tracking Module (`tracking/`)

**Purpose:** Person re-identification and feature extraction

**Files:**
- `feature_extractor.py` - ResNet50-based feature extraction

**Key Classes & Functions:**
- `FeatureExtractor` - Main class for extracting 2048-dim features
- `get_embedding()` - Simple function to extract features from crop
- `compare_embeddings()` - Compare two embeddings with interpretation
- `interpret_similarity()` - Convert score to human-readable result
- `extract_crop_features()` - Extract features from bbox region

**Key Features:**
- ResNet50 backbone (pretrained on ImageNet)
- L2-normalized embeddings
- Cosine similarity comparison
- Batch processing
- Gallery search

---

### 🛠️ Utils Module (`utils/`)

**Purpose:** Shared utility functions

**Files:**
- `device.py` - Auto-detect best device (CUDA/MPS/CPU)
- `image_loader.py` - Load images from URLs or local paths

---

### 📚 Examples (`examples/`)

**Purpose:** Demonstration scripts and usage examples

**Files:**

1. **`reid_example.py`** ⭐ **MAIN SCRIPT**
   - Compare 2 images to check if same person
   - Batch comparison (1 vs many)
   - Command-line interface
   - Similarity interpretation

2. **`detection_example.py`**
   - Object detection from local files
   - Object detection from URLs
   - Custom thresholds

3. **`reid_workflow.py`**
   - Complete pipeline: Detect → Crop → Extract → Compare
   - Shows full ReID workflow

---

## Quick Usage Reference

### Object Detection
```bash
python main.py --image photo.jpg --labels "person" "car"
```

### Person Re-Identification
```bash
python examples/reid_example.py --image1 person1.jpg --image2 person2.jpg
```

### Programmatic
```python
from detection import ObjectDetector
from tracking import FeatureExtractor
from utils import get_device, load_image

# Detection
detector = ObjectDetector(device=get_device())
results, time = detector.detect(image, ["person"])

# ReID
extractor = FeatureExtractor()
similarity = extractor.compute_similarity(features1, features2)
```

---

## Import Structure

```python
# From project root:
from detection import ObjectDetector, DetectionVisualizer
from tracking import FeatureExtractor
from utils import get_device, load_image
import config

# From examples folder (adds parent to path):
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
```

---

## Data Flow

### Detection Pipeline
```
Image → ObjectDetector → Results (boxes, scores, labels) → DetectionVisualizer → Display
```

### ReID Pipeline
```
Image 1 → FeatureExtractor → Features 1 ↘
                                          → Cosine Similarity → Score → Interpretation
Image 2 → FeatureExtractor → Features 2 ↗
```

### Full Tracking Pipeline
```
Image → ObjectDetector → Bounding Boxes → Crop Images → FeatureExtractor → Compare → Match/No Match
```

---

## Configuration

All default settings in `config.py`:
- Model IDs
- Detection thresholds
- Visualization settings (colors, fonts)
- Default labels

---

## Dependencies

Core libraries (see `requirements.txt`):
- `torch` - Deep learning framework
- `torchvision` - ResNet50 model
- `transformers` - Grounding DINO model
- `matplotlib` - Visualization
- `Pillow` - Image processing
- `requests` - URL loading
