# Multi-Feed Object Tracker

A clean, modular object detection system using Grounding DINO for zero-shot object detection.

## Features

- 🚀 Zero-shot object detection with Grounding DINO
- 🎯 Support for custom text labels
- 💻 GPU acceleration (CUDA, MPS, or CPU)
- 📊 Visual bounding box overlay
- ⚡ Performance timing
- 🔧 Configurable thresholds and visualization settings

## Project Structure

```
Multi-Feed_Tracker/
├── config.py              # Configuration settings
├── detector.py            # Object detection model wrapper
├── visualizer.py          # Visualization utilities
├── main.py               # Main CLI script
├── example.py            # Usage examples
├── requirements.txt      # Python dependencies
├── utils/
│   ├── __init__.py
│   ├── device.py         # Device selection utilities
│   └── image_loader.py   # Image loading utilities
└── README.md
```

## Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Basic usage:
```bash
python main.py --image path/to/image.jpg --labels "a person" "a car"
```

With custom thresholds:
```bash
python main.py --image path/to/image.jpg \
               --labels "a cat" "a dog" \
               --threshold 0.5 \
               --text-threshold 0.4
```

Save output without displaying:
```bash
python main.py --image path/to/image.jpg \
               --labels "a person" \
               --save output.jpg \
               --no-display
```

From URL:
```bash
python main.py --image "http://example.com/image.jpg" \
               --labels "a car"
```

### Programmatic Usage

```python
from utils import get_device, load_image
from detector import ObjectDetector
from visualizer import DetectionVisualizer

# Initialize
device = get_device()
detector = ObjectDetector(device=device)
visualizer = DetectionVisualizer()

# Load and detect
image = load_image("path/to/image.jpg")
results, inference_time = detector.detect(image, ["a cat", "a dog"])

# Display results
detector.print_results(results, inference_time)
visualizer.draw_boxes(image, results)
```

See `example.py` for more usage examples.

## Configuration

Edit `config.py` to customize:
- Model selection
- Detection thresholds
- Visualization colors and styles
- Default labels

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- Pillow
- matplotlib
- requests

## Performance

- Supports CUDA (NVIDIA GPU), MPS (Apple Silicon), and CPU
- Inference timing included in output
- Model: Grounding DINO Tiny (lightweight and fast)

## License

MIT License
