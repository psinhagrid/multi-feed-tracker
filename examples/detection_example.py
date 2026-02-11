"""Example script showing how to use the object detection library."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils import get_device, load_image
from detection import ObjectDetector, DetectionVisualizer


def example_local_image():
    """Example: Detect objects in a local image."""
    # Setup
    device = get_device()
    detector = ObjectDetector(device=device)
    visualizer = DetectionVisualizer()
    
    # Load image
    image_path = "/Users/psinha/Desktop/test_images/test_image_2.jpg"
    image = load_image(image_path)
    
    # Detect objects
    labels = ["Green SweaterLady"]
    results, inference_time = detector.detect(image, labels)
    
    # Print and visualize results
    detector.print_results(results, inference_time)
    visualizer.draw_boxes(image, results)


def example_url_image():
    """Example: Detect objects from an image URL."""
    # Setup
    device = get_device()
    detector = ObjectDetector(device=device)
    visualizer = DetectionVisualizer()
    
    # Load image from URL
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = load_image(image_url)
    
    # Detect objects
    labels = ["a cat", "a remote control"]
    results, inference_time = detector.detect(image, labels)
    
    # Print and visualize results
    detector.print_results(results, inference_time)
    visualizer.draw_boxes(image, results)


def example_custom_thresholds():
    """Example: Use custom detection thresholds."""
    device = get_device()
    detector = ObjectDetector(device=device)
    visualizer = DetectionVisualizer()
    
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = load_image(image_url)
    
    # Use higher confidence threshold
    labels = ["a cat", "a remote control"]
    results, inference_time = detector.detect(
        image,
        labels,
        threshold=0.5,  # Higher threshold
        text_threshold=0.4
    )
    
    detector.print_results(results, inference_time)
    visualizer.draw_boxes(image, results, min_confidence=0.5)


if __name__ == "__main__":
    print("=" * 70)
    print("Example 1: Local Image Detection")
    print("=" * 70)
    example_local_image()
    
    # print("\n" + "=" * 70)
    # print("Example 2: URL Image Detection")
    # print("=" * 70)
    # example_url_image()
