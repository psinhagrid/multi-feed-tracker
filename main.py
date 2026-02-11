"""Main script for object detection with Grounding DINO."""

import argparse
from pathlib import Path

from utils import get_device, load_image
from detection import ObjectDetector, DetectionVisualizer
import config


def main(
    image_source,
    labels,
    threshold=None,
    text_threshold=None,
    save_output=None,
    no_display=False
):
    """
    Run object detection on an image.
    
    Args:
        image_source (str): Path or URL to the image
        labels (list): List of text labels to detect
        threshold (float, optional): Detection confidence threshold
        text_threshold (float, optional): Text matching threshold
        save_output (str, optional): Path to save the visualization
        no_display (bool): If True, don't display the image
    """
    # Get device
    device = get_device()
    
    # Load image
    print(f"\nLoading image from: {image_source}")
    image = load_image(image_source)
    print(f"Image size: {image.size}")
    
    # Initialize detector
    detector = ObjectDetector(device=device)
    
    # Run detection
    print(f"\nDetecting objects with labels: {labels}")
    results, inference_time = detector.detect(
        image,
        labels,
        threshold=threshold,
        text_threshold=text_threshold
    )
    
    # Print results
    detector.print_results(results, inference_time)
    
    # Visualize results
    if not no_display or save_output:
        visualizer = DetectionVisualizer()
        visualizer.draw_boxes(
            image,
            results,
            min_confidence=threshold or config.DETECTION_THRESHOLD,
            save_path=save_output
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Object detection using Grounding DINO"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path or URL to the input image"
    )
    
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=config.DEFAULT_LABELS,
        help="List of object labels to detect"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.DETECTION_THRESHOLD,
        help="Detection confidence threshold (0-1)"
    )
    
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=config.TEXT_THRESHOLD,
        help="Text matching threshold (0-1)"
    )
    
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the output visualization"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display the visualization window"
    )
    
    args = parser.parse_args()
    
    main(
        image_source=args.image,
        labels=args.labels,
        threshold=args.threshold,
        text_threshold=args.text_threshold,
        save_output=args.save,
        no_display=args.no_display
    )
