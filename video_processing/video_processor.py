"""Video frame processing with detection and tracking."""

import cv2
from PIL import Image
import sys
from pathlib import Path
import argparse
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "tracker" / "bytetrack"))

from detection import ObjectDetector
from utils import get_device
import config

# Import ByteTrack
from yolox.tracker.byte_tracker import BYTETracker


def process_video(video_path, labels, detection_interval=None, show_display=True, use_tracking=True):
    """
    Process video with object detection and tracking.
    
    Args:
        video_path (str): Path to video file
        labels (list): List of detection labels (e.g., ["a person", "a car"])
        detection_interval (int, optional): Run detection every N frames
        show_display (bool): Whether to show video window
        use_tracking (bool): Whether to use ByteTrack for tracking
    """
    # Use config default if not specified
    if detection_interval is None:
        detection_interval = config.DETECTION_FRAME_INTERVAL
    
    # Initialize detector
    print(f"Initializing detector...")
    device = get_device()
    detector = ObjectDetector(device=device)
    
    # Initialize ByteTrack tracker
    if use_tracking:
        print(f"Initializing ByteTrack tracker...")
        # ByteTrack args (tuned for DINO detections)
        class TrackArgs:
            track_thresh = 0.4  # Lowered to match DINO detection threshold
            track_buffer = 30
            match_thresh = 0.8
            mot20 = False
        
        tracker = BYTETracker(TrackArgs(), frame_rate=30)
        print("Tracker initialized (track_thresh=0.4)")
    
    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    print(f"Detection labels: {labels}")
    print(f"Detection interval: every {detection_interval} frames")
    print(f"Tracking: {'Enabled' if use_tracking else 'Disabled'}")
    print("\nProcessing video... (Press 'q' to quit)\n")
    
    frame_id = 0
    current_results = None  # Store latest detection results
    online_targets = []  # Store tracked objects
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Resize frame for faster detection
        h, w, _ = frame.shape
        new_width = 640
        new_height = int(h * (640 / w))
        frame = cv2.resize(frame, (new_width, new_height))

        # Convert to PIL for DINO
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Detection every N frames
        if frame_id % detection_interval == 0:
            print(f"[Frame {frame_id}/{total_frames}] Running detection...")
            current_results, inference_time = detector.detect(image, labels)
            num_detections = len(current_results['boxes'])
            print(f"  → Detected {num_detections} objects in {inference_time:.3f}s")
            
            # Print detection details
            for idx, (box, score, label) in enumerate(zip(
                current_results['boxes'], 
                current_results['scores'], 
                current_results['labels']
            )):
                print(f"    {idx+1}. {label}: {score:.3f}")
            
            # Convert detections to ByteTrack format
            if use_tracking and num_detections > 0:
                detections = []
                for box, score in zip(current_results['boxes'], current_results['scores']):
                    x1, y1, x2, y2 = box.tolist()
                    detections.append([x1, y1, x2, y2, score.item()])
                
                detections = np.array(detections)
                
                # Update tracker
                online_targets = tracker.update(detections, [height, width], [height, width])
                print(f"  → Tracking {len(online_targets)} objects")
        
        # Draw bounding boxes on frame
        if use_tracking and len(online_targets) > 0:
            # Draw tracked objects with IDs
            for track in online_targets:
                tlwh = track.tlwh
                track_id = track.track_id
                
                xmin, ymin, w, h = int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])
                xmax, ymax = xmin + w, ymin + h
                
                # Draw rectangle
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Draw track ID
                label_text = f"ID: {track_id}"
                cv2.putText(frame, label_text, (xmin, ymin - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        elif current_results is not None and len(current_results['boxes']) > 0:
            # Draw detection boxes (when tracking is disabled)
            for box, score, label in zip(
                current_results['boxes'],
                current_results['scores'],
                current_results['labels']
            ):
                if score >= config.DETECTION_THRESHOLD:
                    # Get box coordinates
                    xmin, ymin, xmax, ymax = box.tolist()
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    
                    # Draw label
                    label_text = f"{label}: {score:.2f}"
                    cv2.putText(frame, label_text, (xmin, ymin - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display frame
        if show_display:
            cv2.imshow("Video Processing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break

    # Cleanup
    cap.release()
    if show_display:
        cv2.destroyAllWindows()
    
    print(f"\nProcessing complete. Processed {frame_id} frames.")


def parse_labels(label_input):
    """
    Parse label input - handles comma-separated and space-separated labels.
    
    Args:
        label_input (str or list): Labels as string or list
        
    Returns:
        list: Parsed labels
        
    Examples:
        "person, car, dog" -> ["person", "car", "dog"]
        "green jacket, red hat" -> ["green jacket", "red hat"]
    """
    if isinstance(label_input, list):
        # Already a list (from argparse)
        return label_input
    
    # Strip any surrounding quotes that user might have added
    label_input = label_input.strip().strip("'\"")
    
    # String input - check for commas
    if ',' in label_input:
        # Comma-separated: split by comma and strip whitespace
        labels = [label.strip().strip("'\"") for label in label_input.split(',')]
        return [label for label in labels if label]  # Remove empty strings
    else:
        # Single label
        return [label_input.strip()]


def interactive_mode():
    """Interactive mode - prompt user for inputs."""
    print("=" * 70)
    print("Video Processing - Interactive Mode")
    print("=" * 70)
    
    # Get video path
    video_path = input("\nEnter video path: ").strip()
    if not video_path:
        print("Error: Video path is required")
        return
    
    # Get labels
    print("\nEnter detection labels (without quotes):")
    print("  - Single: Yellow Raincoat person")
    print("  - Multiple (comma-separated): person, car, dog")
    print("  - Multi-word (comma-separated): green jacket, red hat")
    labels_input = input("Labels: ").strip()
    
    if not labels_input:
        print("Error: At least one label is required")
        return
    
    labels = parse_labels(labels_input)
    print(f"\nParsed labels: {labels}")
    
    # Get detection interval
    interval_input = input(f"\nDetection interval (frames, default {config.DETECTION_FRAME_INTERVAL}): ").strip()
    interval = int(interval_input) if interval_input else None
    
    print("\n" + "=" * 70)
    
    # Process video (always show display)
    process_video(
        video_path=video_path,
        labels=labels,
        detection_interval=interval,
        show_display=True
    )


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Process video with object detection"
    )
    
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file"
    )
    
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Detection labels (e.g., 'a person' 'a car')"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help=f"Run detection every N frames (default: {config.DETECTION_FRAME_INTERVAL})"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't show video window (faster processing)"
    )
    
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable ByteTrack tracking (only show detections)"
    )
    
    args = parser.parse_args()
    
    # Check if running in interactive mode (no arguments provided)
    if not args.video and not args.labels:
        interactive_mode()
    elif not args.video or not args.labels:
        print("Error: Both --video and --labels are required")
        print("\nRun without arguments for interactive mode, or use:")
        print("  python video_processor.py --video VIDEO_PATH --labels LABEL1 LABEL2")
        parser.print_help()
    else:
        # Process video with CLI arguments
        process_video(
            video_path=args.video,
            labels=args.labels,
            detection_interval=args.interval,
            show_display=not args.no_display,
            use_tracking=not args.no_tracking
        )


if __name__ == "__main__":
    main()
