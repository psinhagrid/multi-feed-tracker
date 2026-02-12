#!/usr/bin/env python3
"""
ROI Selector - Interactive region selection and cropping from video
Allows you to pause video, select a region, and save the cropped area.
"""

import cv2
import sys
from pathlib import Path
import argparse
from datetime import datetime
import os
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Import LLM describer
from llm.image_describer import describe_image

# Import ByteTrack for pure tracking
sys.path.append(str(Path(__file__).parent.parent / "tracker" / "bytetrack"))
from yolox.tracker.byte_tracker import BYTETracker, STrack
import numpy as np
import config


def track_selected_roi(video_path, initial_bbox, start_frame, label):
    """
    Track a selected ROI using OpenCV CSRT tracker (no detection model).
    Pure visual tracking based on appearance features.
    
    Args:
        video_path (str): Path to video file
        initial_bbox (tuple): Initial bounding box (x, y, w, h)
        start_frame (int): Frame number where ROI was selected
        label (str): Label name for the tracked object (from LLM)
    """
    print(f"\n{'='*70}")
    print(f"Starting OpenCV CSRT Tracker (Pure Visual Tracking - No Detection)")
    print(f"{'='*70}")
    print(f"Video: {video_path}")
    print(f"Label: '{label}' (from LLM)")
    print(f"Initial ROI: x={initial_bbox[0]}, y={initial_bbox[1]}, w={initial_bbox[2]}, h={initial_bbox[3]}")
    print(f"Start frame: {start_frame}")
    print(f"\nInitializing CSRT tracker...")
    print(f"Press 'q' to quit\n")
    
    # Open video
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
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read start frame")
        cap.release()
        return
    
    # Initialize CSRT tracker (try new API first, fallback to legacy)
    try:
        tracker = cv2.TrackerCSRT.create()
    except AttributeError:
        try:
            tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            print("Error: OpenCV tracking not available. Install opencv-contrib-python:")
            print("  pip uninstall opencv-python")
            print("  pip install opencv-contrib-python")
            cap.release()
            return
    
    tracker.init(frame, initial_bbox)
    print(f"[Frame {start_frame}] CSRT tracker initialized")
    
    frame_delay = int(1000 / fps) if fps > 0 else 30
    frame_id = start_frame
    tracking_success = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video")
            break
        
        frame_id += 1
        
        # Update tracker
        tracking_success, bbox = tracker.update(frame)
        
        # Draw bounding box
        display_frame = frame.copy()
        
        if tracking_success:
            # Bounding box in (x, y, w, h) format
            x, y, w, h = [int(v) for v in bbox]
            
            # Draw green box for successful tracking
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{label}"
            cv2.putText(display_frame, label_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw tracking status
            status_text = "Tracking: OK"
            cv2.putText(display_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Show "Lost" message
            status_text = f"{label} - TRACKING LOST"
            cv2.putText(display_frame, status_text, (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            print(f"[Frame {frame_id}] Tracking lost")
        
        # Display frame info
        info_text = f"Frame: {frame_id}/{total_frames} | CSRT Tracker | Press 'q' to quit"
        cv2.putText(display_frame, info_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(f"CSRT Tracker - {label}", display_frame)
        
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            print("\nStopped by user")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTracking complete. Processed up to frame {frame_id}/{total_frames}")


def select_and_crop_roi(video_path, output_dir="crops"):
    """
    Play video and allow user to select regions of interest (ROI) to crop and save.
    
    Args:
        video_path (str): Path to video file
        output_dir (str): Directory to save cropped images
    
    Controls:
        - SPACE: Pause video and select ROI
        - 'q': Quit video
        - ESC: Cancel ROI selection
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
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
    print("\nControls:")
    print("  SPACE - Pause and select ROI")
    print("  'q'   - Quit")
    print("  ESC   - Cancel ROI selection")
    print("\nPlaying video...\n")
    
    frame_delay = int(1000 / fps) if fps > 0 else 30
    frame_count = 0
    paused = False
    current_frame = None
    crop_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video")
                break
            
            frame_count += 1
            current_frame = frame.copy()
            
            # Display frame info
            info_text = f"Frame: {frame_count}/{total_frames} | Press SPACE to select ROI | 'q' to quit"
            cv2.putText(current_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Video - ROI Selector", current_frame)
            key = cv2.waitKey(frame_delay) & 0xFF
        else:
            # Paused mode
            cv2.imshow("Video - ROI Selector", current_frame)
            key = cv2.waitKey(1) & 0xFF
        
        # Handle keyboard input
        if key == ord('q'):
            print("\nStopped by user")
            break
        elif key == ord(' '):  # Space bar
            if not paused:
                # Pause and prepare for ROI selection
                paused = True
                print(f"\n[Frame {frame_count}] Video paused. Select ROI...")
                print("  → Drag to select region")
                print("  → Press ENTER to confirm")
                print("  → Press ESC to cancel\n")
                
                # Let user select ROI
                roi = cv2.selectROI("Video - ROI Selector", current_frame, 
                                   fromCenter=False, showCrosshair=True)
                x, y, w, h = roi
                
                # Check if valid selection (w and h > 0)
                if w > 0 and h > 0:
                    # Crop the selected region
                    cropped = current_frame[int(y):int(y+h), int(x):int(x+w)]
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"crop_frame{frame_count}_{timestamp}.jpg"
                    filepath = output_path / filename
                    
                    # Save cropped image
                    cv2.imwrite(str(filepath), cropped)
                    crop_count += 1
                    
                    print(f"✓ Saved crop #{crop_count}: {filepath}")
                    print(f"  Region: x={x}, y={y}, w={w}, h={h}")
                    print(f"  Size: {cropped.shape[1]}x{cropped.shape[0]} pixels")
                    
                    # Send to LLM for description
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                    label = None
                    if api_key:
                        print(f"\n  Sending to LLM for description...")
                        try:
                            label = describe_image(str(filepath), api_key)
                            print(f"  🏷️  LLM Label: '{label}'")
                        except Exception as e:
                            print(f"  ✗ LLM error: {e}")
                    else:
                        print(f"  ⚠️  ANTHROPIC_API_KEY not found in .env - skipping LLM description")
                    
                    # Automatically start tracking with the label
                    if label:
                        print(f"\n{'='*60}")
                        print(f"  ROI selected and labeled: '{label}'")
                        print(f"  Starting tracking automatically...")
                        print(f"{'='*60}")
                        
                        # Close current video window
                        cap.release()
                        cv2.destroyAllWindows()
                        
                        # Small delay to ensure window closes
                        import time
                        time.sleep(0.3)
                        
                        # Run hybrid detection + tracking on the selected ROI
                        track_selected_roi(
                            video_path=video_path,
                            initial_bbox=(x, y, w, h),
                            start_frame=frame_count,
                            label=label
                        )
                        
                        # Exit after tracking completes
                        return
                    
                    print("\nPress SPACE again to select another ROI, or any key to resume video\n")
                else:
                    print("✗ No region selected (cancelled)")
                    print("\nPress SPACE to try again, or any key to resume video\n")
                
                # Stay paused for another selection or resume
                paused = True
            else:
                # Already paused, resume playback
                paused = False
                print("Resuming video...\n")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nSession complete:")
    print(f"  Total crops saved: {crop_count}")
    print(f"  Output directory: {output_path.absolute()}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Interactive ROI selector for video - select and crop regions"
    )
    
    parser.add_argument(
        "video_path",
        nargs='?',
        help="Path to video file (will prompt if not provided)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        default="crops",
        help="Output directory for cropped images (default: crops)"
    )
    
    args = parser.parse_args()
    
    # Get video path
    if args.video_path:
        video_path = args.video_path
    else:
        # Interactive mode - prompt for video path
        print("=" * 70)
        print("ROI Selector - Interactive Mode")
        print("=" * 70)
        video_path = input("\nEnter video path: ").strip()
        if not video_path:
            print("Error: Video path is required")
            return
    
    # Run ROI selector
    select_and_crop_roi(video_path, args.output)


if __name__ == "__main__":
    main()
