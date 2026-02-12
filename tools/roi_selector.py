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
                    if api_key:
                        print(f"\n  Sending to LLM for description...")
                        try:
                            label = describe_image(str(filepath), api_key)
                            print(f"  🏷️  LLM Label: '{label}'")
                        except Exception as e:
                            print(f"  ✗ LLM error: {e}")
                    else:
                        print(f"  ⚠️  ANTHROPIC_API_KEY not found in .env - skipping LLM description")
                    
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
