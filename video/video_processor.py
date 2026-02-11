"""Video frame processing utilities."""

import cv2
from PIL import Image
from pathlib import Path


class VideoProcessor:
    """Process video files frame by frame."""
    
    def __init__(self, video_path, detection_interval=20):
        """
        Initialize video processor.
        
        Args:
            video_path (str): Path to video file
            detection_interval (int): Run detection every N frames (default: 20)
        """
        self.video_path = video_path
        self.detection_interval = detection_interval
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.frame_id = 0
        
        print(f"Video loaded: {Path(video_path).name}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Duration: {self.total_frames/self.fps:.2f}s")
        print(f"  Detection interval: every {detection_interval} frames")
    
    def read_frame(self):
        """
        Read next frame from video.
        
        Returns:
            tuple: (frame_id, frame_bgr, frame_pil) or (None, None, None) if end
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None
        
        self.frame_id += 1
        
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        return self.frame_id, frame, frame_pil
    
    def should_detect(self):
        """Check if detection should run on current frame."""
        return self.frame_id % self.detection_interval == 0
    
    def release(self):
        """Release video capture."""
        self.cap.release()
        cv2.destroyAllWindows()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def get_writer(self, output_path, codec='mp4v'):
        """
        Create video writer for output.
        
        Args:
            output_path (str): Output video path
            codec (str): Video codec (default: 'mp4v')
            
        Returns:
            cv2.VideoWriter: Video writer object
        """
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        return writer
    
    @staticmethod
    def draw_bbox(frame, bbox, label, color=(0, 255, 0), thickness=2):
        """
        Draw bounding box on frame.
        
        Args:
            frame: OpenCV frame (BGR)
            bbox: [xmin, ymin, xmax, ymax]
            label: Text label
            color: BGR color tuple
            thickness: Line thickness
        """
        xmin, ymin, xmax, ymax = map(int, bbox)
        
        # Draw rectangle
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_ymin = max(ymin, label_size[1] + 10)
        cv2.rectangle(
            frame,
            (xmin, label_ymin - label_size[1] - 10),
            (xmin + label_size[0], label_ymin),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (xmin, label_ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
    
    @staticmethod
    def show_frame(frame, window_name="Video"):
        """Display frame in window."""
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1) & 0xFF
