"""Visualization utilities for detection results."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import config


class DetectionVisualizer:
    """Class for visualizing object detection results."""
    
    def __init__(
        self,
        box_color=None,
        box_linewidth=None,
        text_fontsize=None,
        text_color=None,
        text_background=None
    ):
        """
        Initialize the visualizer with custom settings.
        
        Args:
            box_color (str, optional): Color for bounding boxes
            box_linewidth (int, optional): Line width for boxes
            text_fontsize (int, optional): Font size for labels
            text_color (str, optional): Color for text
            text_background (str, optional): Background color for text
        """
        self.box_color = box_color or config.BOX_COLOR
        self.box_linewidth = box_linewidth or config.BOX_LINEWIDTH
        self.text_fontsize = text_fontsize or config.TEXT_FONTSIZE
        self.text_color = text_color or config.TEXT_COLOR
        self.text_background = text_background or config.TEXT_BACKGROUND
    
    def draw_boxes(self, image, results, min_confidence=None, save_path=None):
        """
        Draw bounding boxes on the image.
        
        Args:
            image (PIL.Image): Input image
            results (dict): Detection results with 'boxes', 'scores', 'labels'
            min_confidence (float, optional): Minimum confidence to display
            save_path (str, optional): Path to save the output image
        """
        min_confidence = min_confidence or config.DETECTION_THRESHOLD
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["labels"]
        ):
            if score < min_confidence:
                continue
            
            xmin, ymin, xmax, ymax = box.tolist()
            
            # Draw rectangle
            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=self.box_linewidth,
                edgecolor=self.box_color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(
                xmin,
                ymin - 5,
                f"{label}: {score:.2f}",
                color=self.text_color,
                fontsize=self.text_fontsize,
                backgroundcolor=self.text_background
            )
        
        plt.axis("off")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to: {save_path}")
        
        plt.show()
