"""Object detection model wrapper."""

import time
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import config


class ObjectDetector:
    """Wrapper class for Grounding DINO object detection model."""
    
    def __init__(self, model_id=None, device=None):
        """
        Initialize the object detector.
        
        Args:
            model_id (str, optional): Model identifier. Defaults to config.MODEL_ID
            device (str, optional): Device to run model on. Defaults to None (auto-detect)
        """
        self.model_id = model_id or config.MODEL_ID
        self.device = device
        
        print(f"Loading model: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)
        print("Model loaded successfully")
    
    def detect(self, image, text_labels, threshold=None, text_threshold=None):
        """
        Perform object detection on an image.
        
        Args:
            image (PIL.Image): Input image
            text_labels (list): List of text labels to detect
            threshold (float, optional): Detection confidence threshold
            text_threshold (float, optional): Text matching threshold
            
        Returns:
            tuple: (results_dict, inference_time)
                - results_dict: Dictionary with 'boxes', 'scores', 'labels'
                - inference_time: Time taken for inference in seconds
        """
        threshold = threshold or config.DETECTION_THRESHOLD
        text_threshold = text_threshold or config.TEXT_THRESHOLD
        
        # Prepare inputs
        inputs = self.processor(
            images=image,
            text=text_labels,
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference with timing
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )
        
        inference_time = time.time() - start_time
        
        return results[0], inference_time
    
    def print_results(self, results, inference_time):
        """
        Print detection results in a formatted way.
        
        Args:
            results (dict): Detection results dictionary
            inference_time (float): Inference time in seconds
        """
        print(f"\nInference time: {inference_time:.3f} seconds ({inference_time*1000:.1f} ms)")
        print(f"\nDetected {len(results['boxes'])} objects:")
        print("-" * 70)
        
        for box, score, label in zip(
            results["boxes"], 
            results["scores"], 
            results["labels"]
        ):
            box_coords = [round(x, 2) for x in box.tolist()]
            confidence = round(score.item(), 3)
            print(f"  {label:20s} | Confidence: {confidence:.3f} | Box: {box_coords}")
