"""Feature extraction for person re-identification using ResNet50."""

import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torch.nn.functional as F
import ssl


class FeatureExtractor:
    """Extract feature embeddings for person re-identification."""
    
    def __init__(self, device=None):
        """
        Initialize the feature extractor with ResNet50.
        
        Args:
            device (str, optional): Device to run on. Auto-detects if None.
        """
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Initializing feature extractor on device: {self.device}")
        
        # Fix SSL certificate verification for model download
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except:
            pass
        
        # Load ResNet50 with modern weights API
        print("Loading ResNet50 pretrained weights...")
        try:
            # Use the new weights API
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except Exception as e:
            print(f"Warning: Failed to load with weights API, trying alternative method: {e}")
            # Fallback to old method
            self.model = resnet50(pretrained=True)
        
        self.model.fc = torch.nn.Identity()  # Remove classification head
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
        print("Feature extractor ready")
    
    def extract_features(self, image):
        """
        Extract feature embedding from a single image.
        
        Args:
            image (PIL.Image or str): Input image or path to image
            
        Returns:
            torch.Tensor: Feature embedding vector (2048-dim for ResNet50)
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL.Image or file path")
        
        # Preprocess and extract features
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img_tensor)
        
        # Normalize features (L2 normalization)
        features = F.normalize(features, p=2, dim=1)
        
        return features.squeeze(0)
    
    def extract_batch_features(self, images):
        """
        Extract features from a batch of images.
        
        Args:
            images (list): List of PIL.Image objects or paths
            
        Returns:
            torch.Tensor: Batch of feature embeddings
        """
        batch_tensors = []
        
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            batch_tensors.append(self.transform(img))
        
        batch = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def compute_similarity(self, features1, features2, interpret=False):
        """
        Compute cosine similarity between two feature vectors.
        
        Args:
            features1 (torch.Tensor): First feature vector
            features2 (torch.Tensor): Second feature vector
            interpret (bool): If True, return (score, interpretation) tuple
            
        Returns:
            float or tuple: 
                - If interpret=False: Similarity score (0-1, higher is more similar)
                - If interpret=True: (score, interpretation_string)
        """
        if features1.dim() == 1:
            features1 = features1.unsqueeze(0)
        if features2.dim() == 1:
            features2 = features2.unsqueeze(0)
        
        similarity = torch.cosine_similarity(features1, features2)
        score = similarity.item()
        
        if interpret:
            interpretation = interpret_similarity(score)
            return score, interpretation
        
        return score
    
    def find_most_similar(self, query_features, gallery_features):
        """
        Find the most similar image in a gallery.
        
        Args:
            query_features (torch.Tensor): Query feature vector
            gallery_features (torch.Tensor): Gallery feature vectors (N x D)
            
        Returns:
            tuple: (best_match_idx, similarity_score)
        """
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)
        
        # Compute similarities with all gallery images
        similarities = F.cosine_similarity(
            query_features,
            gallery_features,
            dim=1
        )
        
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        
        return best_idx, best_score


def get_embedding(crop_img, transform, model, device):
    """
    Extract normalized embedding from a cropped image.
    
    This is a simplified function matching the Step 3 workflow.
    
    Args:
        crop_img (PIL.Image): Cropped person image
        transform: torchvision transform pipeline
        model: Feature extraction model (ResNet50)
        device (str): Device to run on ('cuda', 'mps', 'cpu')
        
    Returns:
        torch.Tensor: Normalized feature embedding
    """
    img_tensor = transform(crop_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(img_tensor)
    feat = F.normalize(feat, dim=1)
    return feat


def compare_embeddings(embed1, embed2):
    """
    Compare two embeddings and return similarity score with interpretation.
    
    Args:
        embed1 (torch.Tensor): First embedding
        embed2 (torch.Tensor): Second embedding
        
    Returns:
        tuple: (similarity_score, interpretation)
            - similarity_score (float): Cosine similarity (0-1)
            - interpretation (str): Human-readable interpretation
    """
    similarity = torch.cosine_similarity(embed1, embed2)
    score = similarity.item()
    
    # Interpret similarity
    if score > 0.8:
        interpretation = "Very likely same person"
    elif score >= 0.6:
        interpretation = "Possible match"
    else:
        interpretation = "Probably different person"
    
    return score, interpretation


def interpret_similarity(score):
    """
    Interpret a similarity score.
    
    Args:
        score (float): Similarity score (0-1)
        
    Returns:
        str: Interpretation of the similarity
        
    Similarity Thresholds:
        > 0.8:     Very likely same person
        0.6-0.8:   Possible match
        < 0.6:     Probably different person
        
    Note: These thresholds should be tuned based on your specific use case.
    """
    if score > 0.8:
        return "✓ Very likely same person"
    elif score >= 0.6:
        return "~ Possible match (tune threshold)"
    else:
        return "✗ Probably different person"


def extract_crop_features(image, bbox, feature_extractor):
    """
    Extract features from a cropped region of an image.
    
    Args:
        image (PIL.Image): Full image
        bbox (list or tuple): Bounding box [xmin, ymin, xmax, ymax]
        feature_extractor (FeatureExtractor): Feature extractor instance
        
    Returns:
        torch.Tensor: Feature embedding of the cropped region
    """
    xmin, ymin, xmax, ymax = bbox
    cropped = image.crop((xmin, ymin, xmax, ymax))
    return feature_extractor.extract_features(cropped)
