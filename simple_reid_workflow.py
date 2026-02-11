"""
Simple ReID workflow following the 4-step process:
1. Detect with DINO
2. Crop detections
3. Extract embeddings
4. Compare embeddings
"""

import torch
import ssl
from PIL import Image

from utils import get_device, load_image
from detector import ObjectDetector
from feature_extractor import (
    get_embedding,
    compare_embeddings,
    interpret_similarity
)
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights


def simple_reid_workflow():
    """Complete ReID workflow: Detect → Crop → Embed → Compare"""
    
    print("=" * 70)
    print("Simple Person Re-Identification Workflow")
    print("=" * 70)
    
    # Setup
    device = get_device()
    
    # Step 0: Load models
    print("\n[Step 0] Loading models...")
    
    # DINO detector
    detector = ObjectDetector(device=device)
    
    # Feature extractor
    # Fix SSL certificate verification
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except:
        pass
    
    print("Loading ResNet50 pretrained weights...")
    try:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except:
        model = resnet50(pretrained=True)
    
    model.fc = torch.nn.Identity()  # Remove classification head
    model = model.to(device)
    model.eval()
    
    transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])
    
    print("✓ Models loaded")
    
    # Load test image
    image_path = "/Users/psinha/Desktop/test_images/test_image_2.jpg"
    image = load_image(image_path)
    print(f"\n✓ Loaded image: {image_path}")
    print(f"  Image size: {image.size}")
    
    # Step 1: Detect with DINO
    print("\n[Step 1] 🎯 Detecting persons with DINO...")
    labels = ["a person"]
    results, inference_time = detector.detect(image, labels, threshold=0.4)
    
    print(f"✓ Detected {len(results['boxes'])} persons in {inference_time:.3f}s")
    
    if len(results['boxes']) == 0:
        print("No persons detected. Try different labels or lower threshold.")
        return
    
    # Step 2: Crop detections
    print("\n[Step 2] ✂️  Cropping detections...")
    crops = []
    for idx, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
        if score >= 0.4:
            xmin, ymin, xmax, ymax = box.tolist()
            crop = image.crop((xmin, ymin, xmax, ymax))
            crops.append({
                'id': idx,
                'crop': crop,
                'bbox': [xmin, ymin, xmax, ymax],
                'score': score.item()
            })
            print(f"  Person {idx}: bbox={[int(x) for x in [xmin, ymin, xmax, ymax]]}, confidence={score:.3f}")
    
    print(f"✓ Created {len(crops)} crops")
    
    # Step 3: Extract embeddings from crops
    print("\n[Step 3] 🧠 Extracting embeddings from crops...")
    embeddings = []
    for crop_data in crops:
        embedding = get_embedding(crop_data['crop'], transform, model, device)
        embeddings.append(embedding)
        crop_data['embedding'] = embedding
        print(f"  Person {crop_data['id']}: embedding shape {embedding.shape}")
    
    print(f"✓ Extracted {len(embeddings)} embeddings")
    
    # Step 4: Compare embeddings
    if len(embeddings) >= 2:
        print("\n[Step 4] 📊 Comparing embeddings...")
        print("\nSimilarity Matrix:")
        print("-" * 70)
        print(f"{'Pair':<15} {'Similarity':<12} {'Interpretation'}")
        print("-" * 70)
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity, interpretation = compare_embeddings(
                    embeddings[i],
                    embeddings[j]
                )
                print(f"Person {i} vs {j}   {similarity:.4f}       {interpretation}")
        
        print("-" * 70)
        print("\n🧠 Interpreting Similarity Scores:")
        print("  > 0.8     Very likely same person")
        print("  0.6-0.8   Possible match")
        print("  < 0.6     Probably different person")
        print("\n💡 Note: Tune these thresholds based on your use case")
    else:
        print("\n[Step 4] Only one person detected, skipping comparison")
        print("  Try an image with multiple persons to see similarity comparison")
    
    print("\n" + "=" * 70)
    print("Workflow Complete!")
    print("=" * 70)


def quick_compare_two_images():
    """
    Quick comparison: Compare a person across two different images.
    """
    print("\n" + "=" * 70)
    print("Quick Compare: Same person in different images?")
    print("=" * 70)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Fix SSL certificate verification
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except:
        pass
    
    # Load model
    try:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except:
        model = resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    
    transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])
    
    # Load two person crops (or full images)
    img1_path = "/Users/psinha/Desktop/test_images/person1.jpg"
    img2_path = "/Users/psinha/Desktop/test_images/person2.jpg"
    
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        print(f"\nImage 1: {img1_path}")
        print(f"Image 2: {img2_path}")
        
        # Extract embeddings
        embed1 = get_embedding(img1, transform, model, device)
        embed2 = get_embedding(img2, transform, model, device)
        
        # Compare
        similarity, interpretation = compare_embeddings(embed1, embed2)
        
        print(f"\n{'Similarity:':<15} {similarity:.4f}")
        print(f"{'Interpretation:':<15} {interpretation}")
        print(f"\n{interpret_similarity(similarity)}")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  Images not found: {e}")
        print("Update the paths in the script to use your own images")


if __name__ == "__main__":
    # Run the complete workflow
    simple_reid_workflow()
    
    # Uncomment to run quick comparison:
    # quick_compare_two_images()
