"""Example: Person Re-Identification - Compare two images to check if same person."""

import argparse
import torch
from PIL import Image

from feature_extractor import (
    FeatureExtractor,
    get_embedding,
    compare_embeddings,
    interpret_similarity
)


def compare_two_persons(image1_path, image2_path, show_details=True):
    """
    Compare two images to determine if they show the same person.
    
    Args:
        image1_path (str): Path to first image
        image2_path (str): Path to second image
        show_details (bool): Whether to print detailed information
        
    Returns:
        tuple: (similarity_score, interpretation, is_match)
    """
    if show_details:
        print("=" * 70)
        print("Person Re-Identification: Comparing Two Images")
        print("=" * 70)
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Load images
    if show_details:
        print(f"\nImage 1: {image1_path}")
        print(f"Image 2: {image2_path}")
    
    try:
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')
        
        if show_details:
            print(f"  Image 1 size: {img1.size}")
            print(f"  Image 2 size: {img2.size}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not load image - {e}")
        return None, None, False
    
    # Extract features
    if show_details:
        print("\n🧠 Extracting features...")
    
    features1 = extractor.extract_features(img1)
    features2 = extractor.extract_features(img2)
    
    if show_details:
        print(f"  Image 1 features: {features1.shape}")
        print(f"  Image 2 features: {features2.shape}")
    
    # Compute similarity
    if show_details:
        print("\n📊 Computing similarity...")
    
    similarity, interpretation = extractor.compute_similarity(
        features1, 
        features2, 
        interpret=True
    )
    
    # Determine if it's a match (threshold: 0.6)
    is_match = similarity >= 0.6
    
    # Print results
    if show_details:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Similarity Score:  {similarity:.4f}")
        print(f"Interpretation:    {interpretation}")
        print()
        
        if similarity > 0.8:
            print("✅ MATCH: Very likely the same person!")
        elif similarity >= 0.6:
            print("⚠️  POSSIBLE MATCH: Could be the same person")
        else:
            print("❌ NO MATCH: Likely different persons")
        
        print("\n" + "=" * 70)
        print("📊 Similarity Interpretation Guide:")
        print("=" * 70)
        print("  > 0.8     ✅ Very likely same person")
        print("  0.6-0.8   ⚠️  Possible match (tune threshold)")
        print("  < 0.6     ❌ Probably different person")
        print("=" * 70)
    
    return similarity, interpretation, is_match


def batch_compare(reference_image, comparison_images):
    """
    Compare a reference image against multiple images.
    
    Args:
        reference_image (str): Path to reference image
        comparison_images (list): List of image paths to compare against
        
    Returns:
        list: Results for each comparison
    """
    print("=" * 70)
    print("Batch Comparison: Finding Matches")
    print("=" * 70)
    
    extractor = FeatureExtractor()
    
    # Extract reference features
    print(f"\n📷 Reference Image: {reference_image}")
    ref_img = Image.open(reference_image).convert('RGB')
    ref_features = extractor.extract_features(ref_img)
    
    print(f"\n🔍 Comparing against {len(comparison_images)} images...\n")
    
    results = []
    for idx, img_path in enumerate(comparison_images, 1):
        try:
            img = Image.open(img_path).convert('RGB')
            features = extractor.extract_features(img)
            similarity, interpretation = extractor.compute_similarity(
                ref_features,
                features,
                interpret=True
            )
            
            is_match = similarity >= 0.6
            match_indicator = "✅" if similarity > 0.8 else "⚠️" if is_match else "❌"
            
            print(f"  {idx}. {img_path}")
            print(f"     {match_indicator} Similarity: {similarity:.4f} - {interpretation}")
            
            results.append({
                'image': img_path,
                'similarity': similarity,
                'interpretation': interpretation,
                'is_match': is_match
            })
            
        except Exception as e:
            print(f"  {idx}. {img_path}")
            print(f"     ❌ Error: {e}")
            results.append({
                'image': img_path,
                'similarity': 0.0,
                'interpretation': 'Error',
                'is_match': False
            })
    
    # Summary
    matches = [r for r in results if r['is_match']]
    print("\n" + "=" * 70)
    print(f"SUMMARY: Found {len(matches)} potential matches out of {len(comparison_images)} images")
    print("=" * 70)
    
    return results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Compare two images to check if they show the same person using ResNet features"
    )
    
    parser.add_argument(
        "--image1",
        type=str,
        required=True,
        help="Path to first image"
    )
    
    parser.add_argument(
        "--image2",
        type=str,
        nargs="+",
        required=True,
        help="Path to second image (or multiple images for batch comparison)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: compare image1 against all images in image2 list"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only show similarity score)"
    )
    
    args = parser.parse_args()
    
    if args.batch and len(args.image2) > 1:
        # Batch comparison mode
        batch_compare(args.image1, args.image2)
    elif len(args.image2) == 1:
        # Single comparison mode
        similarity, interpretation, is_match = compare_two_persons(
            args.image1,
            args.image2[0],
            show_details=not args.quiet
        )
        
        if args.quiet and similarity is not None:
            print(f"{similarity:.4f}")
    else:
        print("Error: For single comparison, provide 2 images. For batch, use --batch flag.")


if __name__ == "__main__":
    # Check if running with command line arguments
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        # Interactive example mode
        print("\n💡 No arguments provided. Running interactive example mode.\n")
        print("To use command line:")
        print("  python example_reid.py --image1 person1.jpg --image2 person2.jpg")
        print("  python example_reid.py --image1 ref.jpg --image2 img1.jpg img2.jpg --batch")
        print("\n" + "=" * 70)
        
        # Example usage with hardcoded paths
        print("\n📝 Example 1: Compare two specific images")
        print("=" * 70)
        
        # Update these paths to your actual images
        img1 = "/Users/psinha/Desktop/test_images/person1.jpg"
        img2 = "/Users/psinha/Desktop/test_images/person2.jpg"
        
        try:
            similarity, interpretation, is_match = compare_two_persons(img1, img2)
        except FileNotFoundError:
            print("\n⚠️  Default example images not found.")
            print("\nTo use this script:")
            print("1. Update the paths in this script, OR")
            print("2. Run with command line arguments:")
            print(f"   python example_reid.py --image1 YOUR_IMAGE1.jpg --image2 YOUR_IMAGE2.jpg")
            
            # Try with sample images from the project
            print("\n\nTrying with a COCO dataset sample...")
            from utils import load_image
            
            # Use same image twice (should give high similarity)
            test_img = "http://images.cocodataset.org/val2017/000000039769.jpg"
            print(f"\nComparing image to itself (should be 1.0 similarity):")
            
            extractor = FeatureExtractor()
            img = load_image(test_img)
            
            # Compare image to itself
            features1 = extractor.extract_features(img)
            features2 = extractor.extract_features(img)
            
            similarity, interpretation = extractor.compute_similarity(
                features1,
                features2,
                interpret=True
            )
            
            print(f"\nSimilarity Score: {similarity:.4f}")
            print(f"Interpretation: {interpretation}")
            print("\n✅ Self-comparison successful! Feature extraction is working.")
