"""
LLM integration for describing cropped images.
Uses Claude API to generate short labels for detected objects.
"""

import ssl
import anthropic
import json
import base64
from pathlib import Path

# Fix SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        str: Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def describe_image(
    image_path: str, 
    api_key: str, 
    model: str = "claude-sonnet-4-20250514"
) -> str:
    """
    Call Claude API to describe an image in 1-3 words suitable as a detection label.
    
    Args:
        image_path: Path to cropped image
        api_key: Anthropic API key
        model: Claude model to use
        
    Returns:
        str: Short description (e.g., "yellow shirt person", "brown dog", "red car")
        
    Raises:
        Exception: If API call fails
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    # Encode image
    image_data = encode_image_to_base64(image_path)
    
    # Determine media type from file extension
    image_path_obj = Path(image_path)
    extension = image_path_obj.suffix.lower()
    media_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    media_type = media_type_map.get(extension, 'image/jpeg')
    
    prompt = """Analyze this cropped image and provide a label for object detection.

CRITICAL RULES:
1. Start with the base object type (person, dog, car, etc.)
2. Add ONE most distinctive color OR feature
3. Use simple common words only
4. Maximum 2-3 words total

Format: "[color] [object]" OR "[feature] [object]"

GOOD examples:
- "person in yellow"
- "person in red"
- "person in blue"
- "yellow jacket"
- "red car"
- "brown dog"
- "white shirt"

BAD examples (too specific):
- "person with yellow raincoat and umbrella" (too long)
- "golden retriever dog" (too specific breed)
- "person wearing sunglasses" (temporary feature)

Return ONLY the label in this exact format, nothing else."""

    try:
        message = client.messages.create(
            model=model,
            max_tokens=100,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        # Extract text response
        response_text = message.content[0].text.strip()
        
        return response_text
        
    except Exception as e:
        raise Exception(f"LLM API call failed: {e}")


def describe_image_batch(
    image_paths: list[str], 
    api_key: str, 
    model: str = "claude-sonnet-4-20250514"
) -> dict[str, str]:
    """
    Describe multiple images in batch.
    
    Args:
        image_paths: List of image paths
        api_key: Anthropic API key
        model: Claude model to use
        
    Returns:
        dict: Mapping of image_path -> description
    """
    results = {}
    
    for image_path in image_paths:
        try:
            description = describe_image(image_path, api_key, model)
            results[image_path] = description
            print(f"✓ {Path(image_path).name}: {description}")
        except Exception as e:
            results[image_path] = f"ERROR: {str(e)}"
            print(f"✗ {Path(image_path).name}: {str(e)}")
    
    return results


if __name__ == "__main__":
    """Test the image describer."""
    import sys
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in .env file")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Usage: python llm/image_describer.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"Describing image: {image_path}")
    description = describe_image(image_path, api_key)
    print(f"\nLabel: {description}")
