#!/usr/bin/env python3
"""
Simple test script for MandrAIk methods.
"""

import sys
from pathlib import Path

# Add the MandrAIk directory to the path
sys.path.append(str(Path(__file__).parent / 'MandrAIk'))

from MandrAIk import MandrAIk

def test_dream_method():
    """Test the updated dream method with Perlin noise generation."""
    print("Testing updated dream method with Perlin noise...")
    
    # Initialize MandrAIk
    mandrAIk = MandrAIk()
    
    # Find a test image
    test_dir = Path("test_images")
    if not test_dir.exists():
        print("test_images directory not found!")
        return False
    
    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if not test_images:
        print("No test images found!")
        return False
    
    test_image = test_images[0]
    print(f"Using test image: {test_image}")
    
    # Create output directory
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Test dream method
        output_path = output_dir / f"dream_test_{test_image.stem}.jpg"
        
        print("Applying dream method...")
        dreamed_img = mandrAIk.dream(str(test_image), str(output_path))
        
        print(f"Dream method completed successfully!")
        print(f"Original image: {test_image}")
        print(f"Dreamed image: {output_path}")
        print(f"Dreamed image shape: {dreamed_img.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error testing dream method: {e}")
        return False

def test_poison_method():
    """Test the poison method for comparison."""
    print("\nTesting poison method for comparison...")
    
    # Initialize MandrAIk
    mandrAIk = MandrAIk()
    
    # Find a test image
    test_dir = Path("test_images")
    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    test_image = test_images[0]
    
    # Create output directory
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Test poison method
        output_path = output_dir / f"poison_test_{test_image.stem}.jpg"
        
        print("Applying poison method...")
        poisoned_img = mandrAIk.poison(str(test_image), str(output_path), protection_strength=0.15)
        
        print(f"Poison method completed successfully!")
        print(f"Original image: {test_image}")
        print(f"Poisoned image: {output_path}")
        print(f"Poisoned image shape: {poisoned_img.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error testing poison method: {e}")
        return False

def main():
    """Run all tests."""
    print("MandrAIk Method Test")
    print("=" * 30)
    
    # Test dream method
    dream_success = test_dream_method()
    
    # Test poison method for comparison
    poison_success = test_poison_method()
    
    # Summary
    print(f"\n{'='*30}")
    print("Test Summary")
    print("="*30)
    print(f"Dream method: {'✅ PASS' if dream_success else '❌ FAIL'}")
    print(f"Poison method: {'✅ PASS' if poison_success else '❌ FAIL'}")
    
    if dream_success and poison_success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())


