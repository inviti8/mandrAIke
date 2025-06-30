#!/usr/bin/env python3
"""
Simple test runner for MandrAIk effectiveness testing.
"""

import argparse
import sys
from pathlib import Path
from test_mandrAIk_effectiveness import MandrAIkEffectivenessTester

def main():
    parser = argparse.ArgumentParser(description='Run MandrAIk effectiveness test')
    parser.add_argument('--test-images', '-i', default='test_images',
                       help='Directory containing test images (default: test_images)')
    parser.add_argument('--test-dataset', '-d', 
                       help='Directory containing categorized test dataset (e.g., test_dataset)')
    parser.add_argument('--output-dir', '-o', default='test_results',
                       help='Output directory for results (default: test_results)')
    parser.add_argument('--strengths', '-s', nargs='+', 
                       default=['low', 'medium', 'high'],
                       choices=['low', 'medium', 'high'],
                       help='Protection strengths to test (default: all)')
    parser.add_argument('--methods', '-m', nargs='+', 
                       default=['poison', 'fourier'],
                       choices=['poison', 'fourier'],
                       help='Protection methods to test (default: poison, fourier)')
    parser.add_argument('--noise-types', '-n', nargs='+',
                       default=['perlin', 'linear', 'fourier'],
                       choices=['perlin', 'linear', 'fourier'],
                       help='Noise types to test (default: all)')
    parser.add_argument('--images-per-category', type=int, default=30,
                       help='Number of images to test per category (default: 30)')
    parser.add_argument('--categories', nargs='+',
                       help='Specific categories to test (default: all categories found)')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick test with only low strength, poison method, perlin noise, and 5 images per category')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick:
        args.strengths = ['low']
        args.methods = ['poison']
        args.noise_types = ['perlin']
        args.images_per_category = 5
        print("Running quick test with low protection strength, poison method, perlin noise, and 5 images per category...")
    
    print("MandrAIk Effectiveness Test Runner")
    print("=" * 40)
    
    # Determine test mode
    if args.test_dataset:
        print(f"Test dataset: {args.test_dataset}")
        print(f"Images per category: {args.images_per_category}")
        if args.categories:
            print(f"Categories: {', '.join(args.categories)}")
        else:
            print("Categories: all found")
        test_mode = "dataset"
    else:
        print(f"Test images: {args.test_images}")
        test_mode = "images"
    
    print(f"Output directory: {args.output_dir}")
    print(f"Protection strengths: {', '.join(args.strengths)}")
    print(f"Protection methods: {', '.join(args.methods)}")
    print(f"Noise types: {', '.join(args.noise_types)}")
    print()
    
    # Check if test directory exists
    if test_mode == "dataset":
        test_path = Path(args.test_dataset)
    else:
        test_path = Path(args.test_images)
    
    if not test_path.exists():
        print(f"Warning: Test directory '{test_path}' does not exist.")
        print("The test will create sample images automatically.")
        print()
    
    try:
        # Initialize and run test
        if test_mode == "dataset":
            tester = MandrAIkEffectivenessTester(
                test_dataset_dir=args.test_dataset,
                output_dir=args.output_dir
            )
            results = tester.run_comprehensive_dataset_test(
                args.strengths, 
                args.methods, 
                args.noise_types,
                args.images_per_category,
                args.categories
            )
        else:
            tester = MandrAIkEffectivenessTester(
                test_images_dir=args.test_images,
                output_dir=args.output_dir
            )
            results = tester.run_comprehensive_test(args.strengths, args.methods, args.noise_types)
        
        if results:
            print("\n" + "=" * 40)
            print("TEST COMPLETED SUCCESSFULLY!")
            print("=" * 40)
            
            # Print summary by method and noise type
            for method, method_results in results.get('by_method', {}).items():
                print(f"\nProtection Method: {method.upper()}")
                for noise_type, noise_results in method_results.get('by_noise_type', {}).items():
                    print(f"  Noise Type: {noise_type.upper()}")
                    print(f"    Confidence Reduction Success Rate: {noise_results['confidence_reduction_success_rate']:.1%}")
                    print(f"    Average Confidence Change: {noise_results['avg_confidence_reduction']:.3f}")
                    print(f"    Attack Success Rate: {noise_results['attack_success_rate']:.1%}")
                    print(f"    Average PSNR: {noise_results['avg_psnr']:.1f} dB")
            
            # Print summary by category if available
            if 'by_category' in results:
                print(f"\nResults by Category:")
                for category, category_results in results.get('by_category', {}).items():
                    print(f"  {category.upper()}")
                    print(f"    Confidence Reduction Success Rate: {category_results['confidence_reduction_success_rate']:.1%}")
                    print(f"    Average Confidence Change: {category_results['avg_confidence_reduction']:.3f}")
                    print(f"    Attack Success Rate: {category_results['attack_success_rate']:.1%}")
            
            # Print summary by strength
            for strength, strength_results in results.get('by_strength', {}).items():
                print(f"\nProtection Strength: {strength.upper()}")
                print(f"  Confidence Reduction Success Rate: {strength_results['confidence_reduction_success_rate']:.1%}")
                print(f"  Average Confidence Change: {strength_results['avg_confidence_reduction']:.3f}")
                print(f"  Attack Success Rate: {strength_results['attack_success_rate']:.1%}")
            
            print(f"\nDetailed results saved to: {args.output_dir}")
            print("Files generated:")
            print(f"  - effectiveness_results.json (raw data)")
            print(f"  - effectiveness_report.md (detailed report)")
            print(f"  - effectiveness_analysis.png (visualizations)")
            if test_mode == "dataset":
                print(f"  - category_analysis.png (category-specific visualizations)")
            print(f"  - protected_*.jpg (protected test images)")
            
        else:
            print("Test failed to complete!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running test: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 