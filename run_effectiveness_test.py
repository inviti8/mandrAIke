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
    parser.add_argument('--output-dir', '-o', default='test_results',
                       help='Output directory for results (default: test_results)')
    parser.add_argument('--strengths', '-s', nargs='+', 
                       default=['low', 'medium', 'high'],
                       choices=['low', 'medium', 'high'],
                       help='Protection strengths to test (default: all)')
    parser.add_argument('--methods', '-m', nargs='+',
                       default=['poison', 'hallucinogen'],
                       choices=['poison', 'hallucinogen'],
                       help='Protection methods to test (default: all)')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick test with only low strength and poison method')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick:
        args.strengths = ['low']
        args.methods = ['poison']
        print("Running quick test with low protection strength and poison method only...")
    
    print("MandrAIk Effectiveness Test Runner")
    print("=" * 40)
    print(f"Test images: {args.test_images}")
    print(f"Output directory: {args.output_dir}")
    print(f"Protection strengths: {', '.join(args.strengths)}")
    print(f"Protection methods: {', '.join(args.methods)}")
    print()
    
    # Check if test images directory exists
    test_images_path = Path(args.test_images)
    if not test_images_path.exists():
        print(f"Warning: Test images directory '{args.test_images}' does not exist.")
        print("The test will create sample images automatically.")
        print()
    
    try:
        # Initialize and run test
        tester = MandrAIkEffectivenessTester(
            test_images_dir=args.test_images,
            output_dir=args.output_dir
        )
        
        results = tester.run_comprehensive_test(args.strengths, args.methods)
        
        if results:
            print("\n" + "=" * 40)
            print("TEST COMPLETED SUCCESSFULLY!")
            print("=" * 40)
            
            # Print summary by method
            for method, method_results in results.get('by_method', {}).items():
                print(f"\nProtection Method: {method.upper()}")
                print(f"  Confidence Reduction Success Rate: {method_results['confidence_reduction_success_rate']:.1%}")
                print(f"  Average Confidence Reduction: {method_results['avg_confidence_reduction']:.3f}")
                print(f"  Attack Success Rate: {method_results['attack_success_rate']:.1%}")
                print(f"  Average PSNR: {method_results['avg_psnr']:.1f} dB")
            
            # Print summary by strength
            for strength, strength_results in results.get('by_strength', {}).items():
                print(f"\nProtection Strength: {strength.upper()}")
                print(f"  Confidence Reduction Success Rate: {strength_results['confidence_reduction_success_rate']:.1%}")
                print(f"  Average Confidence Reduction: {strength_results['avg_confidence_reduction']:.3f}")
                print(f"  Attack Success Rate: {strength_results['attack_success_rate']:.1%}")
            
            print(f"\nDetailed results saved to: {args.output_dir}")
            print("Files generated:")
            print(f"  - effectiveness_results.json (raw data)")
            print(f"  - effectiveness_report.md (detailed report)")
            print(f"  - effectiveness_analysis.png (visualizations)")
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