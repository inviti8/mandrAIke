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
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick test with only low strength')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick:
        args.strengths = ['low']
        print("Running quick test with low protection strength only...")
    
    print("MandrAIk Effectiveness Test Runner")
    print("=" * 40)
    print(f"Test images: {args.test_images}")
    print(f"Output directory: {args.output_dir}")
    print(f"Protection strengths: {', '.join(args.strengths)}")
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
        
        results = tester.run_comprehensive_test(args.strengths)
        
        if results:
            print("\n" + "=" * 40)
            print("TEST COMPLETED SUCCESSFULLY!")
            print("=" * 40)
            
            # Print summary
            for strength, strength_results in results.items():
                print(f"\nProtection Strength: {strength.upper()}")
                
                # Find best performing model
                best_model = None
                best_rate = 0
                
                for model_name, model_results in strength_results.items():
                    if model_name in ['quality', 'processing_time']:
                        continue
                    
                    attack_rate = model_results['attack_success_rate']
                    if attack_rate > best_rate:
                        best_rate = attack_rate
                        best_model = model_name
                
                if best_model:
                    print(f"  Best attack success rate: {best_rate:.1%} ({best_model})")
                
                # Print quality metrics if available
                if 'quality' in strength_results:
                    quality = strength_results['quality']
                    print(f"  Average PSNR: {quality['avg_psnr']:.1f} dB")
                    print(f"  Average SSIM: {quality['avg_ssim']:.3f}")
            
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