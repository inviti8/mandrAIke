#!/usr/bin/env python3
"""
Comprehensive Test Runner for MandrAIk Adversarial Image Generator

This script runs all test suites and generates detailed evaluation reports.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import unittest
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_test_suite(test_module, test_class, suite_name):
    """Run a specific test suite and return results."""
    print(f"\n{'='*60}")
    print(f"Running {suite_name}")
    print(f"{'='*60}")
    
    # Import test module
    try:
        module = __import__(test_module)
        test_class_obj = getattr(module, test_class)
    except ImportError as e:
        print(f"❌ Failed to import {test_module}: {e}")
        return None
    except AttributeError as e:
        print(f"❌ Failed to find {test_class} in {test_module}: {e}")
        return None
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(test_class_obj)
    
    # Run tests
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    end_time = time.time()
    
    # Compile results
    suite_results = {
        'suite_name': suite_name,
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
        'execution_time': end_time - start_time,
        'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
        'was_successful': result.wasSuccessful(),
        'failures_details': [(str(test), traceback) for test, traceback in result.failures],
        'errors_details': [(str(test), traceback) for test, traceback in result.errors]
    }
    
    return suite_results

def generate_test_report(all_results, output_dir):
    """Generate comprehensive test report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"test_report_{timestamp}.json")
    
    # Calculate overall statistics
    total_tests = sum(r['tests_run'] for r in all_results if r)
    total_failures = sum(r['failures'] for r in all_results if r)
    total_errors = sum(r['errors'] for r in all_results if r)
    total_time = sum(r['execution_time'] for r in all_results if r)
    overall_success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
    
    # Compile report
    report = {
        'timestamp': timestamp,
        'overall_summary': {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_execution_time': total_time,
            'overall_success_rate': overall_success_rate,
            'all_suites_passed': all(r['was_successful'] for r in all_results if r)
        },
        'suite_results': all_results,
        'recommendations': generate_recommendations(all_results)
    }
    
    # Save report
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report, report_file

def generate_recommendations(results):
    """Generate recommendations based on test results."""
    recommendations = []
    
    # Check for failed suites
    failed_suites = [r for r in results if r and not r['was_successful']]
    if failed_suites:
        recommendations.append({
            'type': 'critical',
            'message': f"{len(failed_suites)} test suite(s) failed. Review failures and fix issues.",
            'suites': [r['suite_name'] for r in failed_suites]
        })
    
    # Check for low success rates
    low_success_suites = [r for r in results if r and r['success_rate'] < 80]
    if low_success_suites:
        recommendations.append({
            'type': 'warning',
            'message': f"{len(low_success_suites)} suite(s) have low success rates (<80%).",
            'suites': [(r['suite_name'], f"{r['success_rate']:.1f}%") for r in low_success_suites]
        })
    
    # Check for long execution times
    slow_suites = [r for r in results if r and r['execution_time'] > 300]  # 5 minutes
    if slow_suites:
        recommendations.append({
            'type': 'performance',
            'message': f"{len(slow_suites)} suite(s) took longer than 5 minutes to execute.",
            'suites': [(r['suite_name'], f"{r['execution_time']:.1f}s") for r in slow_suites]
        })
    
    # Check for no failures (might indicate insufficient testing)
    perfect_suites = [r for r in results if r and r['failures'] == 0 and r['errors'] == 0 and r['tests_run'] > 0]
    if len(perfect_suites) == len(results):
        recommendations.append({
            'type': 'info',
            'message': "All test suites passed! Consider adding more edge case tests for robustness."
        })
    
    return recommendations

def print_summary_report(report):
    """Print a summary of the test report."""
    print(f"\n{'='*80}")
    print("TEST EXECUTION SUMMARY")
    print(f"{'='*80}")
    
    summary = report['overall_summary']
    print(f"Total Tests Run: {summary['total_tests']}")
    print(f"Total Failures: {summary['total_failures']}")
    print(f"Total Errors: {summary['total_errors']}")
    print(f"Total Execution Time: {summary['total_execution_time']:.2f} seconds")
    print(f"Overall Success Rate: {summary['overall_success_rate']:.1f}%")
    print(f"All Suites Passed: {'✅ Yes' if summary['all_suites_passed'] else '❌ No'}")
    
    print(f"\n{'='*80}")
    print("SUITE DETAILS")
    print(f"{'='*80}")
    
    for suite_result in report['suite_results']:
        if suite_result:
            print(f"\n{suite_result['suite_name']}:")
            print(f"  Tests: {suite_result['tests_run']}")
            print(f"  Failures: {suite_result['failures']}")
            print(f"  Errors: {suite_result['errors']}")
            print(f"  Success Rate: {suite_result['success_rate']:.1f}%")
            print(f"  Execution Time: {suite_result['execution_time']:.2f}s")
            print(f"  Status: {'✅ PASSED' if suite_result['was_successful'] else '❌ FAILED'}")
    
    if report['recommendations']:
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}")
        
        for rec in report['recommendations']:
            print(f"\n{rec['type'].upper()}: {rec['message']}")
            if 'suites' in rec:
                for suite_info in rec['suites']:
                    if isinstance(suite_info, tuple):
                        print(f"  - {suite_info[0]}: {suite_info[1]}")
                    else:
                        print(f"  - {suite_info}")

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run comprehensive tests for MandrAIk')
    parser.add_argument('--output-dir', default='test_results', 
                       help='Directory to save test results (default: test_results)')
    parser.add_argument('--suite', choices=['all', 'effectiveness', 'benchmark', 'performance'],
                       default='all', help='Which test suite to run (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define test suites
    test_suites = [
        ('test_mandrAIk', 'TestMandrAIkEffectiveness', 'Effectiveness Tests'),
        ('test_mandrAIk', 'TestMandrAIkPerformance', 'Performance Tests'),
        ('test_benchmark', 'BenchmarkTest', 'Benchmark Tests')
    ]
    
    # Filter suites based on argument
    if args.suite != 'all':
        if args.suite == 'effectiveness':
            test_suites = [test_suites[0]]
        elif args.suite == 'benchmark':
            test_suites = [test_suites[2]]
        elif args.suite == 'performance':
            test_suites = [test_suites[1]]
    
    print("🚀 Starting MandrAIk Test Suite")
    print(f"Output directory: {args.output_dir}")
    print(f"Test suites: {[suite[2] for suite in test_suites]}")
    
    # Run test suites
    all_results = []
    for test_module, test_class, suite_name in test_suites:
        try:
            result = run_test_suite(test_module, test_class, suite_name)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Error running {suite_name}: {e}")
            all_results.append(None)
    
    # Generate report
    report, report_file = generate_test_report(all_results, args.output_dir)
    
    # Print summary
    print_summary_report(report)
    
    # Save detailed results if available
    for suite_result in all_results:
        if suite_result and hasattr(suite_result, 'benchmark_results'):
            # Save benchmark results
            benchmark_file = os.path.join(args.output_dir, f"{suite_result['suite_name'].lower().replace(' ', '_')}_results.json")
            with open(benchmark_file, 'w') as f:
                json.dump(suite_result.benchmark_results, f, indent=2)
            print(f"\n📊 Detailed results saved to: {benchmark_file}")
    
    print(f"\n📋 Full report saved to: {report_file}")
    
    # Exit with appropriate code
    if report['overall_summary']['all_suites_passed']:
        print("\n✅ All test suites completed successfully!")
        return 0
    else:
        print("\n❌ Some test suites failed. Check the report for details.")
        return 1

if __name__ == "__main__":
    exit(main()) 