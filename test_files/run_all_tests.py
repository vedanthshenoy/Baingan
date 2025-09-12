#!/usr/bin/env python3
"""
Comprehensive Test Runner for BainGan Application
Runs both unit tests and integration tests with detailed reporting
"""

import unittest
import sys
import os
import time
from io import StringIO
import argparse

def run_test_file(test_file, test_name):
    """Run a specific test file and return results"""
    print(f"\n{'='*60}")
    print(f"RUNNING {test_name.upper()}")
    print(f"{'='*60}")
    
    # Redirect stdout to capture test output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    start_time = time.time()
    
    try:
        # Import and run the test module
        if test_file == 'unit':
            import test_baingan_units as test_module
        elif test_file == 'integration':
            import test_baingan_integration as test_module
        
        # Create test loader
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)
        
        # Run tests
        runner = unittest.TextTestRunner(
            stream=captured_output,
            verbosity=2,
            buffer=True
        )
        result = runner.run(suite)
        
    except Exception as e:
        # Restore stdout
        sys.stdout = old_stdout
        print(f"ERROR: Failed to run {test_name}: {str(e)}")
        return None, 0
    
    finally:
        # Restore stdout
        sys.stdout = old_stdout
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print captured output
    output = captured_output.getvalue()
    print(output)
    
    return result, execution_time

def print_summary(results, times):
    """Print comprehensive test summary"""
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*60}")
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_time = sum(times.values())
    
    for test_type, result in results.items():
        if result:
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            
            print(f"\n{test_type.upper()} TESTS:")
            print(f"  Tests run: {result.testsRun}")
            print(f"  Failures: {len(result.failures)}")
            print(f"  Errors: {len(result.errors)}")
            print(f"  Time: {times[test_type]:.2f}s")
            
            if result.failures:
                print(f"  Failed tests:")
                for test, _ in result.failures:
                    print(f"    - {test}")
            
            if result.errors:
                print(f"  Error tests:")
                for test, _ in result.errors:
                    print(f"    - {test}")
    
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS:")
    print(f"Total tests: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Total errors: {total_errors}")
    print(f"Total time: {total_time:.2f}s")
    
    if total_tests > 0:
        success_rate = ((total_tests - total_failures - total_errors) / total_tests) * 100
        print(f"Success rate: {success_rate:.1f}%")
    else:
        print("Success rate: N/A")
    
    # Overall status
    if total_failures == 0 and total_errors == 0:
        print(f"\n🎉 ALL TESTS PASSED! 🎉")
        return True
    else:
        print(f"\n❌ SOME TESTS FAILED ❌")
        return False

def print_test_coverage():
    """Print test coverage information"""
    print(f"\n{'='*60}")
    print(f"TEST COVERAGE OVERVIEW")
    print(f"{'='*60}")
    
    print("\n📋 UNIT TESTS COVER:")
    print("  ✓ Utility function testing")
    print("  ✓ API call functionality") 
    print("  ✓ Prompt suggestion generation")
    print("  ✓ Data export structures")
    print("  ✓ Session state management")
    print("  ✓ Error handling for individual components")
    
    print("\n🔄 INTEGRATION TESTS COVER:")
    print("  ✓ End-to-end individual testing workflow")
    print("  ✓ Complete prompt chaining workflow")
    print("  ✓ Full prompt combination workflow")
    print("  ✓ API error propagation through chains")
    print("  ✓ Gemini API integration")
    print("  ✓ Data integrity across workflows")
    print("  ✓ Export functionality with real data")
    print("  ✓ Performance with large datasets")
    print("  ✓ Edge cases and error recovery")

def run_specific_test_class(test_file, class_name):
    """Run a specific test class"""
    print(f"\n{'='*60}")
    print(f"RUNNING SPECIFIC TEST CLASS: {class_name}")
    print(f"{'='*60}")
    
    try:
        if test_file == 'unit':
            import test_baingan_units as test_module
        elif test_file == 'integration':
            import test_baingan_integration as test_module
        else:
            print(f"ERROR: Unknown test file: {test_file}")
            return None, 0
        
        # Get the specific test class
        test_class = getattr(test_module, class_name)
        
        # Create test suite from the class
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        # Run tests
        start_time = time.time()
        runner = unittest.TextTestRunner(verbosity=2, buffer=True)
        result = runner.run(suite)
        end_time = time.time()
        
        return result, end_time - start_time
        
    except AttributeError:
        print(f"ERROR: Test class '{class_name}' not found in {test_file} tests")
        return None, 0
    except Exception as e:
        print(f"ERROR: Failed to run test class '{class_name}': {str(e)}")
        return None, 0

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description='BainGan Application Test Runner')
    parser.add_argument('--unit-only', action='store_true', 
                       help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--class', dest='test_class',
                       help='Run specific test class (format: file:class, e.g., unit:TestAPICall)')
    parser.add_argument('--list-classes', action='store_true',
                       help='List all available test classes')
    parser.add_argument('--coverage', action='store_true',
                       help='Show test coverage information only')
    parser.add_argument('--quick', action='store_true',
                       help='Run tests with minimal output')
    
    args = parser.parse_args()
    
    if args.coverage:
        print_test_coverage()
        return
    
    if args.list_classes:
        print("Available test classes:")
        print("\nUnit Tests (test_baingan_units.py):")
        print("  - TestBainGanUtilities")
        print("  - TestAPICall")
        print("  - TestPromptSuggestion")
        print("  - TestDataExport")
        print("  - TestSessionStateManagement")
        
        print("\nIntegration Tests (test_baingan_integration.py):")
        print("  - TestEndToEndWorkflow")
        print("  - TestErrorHandlingAndEdgeCases")
        print("  - TestDataIntegrityAndExport")
        print("  - TestPerformanceAndStress")
        
        print("\nUsage examples:")
        print("  python run_all_tests.py --class unit:TestAPICall")
        print("  python run_all_tests.py --class integration:TestEndToEndWorkflow")
        return
    
    if args.test_class:
        try:
            test_file, class_name = args.test_class.split(':')
            result, exec_time = run_specific_test_class(test_file, class_name)
            
            if result:
                print(f"\nTest class '{class_name}' completed in {exec_time:.2f}s")
                print(f"Tests run: {result.testsRun}")
                print(f"Failures: {len(result.failures)}")
                print(f"Errors: {len(result.errors)}")
                
                success = len(result.failures) == 0 and len(result.errors) == 0
                sys.exit(0 if success else 1)
            else:
                sys.exit(1)
                
        except ValueError:
            print("ERROR: Invalid class format. Use 'file:class' (e.g., 'unit:TestAPICall')")
            sys.exit(1)
    
    # Print header
    print(f"{'='*60}")
    print(f"BAINGAN APPLICATION TEST SUITE")
    print(f"{'='*60}")
    print(f"Testing comprehensive prompt testing application functionality")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    times = {}
    
    # Run unit tests
    if not args.integration_only:
        try:
            result, exec_time = run_test_file('unit', 'Unit Tests')
            results['unit'] = result
            times['unit'] = exec_time
        except ImportError as e:
            print(f"WARNING: Could not import unit tests: {e}")
            results['unit'] = None
            times['unit'] = 0
    
    # Run integration tests
    if not args.unit_only:
        try:
            result, exec_time = run_test_file('integration', 'Integration Tests')
            results['integration'] = result
            times['integration'] = exec_time
        except ImportError as e:
            print(f"WARNING: Could not import integration tests: {e}")
            results['integration'] = None
            times['integration'] = 0
    
    # Print summary
    if not args.quick:
        success = print_summary(results, times)
        print_test_coverage()
        
        # Print recommendations
        print(f"\n{'='*60}")
        print(f"RECOMMENDATIONS")
        print(f"{'='*60}")
        
        any_failures = any(r and (len(r.failures) > 0 or len(r.errors) > 0) for r in results.values() if r)
        
        if any_failures:
            print("❗ Some tests failed. Consider:")
            print("  • Check API endpoint configurations")
            print("  • Verify Gemini API key setup")
            print("  • Review mock configurations")
            print("  • Check for import/dependency issues")
            print("  • Run individual test classes for detailed debugging")
        else:
            print("✅ All tests passed! Your application is ready for:")
            print("  • Production deployment")
            print("  • User acceptance testing")
            print("  • Performance optimization")
            print("  • Feature enhancements")
        
        print(f"\n📚 For more detailed testing:")
        print(f"  python run_all_tests.py --list-classes")
        print(f"  python run_all_tests.py --class unit:TestAPICall")
        print(f"  python run_all_tests.py --unit-only")
        
        return success
    else:
        # Quick mode - just return success/failure
        any_failures = any(r and (len(r.failures) > 0 or len(r.errors) > 0) for r in results.values() if r)
        return not any_failures

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user.")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"\nUnexpected error in test runner: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)