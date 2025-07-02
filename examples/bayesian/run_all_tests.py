#!/usr/bin/env python3
"""
Comprehensive Test Runner for Bayesian Optimization System.
Runs all tests for new features and existing functionality to ensure no regressions.
"""

import sys
import time
import traceback
import importlib
from pathlib import Path
from typing import Dict, List, Tuple


def run_test_module(module_name: str, test_function: str) -> Tuple[bool, str, float]:
    """
    Run a test module and return success status, message, and duration.
    """
    try:
        print(f"\n{'='*70}")
        print(f"🧪 Running {module_name}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Import the module
        module = importlib.import_module(module_name)
        
        # Get the test function
        test_func = getattr(module, test_function)
        
        # Run the test
        test_func()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ {module_name} PASSED (Duration: {duration:.2f}s)")
        return True, f"PASSED in {duration:.2f}s", duration
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        error_msg = f"FAILED: {type(e).__name__}: {str(e)}"
        print(f"\n❌ {module_name} FAILED")
        print(f"Error: {error_msg}")
        print(f"Traceback:")
        traceback.print_exc()
        
        return False, error_msg, duration


def main():
    """
    Run comprehensive test suite for Bayesian optimization system.
    """
    print("🚀 Comprehensive Bayesian Optimization Test Suite")
    print("=" * 70)
    print("This will test all features including new implementations.")
    print("Expected to validate functionality and check for regressions.")
    
    # Define all tests to run
    test_suite = [
        # Existing tests (regression testing)
        {
            'module': 'examples.bayesian.test_automatic_initialization',
            'function': 'test_automatic_initialization',
            'category': 'Existing Features',
            'description': 'Automatic initialization when insufficient samples'
        },
        {
            'module': 'examples.bayesian.test_initialization_sampling',
            'function': 'test_initialization_sampling',
            'category': 'Existing Features', 
            'description': 'Different initialization sampling methods'
        },
        {
            'module': 'examples.bayesian.test_full_initialization_workflow',
            'function': 'test_full_initialization_workflow',
            'category': 'Existing Features',
            'description': 'Complete initialization to optimization workflow'
        },
        {
            'module': 'examples.bayesian.test_mixed_optimization',
            'function': 'run_mixed_test',
            'category': 'Existing Features',
            'description': 'Mixed continuous and categorical optimization'
        },
        {
            'module': 'examples.bayesian.test_categorical_optimization',
            'function': 'run_categorical_test',
            'category': 'Existing Features',
            'description': 'Pure categorical optimization'
        },
        {
            'module': 'examples.bayesian.basic_optimization',
            'function': 'run_optimization_example',
            'category': 'Existing Features',
            'description': 'Basic optimization with context'
        },
        
        # New feature tests
        {
            'module': 'examples.bayesian.test_multi_objective_optimization',
            'function': 'test_multi_objective_optimization',
            'category': 'New Features',
            'description': 'Multi-objective optimization (qEHVI, qNEHVI)'
        },
        {
            'module': 'examples.bayesian.test_advanced_acquisition_functions',
            'function': 'test_advanced_acquisition_functions',
            'category': 'New Features',
            'description': 'Advanced acquisition functions (KG, Thompson Sampling)'
        },
        {
            'module': 'examples.bayesian.test_constraint_handling',
            'function': 'test_constraint_handling',
            'category': 'New Features',
            'description': 'Constraint handling with inequality constraints'
        },
        {
            'module': 'examples.bayesian.test_model_serialization',
            'function': 'test_model_serialization',
            'category': 'New Features',
            'description': 'Model serialization and persistence'
        },
        {
            'module': 'examples.bayesian.test_error_handling_validation',
            'function': 'test_error_handling_and_validation',
            'category': 'New Features',
            'description': 'Error handling and input validation'
        },
    ]
    
    # Results tracking
    results = {}
    total_duration = 0
    passed_count = 0
    failed_count = 0
    
    print(f"\n📋 Test Plan:")
    print(f"Total tests to run: {len(test_suite)}")
    
    # Group by category
    categories = {}
    for test in test_suite:
        cat = test['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(test)
    
    for category, tests in categories.items():
        print(f"\n{category}:")
        for test in tests:
            print(f"  • {test['description']}")
    
    print(f"\n{'='*70}")
    print("🏃 Starting Test Execution")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Run all tests
    for i, test in enumerate(test_suite, 1):
        print(f"\n[{i}/{len(test_suite)}] {test['description']}")
        print("-" * 50)
        
        success, message, duration = run_test_module(test['module'], test['function'])
        
        results[test['module']] = {
            'success': success,
            'message': message,
            'duration': duration,
            'category': test['category'],
            'description': test['description']
        }
        
        total_duration += duration
        if success:
            passed_count += 1
        else:
            failed_count += 1
        
        # Short pause between tests
        time.sleep(0.5)
    
    end_time = time.time()
    total_suite_duration = end_time - start_time
    
    # Generate comprehensive report
    print(f"\n{'='*70}")
    print("📊 COMPREHENSIVE TEST REPORT")
    print(f"{'='*70}")
    
    print(f"\n🏆 Overall Results:")
    print(f"   Total tests: {len(test_suite)}")
    print(f"   Passed: {passed_count}")
    print(f"   Failed: {failed_count}")
    print(f"   Success rate: {passed_count/len(test_suite)*100:.1f}%")
    print(f"   Total duration: {total_suite_duration:.2f}s")
    print(f"   Average per test: {total_duration/len(test_suite):.2f}s")
    
    # Results by category
    for category in categories.keys():
        category_results = [(k, v) for k, v in results.items() if v['category'] == category]
        category_passed = sum(1 for _, v in category_results if v['success'])
        category_total = len(category_results)
        
        print(f"\n📂 {category}:")
        print(f"   Passed: {category_passed}/{category_total} ({category_passed/category_total*100:.1f}%)")
        
        for module, result in category_results:
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['description']}: {result['message']}")
    
    # Failed tests details
    if failed_count > 0:
        print(f"\n❌ Failed Tests Details:")
        for module, result in results.items():
            if not result['success']:
                print(f"\n   {module}:")
                print(f"   Description: {result['description']}")
                print(f"   Error: {result['message']}")
    
    # Performance analysis
    print(f"\n⏱️ Performance Analysis:")
    sorted_by_duration = sorted(results.items(), key=lambda x: x[1]['duration'], reverse=True)
    print(f"   Slowest tests:")
    for i, (module, result) in enumerate(sorted_by_duration[:3]):
        print(f"   {i+1}. {result['description']}: {result['duration']:.2f}s")
    
    print(f"   Fastest tests:")
    for i, (module, result) in enumerate(sorted_by_duration[-3:]):
        print(f"   {i+1}. {result['description']}: {result['duration']:.2f}s")
    
    # Regression analysis
    existing_results = [(k, v) for k, v in results.items() if v['category'] == 'Existing Features']
    existing_passed = sum(1 for _, v in existing_results if v['success'])
    existing_total = len(existing_results)
    
    print(f"\n🔍 Regression Analysis:")
    if existing_passed == existing_total:
        print(f"   ✅ No regressions detected! All {existing_total} existing tests passed.")
    else:
        print(f"   ❌ Potential regressions: {existing_total - existing_passed}/{existing_total} existing tests failed.")
        for module, result in existing_results:
            if not result['success']:
                print(f"       - {result['description']}: {result['message']}")
    
    # New features analysis
    new_results = [(k, v) for k, v in results.items() if v['category'] == 'New Features']
    new_passed = sum(1 for _, v in new_results if v['success'])
    new_total = len(new_results)
    
    print(f"\n🆕 New Features Analysis:")
    print(f"   New features working: {new_passed}/{new_total} ({new_passed/new_total*100:.1f}%)")
    
    if new_passed == new_total:
        print(f"   ✅ All new features are working correctly!")
    else:
        print(f"   ⚠️  Some new features need implementation:")
        for module, result in new_results:
            if not result['success']:
                print(f"       - {result['description']}: {result['message']}")
    
    # Final recommendations
    print(f"\n💡 Recommendations:")
    
    if failed_count == 0:
        print(f"   🎉 Perfect! All tests are passing.")
        print(f"   🚀 Ready to implement all new features.")
    elif existing_passed == existing_total:
        print(f"   ✅ No regressions - existing functionality is intact.")
        print(f"   🔧 Focus on implementing the {new_total - new_passed} failing new features.")
    else:
        print(f"   🚨 Priority: Fix regressions in existing functionality first.")
        print(f"   🔧 Then implement the new features.")
    
    print(f"\n🎯 Next Steps:")
    if new_passed < new_total:
        print(f"   1. Implement missing features that are currently failing")
        print(f"   2. Add the corresponding functionality to the core system")
        print(f"   3. Re-run tests to validate implementations")
    
    if failed_count > 0:
        print(f"   4. Debug and fix any failing tests")
        print(f"   5. Ensure error handling is robust")
    
    print(f"\n✅ Test suite execution completed!")
    
    # Exit with appropriate code
    exit_code = 0 if failed_count == 0 else 1
    print(f"\nExit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test suite crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1) 