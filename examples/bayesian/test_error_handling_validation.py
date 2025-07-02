#!/usr/bin/env python3
"""
Test Error Handling and Input Validation.
Expected to gracefully handle invalid inputs, malformed data, and edge cases.
"""

import numpy as np
import time

from beam.bayesian import BayesianHPOServiceConfig, HPOService
from beam.bayesian.hp_scheme import BaseParameters
from pydantic import Field, ValidationError


class ValidTestHyperparameters(BaseParameters):
    """Valid hyperparameter schema for testing validation."""
    x: float = Field(ge=-1.0, le=1.0, description="X parameter")
    y: int = Field(ge=1, le=10, description="Y parameter")
    choice: str = Field(description="Categorical choice", json_schema_extra={
        "enum": ["a", "b", "c"]
    })


def simple_objective(h: ValidTestHyperparameters) -> float:
    """Simple objective for testing."""
    return h.x**2 + h.y * 0.1 + (0.5 if h.choice == "b" else 0.0)


def test_error_handling_and_validation():
    """
    Test comprehensive error handling and input validation.
    
    Expected behavior:
    1. Should validate inputs and provide helpful error messages
    2. Should handle malformed data gracefully
    3. Should prevent crashes and provide recovery suggestions
    4. Should validate parameter bounds and types
    """
    print("🛡️ Testing Error Handling and Input Validation")
    print("=" * 70)
    
    # Initialize service
    hparams = BayesianHPOServiceConfig(device='cpu')
    hpo_service = HPOService(hparams=hparams)
    print("✅ HPOService initialized")
    
    # Test 1: Schema validation
    print("\n🔬 Test 1: Schema Validation")
    print("-" * 40)
    
    try:
        x_scheme = ValidTestHyperparameters.model_json_schema()
        result = hpo_service.register('validation_test', x_scheme)
        print(f"✅ Valid schema registered: {result['message']}")
    except Exception as e:
        print(f"❌ Unexpected error with valid schema: {e}")
    
    # Test 2: Invalid parameter values
    print("\n🔬 Test 2: Invalid Parameter Values")
    print("-" * 40)
    
    invalid_samples = [
        # Out of bounds
        {"x": 2.0, "y": 5, "choice": "a"},  # x > 1.0
        {"x": 0.5, "y": 15, "choice": "b"},  # y > 10
        {"x": -2.0, "y": 3, "choice": "c"},  # x < -1.0
        {"x": 0.0, "y": 0, "choice": "a"},  # y < 1
        
        # Invalid categorical values
        {"x": 0.0, "y": 5, "choice": "invalid"},
        {"x": 0.0, "y": 5, "choice": "d"},
        
        # Wrong types
        {"x": "not_a_number", "y": 5, "choice": "a"},
        {"x": 0.5, "y": "not_an_int", "choice": "b"},
        {"x": 0.5, "y": 5, "choice": 123},
        
        # Missing required fields
        {"x": 0.5, "y": 5},  # missing choice
        {"x": 0.5, "choice": "a"},  # missing y
        {"y": 5, "choice": "a"},  # missing x
        {},  # missing all
        
        # Extra fields (should be handled gracefully)
        {"x": 0.5, "y": 5, "choice": "a", "extra_field": "should_be_ignored"},
    ]
    
    validation_results = []
    for i, sample in enumerate(invalid_samples):
        try:
            # Test direct validation
            h = ValidTestHyperparameters(**sample)
            print(f"   Sample {i+1}: ❌ Should have failed validation but didn't: {sample}")
            validation_results.append({"sample": sample, "error": None, "unexpected_pass": True})
        except ValidationError as e:
            print(f"   Sample {i+1}: ✅ Correctly caught validation error: {sample}")
            print(f"              Error: {str(e).split('\\n')[0][:80]}...")
            validation_results.append({"sample": sample, "error": str(e), "unexpected_pass": False})
        except Exception as e:
            print(f"   Sample {i+1}: ⚠️  Unexpected error type: {type(e).__name__}: {e}")
            validation_results.append({"sample": sample, "error": str(e), "unexpected_pass": False})
    
    expected_failures = len(invalid_samples)
    actual_failures = sum(1 for r in validation_results if not r["unexpected_pass"])
    print(f"   Validation summary: {actual_failures}/{expected_failures} expected failures caught")
    
    # Test 3: Service-level error handling
    print("\n🔬 Test 3: Service-Level Error Handling")
    print("-" * 40)
    
    # Test unregistered problem
    try:
        result = hpo_service.sample('nonexistent_problem', n_samples=1)
        print(f"   ❌ Should have failed for nonexistent problem: {result}")
    except Exception as e:
        print(f"   ✅ Correctly handled nonexistent problem: {type(e).__name__}")
    
    # Test invalid training data
    try:
        # Mix of valid and invalid data
        x_mixed = [
            {"x": 0.5, "y": 5, "choice": "a"},  # Valid
            {"x": 2.0, "y": 5, "choice": "a"},  # Invalid x
            {"x": 0.0, "y": 5, "choice": "a"},  # Valid
        ]
        y_mixed = [1.0, 2.0, 3.0]
        
        result = hpo_service.add('validation_test', x_mixed, y_mixed)
        print(f"   ⚠️  Service accepted mixed valid/invalid data: {result}")
    except Exception as e:
        print(f"   ✅ Service correctly rejected mixed data: {type(e).__name__}: {e}")
    
    # Test mismatched x/y lengths
    try:
        x_valid = [{"x": 0.5, "y": 5, "choice": "a"}]
        y_mismatched = [1.0, 2.0, 3.0]  # Too many y values
        
        result = hpo_service.add('validation_test', x_valid, y_mismatched)
        print(f"   ❌ Should have failed for mismatched lengths: {result}")
    except Exception as e:
        print(f"   ✅ Correctly caught length mismatch: {type(e).__name__}")
    
    # Test 4: Edge cases
    print("\n🔬 Test 4: Edge Cases")
    print("-" * 40)
    
    # Empty data
    try:
        result = hpo_service.add('validation_test', [], [])
        print(f"   ⚠️  Service accepted empty data: {result}")
    except Exception as e:
        print(f"   ✅ Service correctly rejected empty data: {type(e).__name__}")
    
    # Single data point
    try:
        x_single = [{"x": 0.0, "y": 5, "choice": "a"}]
        y_single = [1.0]
        result = hpo_service.add('validation_test', x_single, y_single)
        print(f"   ✅ Service accepted single data point: {result['message']}")
    except Exception as e:
        print(f"   ⚠️  Service rejected single data point: {type(e).__name__}: {e}")
    
    # Very large values
    try:
        x_large = [{"x": 0.999999, "y": 10, "choice": "a"}] * 5  # At bounds
        y_large = [1e10] * 5  # Very large objective values
        result = hpo_service.add('validation_test', x_large, y_large)
        print(f"   ✅ Service handled large values: {result['message']}")
    except Exception as e:
        print(f"   ⚠️  Service failed with large values: {type(e).__name__}: {e}")
    
    # NaN and infinite values
    try:
        x_nan = [{"x": 0.5, "y": 5, "choice": "a"}] * 3
        y_nan = [1.0, float('nan'), float('inf')]
        result = hpo_service.add('validation_test', x_nan, y_nan)
        print(f"   ⚠️  Service accepted NaN/inf values: {result}")
    except Exception as e:
        print(f"   ✅ Service correctly rejected NaN/inf: {type(e).__name__}")
    
    # Test 5: Device and configuration errors
    print("\n🔬 Test 5: Device and Configuration Errors")
    print("-" * 40)
    
    # Invalid device
    try:
        invalid_hparams = BayesianHPOServiceConfig(device='invalid_device')
        invalid_service = HPOService(hparams=invalid_hparams)
        print(f"   ⚠️  Accepted invalid device configuration")
    except Exception as e:
        print(f"   ✅ Correctly rejected invalid device: {type(e).__name__}")
    
    # Invalid acquisition function
    try:
        invalid_hparams = BayesianHPOServiceConfig(acquisition_function='NonExistentAcquisition')
        invalid_service = HPOService(hparams=invalid_hparams)
        x_scheme = ValidTestHyperparameters.model_json_schema()
        invalid_service.register('test', x_scheme)
        # Add some data to trigger model building
        x_valid = [{"x": 0.5, "y": 5, "choice": "a"}] * 5
        y_valid = [1.0] * 5
        invalid_service.add('test', x_valid, y_valid)
        # Try to sample (this should fail)
        result = invalid_service.sample('test', n_samples=1)
        print(f"   ❌ Should have failed with invalid acquisition function: {result}")
    except Exception as e:
        print(f"   ✅ Correctly rejected invalid acquisition function: {type(e).__name__}")
    
    # Test 6: Recovery and suggestions
    print("\n🔬 Test 6: Recovery and Error Messages")
    print("-" * 40)
    
    # Test if error messages are helpful
    try:
        h = ValidTestHyperparameters(x=5.0, y=5, choice="a")  # x out of bounds
    except ValidationError as e:
        error_msg = str(e)
        print(f"   Error message quality check:")
        print(f"   ✅ Contains field name: {'x' in error_msg}")
        print(f"   ✅ Contains constraint info: {'ge=-1.0' in error_msg or 'le=1.0' in error_msg}")
        print(f"   ✅ Mentions validation: {'validation' in error_msg.lower()}")
        print(f"   Message: {error_msg.split(chr(10))[0][:100]}...")
    
    # Test sampling without training
    try:
        # Create fresh service
        fresh_service = HPOService(hparams=BayesianHPOServiceConfig(device='cpu'))
        x_scheme = ValidTestHyperparameters.model_json_schema()
        fresh_service.register('untrained_test', x_scheme)
        result = fresh_service.sample('untrained_test', n_samples=1)
        print(f"   ⚠️  Service allowed sampling without training: {result}")
    except Exception as e:
        error_msg = str(e)
        print(f"   ✅ Correctly prevented sampling without training")
        print(f"   Message helpful: {'train' in error_msg.lower() or 'model' in error_msg.lower()}")
    
    # Test 7: Stress testing
    print("\n🔬 Test 7: Stress Testing")
    print("-" * 40)
    
    try:
        # Many rapid requests
        x_valid = [{"x": 0.0, "y": 5, "choice": "a"}] * 20
        y_valid = list(range(20))
        hpo_service.add('validation_test', x_valid, y_valid)
        
        # Rapid sampling
        start_time = time.time()
        for i in range(5):
            result = hpo_service.sample('validation_test', n_samples=1)
            candidate = result['samples'][0]
            y = simple_objective(ValidTestHyperparameters(**candidate))
            hpo_service.add('validation_test', [candidate], [y])
        
        end_time = time.time()
        print(f"   ✅ Handled rapid requests: {end_time - start_time:.2f}s for 5 iterations")
        
    except Exception as e:
        print(f"   ⚠️  Failed stress test: {type(e).__name__}: {e}")
    
    print(f"\n✅ Error handling and validation test completed!")
    
    # Expected Outputs
    print(f"\n📋 Expected Test Outcomes:")
    print(f"   • Should catch all validation errors for invalid parameter values")
    print(f"   • Should provide helpful error messages with field names and constraints")
    print(f"   • Should handle service-level errors gracefully (nonexistent problems, etc.)")
    print(f"   • Should reject empty data, mismatched lengths, NaN/inf values")
    print(f"   • Should validate device and configuration parameters")
    print(f"   • Should prevent sampling from untrained models")
    print(f"   • Should handle rapid requests without crashing")
    print(f"   • Error messages should guide users toward fixing issues")


if __name__ == "__main__":
    test_error_handling_and_validation() 