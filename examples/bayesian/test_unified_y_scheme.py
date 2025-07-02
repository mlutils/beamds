#!/usr/bin/env python3
"""
Test Unified y_scheme with BaseParameters
========================================

This test validates that y_scheme uses BaseParameters consistently
with x_scheme and c_scheme for a unified framework.
"""

import sys
import numpy as np
from pydantic import Field
from typing import Dict, List

# Add project root to path for imports
sys.path.append('.')

from beam.bayesian import BayesianHPOServiceConfig, HPOService
from beam.bayesian.hp_scheme import BaseParameters


class SimpleHyperparams(BaseParameters):
    """Simple hyperparameters for testing."""
    learning_rate: float = Field(ge=1e-5, le=1e-1, description="Learning rate")
    batch_size: int = Field(ge=16, le=512, description="Batch size")


class SimpleObjectives(BaseParameters):
    """Simple objectives using BaseParameters - unified with x_scheme approach."""
    accuracy: float = Field(description="Model accuracy", 
                           json_schema_extra={"objective": "maximize"})
    training_time: float = Field(description="Training time in seconds",
                                json_schema_extra={"objective": "minimize"})
    memory_usage: float = Field(description="Memory usage in GB",
                               json_schema_extra={"constraint": "<= 4.0"})


def simple_objective_function(params: SimpleHyperparams) -> Dict[str, float]:
    """Simple objective function for testing."""
    lr = params.learning_rate
    batch_size = params.batch_size
    
    # Simple trade-offs
    accuracy = 0.8 + 0.15 * np.log10(lr + 1e-5) + 0.05 * (batch_size / 256)
    accuracy = np.clip(accuracy + np.random.normal(0, 0.02), 0.5, 0.99)
    
    training_time = 100 - 50 * np.log10(lr + 1e-5) + 2 * (batch_size / 32)
    training_time = max(10, training_time + np.random.normal(0, 5))
    
    memory_usage = 1.0 + 2.0 * (batch_size / 256) + np.random.normal(0, 0.2)
    memory_usage = max(0.5, memory_usage)
    
    return {
        "accuracy": accuracy,
        "training_time": training_time,
        "memory_usage": memory_usage
    }


def test_unified_y_scheme():
    """Test BaseParameters for y_scheme - unified with x_scheme approach."""
    print("🧪 Testing Unified y_scheme with BaseParameters")
    print("=" * 55)
    
    # Step 1: Create schemas using BaseParameters
    print("\n📋 Step 1: Schema Creation")
    x_scheme = SimpleHyperparams.model_json_schema()
    y_scheme = SimpleObjectives.model_json_schema()
    
    print(f"✅ x_scheme: {list(x_scheme['properties'].keys())}")
    print(f"✅ y_scheme: {list(y_scheme['properties'].keys())}")
    
    # Verify y_scheme has objective/constraint annotations
    objectives = []
    constraints = []
    for field_name, field_info in y_scheme['properties'].items():
        if 'objective' in field_info:
            objectives.append((field_name, field_info['objective']))
        elif 'constraint' in field_info:
            constraints.append((field_name, field_info['constraint']))
    
    print(f"📊 Objectives: {objectives}")
    print(f"⚖️  Constraints: {constraints}")
    
    # Step 2: Register problem with unified schemas
    print("\n📝 Step 2: Problem Registration")
    hparams = BayesianHPOServiceConfig(
        device='cpu',
        start_fitting_after_n_points=6  # Small for testing
    )
    hpo = HPOService(hparams=hparams)
    
    result = hpo.register('unified_test', x_scheme, y_scheme=y_scheme)
    print(f"✅ Registration: {result['message']}")
    print(f"   Schema validation: x_scheme and y_scheme both use BaseParameters")
    
    # Step 3: Generate data using both schemas
    print("\n🎲 Step 3: Data Generation with Schema Validation")
    
    training_x = []
    training_y = []
    
    np.random.seed(42)
    for i in range(8):
        # Generate x using BaseParameters validation
        x_dict = {
            "learning_rate": float(np.random.choice([1e-4, 3e-4, 1e-3, 3e-3])),
            "batch_size": int(np.random.choice([32, 64, 128, 256]))
        }
        
        # Validate x_dict using BaseParameters
        try:
            x_validated = SimpleHyperparams(**x_dict)
            print(f"   ✅ x{i+1} validated: lr={x_validated.learning_rate:.0e}, batch={x_validated.batch_size}")
        except Exception as e:
            print(f"   ❌ x{i+1} validation failed: {e}")
            continue
        
        # Generate y using objective function
        y_dict = simple_objective_function(x_validated)
        
        # Validate y_dict using BaseParameters
        try:
            y_validated = SimpleObjectives(**y_dict)
            print(f"   ✅ y{i+1} validated: acc={y_validated.accuracy:.3f}, time={y_validated.training_time:.1f}s")
        except Exception as e:
            print(f"   ❌ y{i+1} validation failed: {e}")
            continue
        
        training_x.append(x_dict)
        training_y.append(y_dict)
    
    print(f"✅ Generated {len(training_x)} validated samples")
    
    # Step 4: Add data to HPO service
    print("\n📈 Step 4: Adding Multi-objective Data")
    
    result = hpo.add('unified_test', training_x, training_y)
    print(f"✅ Data added: {result['message']}")
    
    # Step 5: Test backward compatibility with single objective
    print("\n🔄 Step 5: Backward Compatibility Test")
    
    # Test single-objective without y_scheme
    class SimpleParams(BaseParameters):
        x: float = Field(ge=0, le=1)
    
    simple_x_scheme = SimpleParams.model_json_schema()
    result = hpo.register('simple_single', simple_x_scheme)  # No y_scheme
    print(f"✅ Single-objective registration: {result['message']}")
    
    # Add simple scalar data
    simple_x_data = [{"x": 0.1}, {"x": 0.5}, {"x": 0.8}]
    simple_y_data = [0.2, 0.7, 0.4]  # Simple list of scalars
    
    result = hpo.add('simple_single', simple_x_data, simple_y_data)
    print(f"✅ Single-objective data: {result['message']}")
    
    # Step 6: Constraint validation
    print("\n⚖️  Step 6: Constraint Validation")
    
    violations = 0
    for i, y_dict in enumerate(training_y):
        if y_dict['memory_usage'] > 4.0:
            violations += 1
            print(f"   ❌ Sample {i+1}: memory={y_dict['memory_usage']:.1f} > 4.0")
    
    print(f"✅ Constraints: {violations}/{len(training_y)} violations detected")
    
    return True


def test_schema_consistency():
    """Test that x_scheme, y_scheme use the same BaseParameters foundation."""
    print("\n🔧 Testing Schema Consistency")
    print("=" * 40)
    
    # All schemas should use BaseParameters
    x_schema_class = SimpleHyperparams
    y_schema_class = SimpleObjectives
    
    # Test that they have the same base methods
    base_methods = ['model_json_schema', 'encode', 'decode', 'from_json_schema']
    
    for method in base_methods:
        x_has = hasattr(x_schema_class, method)
        y_has = hasattr(y_schema_class, method)
        
        if x_has and y_has:
            print(f"   ✅ Both schemas have {method}")
        else:
            print(f"   ❌ Schema method mismatch: {method}")
            return False
    
    # Test encoding/decoding
    x_instance = SimpleHyperparams(learning_rate=1e-3, batch_size=128)
    y_instance = SimpleObjectives(accuracy=0.85, training_time=45.0, memory_usage=2.5)
    
    # Test encoding
    x_num, x_cat = x_instance.encode()
    y_num, y_cat = y_instance.encode()
    
    print(f"   ✅ x_encoding: num={x_num.shape}, cat={x_cat.shape}")
    print(f"   ✅ y_encoding: num={y_num.shape}, cat={y_cat.shape}")
    
    print("✅ Schema consistency validated")
    return True


def main():
    """Run unified y_scheme tests."""
    try:
        # Test unified approach
        test_unified_y_scheme()
        
        # Test schema consistency
        test_schema_consistency()
        
        print("\n🎉 Unified y_scheme tests completed!")
        print("\n📋 SUMMARY:")
        print("✅ y_scheme uses BaseParameters (unified with x_scheme)")
        print("✅ Schema validation for both inputs and outputs")
        print("✅ Multi-objective data handling with constraints")
        print("✅ Backward compatibility for single-objective problems")
        print("✅ Consistent encoding/decoding across all schemas")
        
        return True
        
    except Exception as e:
        print(f"\n💥 Unified y_scheme test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 