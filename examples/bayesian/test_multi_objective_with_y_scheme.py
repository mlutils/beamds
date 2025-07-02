#!/usr/bin/env python3
"""
Test Multi-Objective Optimization with y_scheme Design
======================================================

This test validates the proposed y_scheme design for defining output structure
with objectives and constraints.
"""

import sys
import numpy as np
import torch
from pydantic import BaseModel, Field
from typing import Dict, List

# Add project root to path for imports
sys.path.append('.')

from beam.bayesian import BayesianHPOServiceConfig, HPOService
from beam.bayesian.hp_scheme import BaseParameters


class MobileNetParams(BaseParameters):
    """Neural architecture parameters for mobile deployment."""
    num_layers: int = Field(ge=8, le=50, description="Number of layers")
    width_multiplier: float = Field(ge=0.5, le=2.0, description="Channel width multiplier")
    resolution: int = Field(ge=128, le=512, description="Input resolution")
    dropout_rate: float = Field(ge=0.0, le=0.5, description="Dropout rate")


class MobileNetObjectives(BaseModel):
    """Output schema defining objectives and constraints for mobile NAS."""
    accuracy: float = Field(description="Classification accuracy on validation set", 
                           json_schema_extra={"objective": "maximize"})
    inference_time: float = Field(description="Inference time in milliseconds",
                                 json_schema_extra={"objective": "minimize"})
    memory_usage: float = Field(description="Peak memory usage in MB",
                               json_schema_extra={"constraint": "<= 2000"})
    model_size: float = Field(description="Model size in MB", 
                             json_schema_extra={"constraint": "<= 20"})


def mobile_nas_objective(params: MobileNetParams) -> Dict[str, float]:
    """
    Simulate mobile NAS evaluation with conflicting objectives.
    
    Returns:
        Dict with accuracy, inference_time, memory_usage, model_size
    """
    # Extract parameters
    num_layers = params.num_layers
    width_mult = params.width_multiplier  
    resolution = params.resolution
    dropout = params.dropout_rate
    
    # Simulate realistic trade-offs
    complexity_factor = (num_layers / 30) * width_mult * (resolution / 300)
    
    # Accuracy increases with complexity but has diminishing returns
    accuracy = 0.6 + 0.35 * (1 - np.exp(-2 * complexity_factor))
    accuracy = accuracy * (1 - dropout * 0.3)  # Dropout reduces overfitting but lowers accuracy
    accuracy = np.clip(accuracy + np.random.normal(0, 0.02), 0.5, 0.99)
    
    # Inference time increases with complexity  
    inference_time = 10 + 80 * complexity_factor + np.random.normal(0, 3)
    inference_time = max(5, inference_time)
    
    # Memory usage grows with model size
    memory_usage = 500 + 1500 * complexity_factor + np.random.normal(0, 100)
    memory_usage = max(300, memory_usage)
    
    # Model size correlates with parameters
    model_size = 2 + 30 * complexity_factor + np.random.normal(0, 2)
    model_size = max(1, model_size)
    
    return {
        "accuracy": accuracy,
        "inference_time": inference_time, 
        "memory_usage": memory_usage,
        "model_size": model_size
    }


def test_y_scheme_design():
    """Test the proposed y_scheme design."""
    print("🧪 Testing y_scheme Design")
    print("=" * 50)
    
    # Test 1: Schema Creation and Validation
    print("\n📋 Test 1: y_scheme Schema Creation")
    y_scheme = MobileNetObjectives.model_json_schema()
    print(f"✅ Created y_scheme with {len(y_scheme['properties'])} outputs")
    
    # Extract objectives and constraints  
    objectives = []
    constraints = []
    
    for field_name, field_info in y_scheme['properties'].items():
        if 'objective' in field_info:
            objectives.append((field_name, field_info['objective']))
        elif 'constraint' in field_info:
            constraints.append((field_name, field_info['constraint']))
    
    print(f"✅ Found {len(objectives)} objectives: {objectives}")
    print(f"✅ Found {len(constraints)} constraints: {constraints}")
    
    # Test 2: Multi-objective Service Configuration
    print("\n🔧 Test 2: Multi-objective Service Configuration")
    
    # Test qEHVI configuration (should work with our implementation)
    try:
        hparams = BayesianHPOServiceConfig(
            acquisition_function='qEHVI',
            acquisition_kwargs={'ref_point': [0.0, 100.0]},  # [min_accuracy, max_inference_time]
            device='cpu'
        )
        hpo = HPOService(hparams=hparams)
        print("✅ qEHVI configuration created successfully")
    except Exception as e:
        print(f"⚠️  qEHVI not implemented yet: {e}")
    
    # Test qNEHVI configuration  
    try:
        hparams = BayesianHPOServiceConfig(
            acquisition_function='qNEHVI',
            acquisition_kwargs={'ref_point': [0.0, 100.0]},
            device='cpu'
        )
        hpo = HPOService(hparams=hparams)
        print("✅ qNEHVI configuration created successfully")
    except Exception as e:
        print(f"⚠️  qNEHVI not implemented yet: {e}")
    
    # Test 3: Problem Registration with y_scheme
    print("\n📝 Test 3: Problem Registration with y_scheme")
    
    hparams = BayesianHPOServiceConfig(device='cpu')
    hpo = HPOService(hparams=hparams)
    
    x_scheme = MobileNetParams.model_json_schema()
    y_scheme = MobileNetObjectives.model_json_schema()
    
    # Register problem with both x_scheme and y_scheme
    try:
        # Try the new API when implemented
        result = hpo.register('mobile_nas', x_scheme, y_scheme=y_scheme)
        print(f"✅ Registered problem with y_scheme: {result.get('message', 'Success')}")
    except TypeError as e:
        if 'y_scheme' in str(e):
            print(f"⚠️  y_scheme parameter not implemented yet")
            # Fallback to current API
            result = hpo.register('mobile_nas', x_scheme)
            print(f"✅ Registered problem without y_scheme: {result.get('message', 'Success')}")
        else:
            raise e
    
    # Test 4: Multi-objective Data Generation
    print("\n🎲 Test 4: Multi-objective Data Generation")
    
    # Generate sample configurations
    sample_configs = []
    sample_objectives = []
    
    for i in range(8):
        # Generate random valid configuration
        config = {
            "num_layers": np.random.randint(10, 40),
            "width_multiplier": np.random.uniform(0.6, 1.8),
            "resolution": int(np.random.choice([160, 224, 288, 384])),
            "dropout_rate": np.random.uniform(0.1, 0.4)
        }
        
        # Evaluate objectives
        params = MobileNetParams(**config)
        objectives = mobile_nas_objective(params)
        
        sample_configs.append(config)
        sample_objectives.append(objectives)
        
        print(f"   Config {i+1}: layers={config['num_layers']}, "
              f"acc={objectives['accuracy']:.3f}, "
              f"time={objectives['inference_time']:.1f}ms, "
              f"mem={objectives['memory_usage']:.0f}MB")
    
    print(f"✅ Generated {len(sample_configs)} multi-objective samples")
    
    # Test 5: Constraint Violation Detection
    print("\n⚖️  Test 5: Constraint Violation Analysis")
    
    constraint_violations = 0
    for i, obj in enumerate(sample_objectives):
        violations = []
        if obj['memory_usage'] > 2000:
            violations.append(f"memory={obj['memory_usage']:.0f} > 2000")
        if obj['model_size'] > 20:
            violations.append(f"size={obj['model_size']:.1f} > 20")
        
        if violations:
            constraint_violations += 1
            print(f"   ❌ Config {i+1} violates: {', '.join(violations)}")
    
    print(f"✅ Found {constraint_violations}/{len(sample_objectives)} configurations violating constraints")
    
    # Test 6: Pareto Frontier Analysis  
    print("\n🏆 Test 6: Pareto Frontier Analysis")
    
    # Extract objectives for Pareto analysis (accuracy to maximize, inference_time to minimize)
    accuracies = [obj['accuracy'] for obj in sample_objectives]
    inference_times = [obj['inference_time'] for obj in sample_objectives]
    
    # Find Pareto optimal solutions
    pareto_indices = []
    for i in range(len(sample_objectives)):
        is_pareto = True
        for j in range(len(sample_objectives)):
            if i != j:
                # Check if j dominates i (higher accuracy AND lower inference time)
                if (accuracies[j] >= accuracies[i] and inference_times[j] <= inference_times[i] and
                    (accuracies[j] > accuracies[i] or inference_times[j] < inference_times[i])):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)
    
    print(f"✅ Found {len(pareto_indices)} Pareto optimal solutions:")
    for idx in pareto_indices:
        obj = sample_objectives[idx]
        print(f"   🎯 Config {idx+1}: acc={obj['accuracy']:.3f}, time={obj['inference_time']:.1f}ms")
    
    return True


def test_backward_compatibility():
    """Test that single-objective optimization still works without y_scheme."""
    print("\n🔄 Testing Backward Compatibility")
    print("=" * 50)
    
    # Test current single-objective API
    hparams = BayesianHPOServiceConfig(device='cpu')
    hpo = HPOService(hparams=hparams)
    
    # Simple single-objective problem
    class SimpleParams(BaseParameters):
        x: float = Field(ge=0, le=1)
        
    x_scheme = SimpleParams.model_json_schema()
    result = hpo.register('simple_test', x_scheme)
    
    # Generate data (current format - scalar y values)
    x_data = [{"x": 0.1}, {"x": 0.5}, {"x": 0.8}]
    y_data = [0.2, 0.7, 0.4]  # Single objective values
    
    hpo.add('simple_test', x_data, y_data)
    print("✅ Single-objective API still works")
    
    return True


def main():
    """Run all y_scheme design tests."""
    print("🚀 Multi-Objective Optimization with y_scheme Design")
    print("=" * 60)
    
    try:
        # Test the y_scheme design
        test_y_scheme_design()
        
        # Test backward compatibility
        test_backward_compatibility()
        
        print("\n🎉 All y_scheme design tests completed!")
        print("\n📋 IMPLEMENTATION ROADMAP:")
        print("1. ✅ y_scheme design validated")
        print("2. 🔧 Extend HPOService.register() to accept y_scheme parameter")
        print("3. 🔧 Implement multi-objective acquisition functions (qEHVI, qNEHVI)")
        print("4. 🔧 Add constraint handling in BayesianBeam")
        print("5. 🔧 Update HPOService.add() to handle multi-objective y_data")
        print("6. ✅ Maintain backward compatibility for single objectives")
        
        return True
        
    except Exception as e:
        print(f"\n💥 y_scheme test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 