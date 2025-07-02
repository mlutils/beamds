#!/usr/bin/env python3
"""
Test automatic initialization in sample() method
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import Literal
from pydantic import BaseModel, Field
from beam.bayesian import HPOService

def test_automatic_initialization():
    print("🧪 Testing Automatic Initialization in sample() Method")
    print("=" * 55)
    
    # Create a simple hyperparameter schema
    class SimpleHyperparameters(BaseModel):
        learning_rate: float = Field(ge=1e-4, le=1e-1)
        optimizer: Literal["adam", "sgd"]
    
    x_scheme = SimpleHyperparameters.model_json_schema()
    
    # Initialize service with sobol initialization
    hpo_service = HPOService(
        device='cpu',
        initialization_method='sobol',
        start_fitting_after_n_points=8  # Need 8 samples before BO
    )
    
    # Register problem
    result = hpo_service.register('auto_init_test', x_scheme)
    print(f"✅ {result['message']}")
    
    # First call to sample() - should trigger initialization
    print(f"\n🎲 First call to sample() (should trigger initialization):")
    result1 = hpo_service.sample('auto_init_test', n_samples=3)
    print(f"📊 Method: {result1['method']}")
    print(f"📊 Message: {result1['message']}")
    print(f"📊 Samples received: {len(result1['samples'])}")
    
    if result1['method'] == 'initialize':
        print(f"✅ Automatic initialization worked!")
        samples1 = result1['samples']
        
        # Show samples
        for i, sample in enumerate(samples1):
            print(f"   Sample {i+1}: lr={sample['learning_rate']:.6f}, opt={sample['optimizer']}")
        
        # Define simple objective
        def objective(params):
            score = 0.0
            if params['learning_rate'] < 1e-2:  # Prefer smaller LR
                score += 1.0
            if params['optimizer'] == 'adam':    # Prefer adam
                score += 0.5
            import random
            return score + random.uniform(-0.1, 0.1)
        
        # Evaluate and add samples
        y1 = [objective(sample) for sample in samples1]
        add_result = hpo_service.add('auto_init_test', samples1, y1)
        print(f"✅ Added samples: {add_result['message']}")
        
        # Call sample() again - might still need more samples
        print(f"\n🎲 Second call to sample():")
        result2 = hpo_service.sample('auto_init_test', n_samples=3)
        print(f"📊 Method: {result2['method']}")
        print(f"📊 Message: {result2['message']}")
        print(f"📊 Samples received: {len(result2['samples'])}")
        
        if result2['method'] == 'initialize':
            print(f"✅ Still initializing - need more samples")
            samples2 = result2['samples']
            y2 = [objective(sample) for sample in samples2]
            add_result = hpo_service.add('auto_init_test', samples2, y2)
            print(f"✅ Added more samples: {add_result['message']}")
            
            # Try one more time
            print(f"\n🎲 Third call to sample():")
            result3 = hpo_service.sample('auto_init_test', n_samples=3)
            print(f"📊 Method: {result3['method']}")
            print(f"📊 Message: {result3['message']}")
            print(f"📊 Samples received: {len(result3['samples'])}")
            
            if result3['method'] == 'optimize':
                print(f"🎉 Now doing Bayesian optimization!")
                for i, candidate in enumerate(result3['samples']):
                    score = objective(candidate)
                    print(f"   Optimized {i+1}: lr={candidate['learning_rate']:.6f}, "
                          f"opt={candidate['optimizer']} (score: {score:.3f})")
            else:
                print(f"📝 Still initializing: {result3}")
                
        elif result2['method'] == 'optimize':
            print(f"🎉 Already doing Bayesian optimization!")
            for i, candidate in enumerate(result2['samples']):
                score = objective(candidate)
                print(f"   Optimized {i+1}: lr={candidate['learning_rate']:.6f}, "
                      f"opt={candidate['optimizer']} (score: {score:.3f})")
        
    else:
        print(f"🤔 Unexpected result: {result1}")

def test_different_initialization_methods():
    """Test different initialization methods automatically trigger"""
    print(f"\n" + "=" * 55)
    print(f"🔄 Testing Different Initialization Methods")
    print("=" * 55)
    
    class TestHyperparameters(BaseModel):
        x: float = Field(ge=0.0, le=1.0)
        choice: Literal["A", "B", "C"]
    
    x_scheme = TestHyperparameters.model_json_schema()
    
    methods = ['uniform', 'sobol', 'halton', 'random']
    
    for method in methods:
        print(f"\n🎲 Testing {method.upper()} initialization:")
        
        hpo_service = HPOService(
            device='cpu',
            initialization_method=method,
            start_fitting_after_n_points=6
        )
        
        problem_name = f'test_{method}'
        result = hpo_service.register(problem_name, x_scheme)
        
        # Sample should trigger initialization
        sample_result = hpo_service.sample(problem_name, n_samples=2)
        
        if sample_result['method'] == 'initialize':
            print(f"✅ {method} initialization triggered automatically")
            print(f"   Generated {len(sample_result['samples'])} samples")
            print(f"   Method used: {sample_result.get('initialization_method', 'unknown')}")
            
            # Show first sample
            sample = sample_result['samples'][0]
            print(f"   Sample: x={sample['x']:.4f}, choice={sample['choice']}")
        else:
            print(f"❌ Expected initialization but got: {sample_result['method']}")

if __name__ == "__main__":
    test_automatic_initialization()
    test_different_initialization_methods()
    print(f"\n🎉 Automatic initialization tests completed!") 