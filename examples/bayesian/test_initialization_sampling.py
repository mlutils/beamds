#!/usr/bin/env python3
"""
Test initialization sampling methods for Bayesian optimization
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from typing import Literal
from pydantic import BaseModel, Field
from beam.bayesian import HPOService

def test_initialization_sampling():
    print("🧪 Testing Initialization Sampling Methods")
    print("=" * 50)
    
    # Create a comprehensive hyperparameter schema with different types
    class TestHyperparameters(BaseModel):
        # Numerical with bounds
        learning_rate: float = Field(ge=1e-5, le=1e-1)
        dropout: float = Field(ge=0.0, le=0.9)
        
        # Numerical without bounds (will use default 0-1 range) 
        temperature: float
        
        # Categorical
        optimizer: Literal["adam", "sgd", "rmsprop", "adamw"]
        activation: Literal["relu", "gelu", "tanh", "sigmoid"]
    
    x_scheme = TestHyperparameters.model_json_schema()
    
    # Test different initialization methods
    methods = ['uniform', 'sobol', 'halton', 'random']
    n_samples = 8
    
    for method in methods:
        print(f"\n🎲 Testing {method.upper()} sampling:")
        print("-" * 30)
        
        # Initialize service with the specific method
        hpo_service = HPOService(device='cpu', initialization_method=method)
        
        # Register problem
        result = hpo_service.register('test_init', x_scheme)
        print(f"✅ {result['message']}")
        
        # Try to sample - this will automatically trigger initialization
        try:
            result = hpo_service.sample('test_init', n_samples=5)
            
            if result['method'] == 'initialize':
                samples = result['samples']
                print(f"✅ Generated {len(samples)} samples using {method}")
                
                # Display first few samples
                for i, sample in enumerate(samples[:3]):
                    print(f"   Sample {i+1}: {sample}")
                
                # Validate bounds
                valid = True
                for sample in samples:
                    # Check learning_rate bounds
                    if not (1e-5 <= sample['learning_rate'] <= 1e-1):
                        print(f"❌ learning_rate out of bounds: {sample['learning_rate']}")
                        valid = False
                    
                    # Check dropout bounds
                    if not (0.0 <= sample['dropout'] <= 0.9):
                        print(f"❌ dropout out of bounds: {sample['dropout']}")
                        valid = False
                    
                    # Check temperature (should be in default 0-1 range)
                    if not (0.0 <= sample['temperature'] <= 1.0):
                        print(f"❌ temperature out of default bounds: {sample['temperature']}")
                        valid = False
                    
                    # Check categorical values
                    valid_optimizers = ["adam", "sgd", "rmsprop", "adamw"]
                    valid_activations = ["relu", "gelu", "tanh", "sigmoid"]
                    
                    if sample['optimizer'] not in valid_optimizers:
                        print(f"❌ invalid optimizer: {sample['optimizer']}")
                        valid = False
                    
                    if sample['activation'] not in valid_activations:
                        print(f"❌ invalid activation: {sample['activation']}")
                        valid = False
                
                if valid:
                    print(f"✅ All samples are valid for {method}")
                else:
                    print(f"❌ Some samples are invalid for {method}")
            else:
                print(f"🎯 Got optimized samples instead of initialization")
                
        except Exception as e:
            print(f"❌ Error in {method} sampling: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Initialization sampling test completed!")

def test_mixed_optimization_with_initialization():
    """Test that initialization works in the full optimization workflow."""
    print("\n" + "=" * 60)
    print("🚀 Testing Mixed Optimization with Auto-Initialization")
    print("=" * 60)
    
    from typing import Literal
    from pydantic import BaseModel, Field
    
    class MixedHyperparameters(BaseModel):
        lr: float = Field(ge=1e-4, le=1e-1)
        weight_decay: float = Field(ge=1e-6, le=1e-2)
        optimizer: Literal["adam", "sgd"]
        scheduler: Literal["cosine", "step"]
    
    x_scheme = MixedHyperparameters.model_json_schema()
    
    # Initialize with Sobol sampling
    hpo_service = HPOService(
        device='cpu', 
        initialization_method='sobol',
        start_fitting_after_n_points=15  # Set higher to test initialization
    )
    
    result = hpo_service.register('mixed_test', x_scheme)
    print(f"✅ {result['message']}")
    
    # Try to sample - this will automatically trigger initialization
    result = hpo_service.sample('mixed_test', n_samples=5)
    
    if result['method'] == 'initialize':
        init_samples = result['samples']
        print(f"📊 Generated {len(init_samples)} initial samples:")
        for i, sample in enumerate(init_samples[:5]):
            print(f"   {i+1}. lr={sample['lr']:.6f}, wd={sample['weight_decay']:.7f}, "
                  f"opt={sample['optimizer']}, sched={sample['scheduler']}")
        
        # Test objective function
        def objective(params):
            score = 0.0
            # Prefer lower learning rates
            if params['lr'] < 0.01:
                score += 1.0
            # Prefer adam optimizer
            if params['optimizer'] == 'adam':
                score += 0.5
            # Prefer cosine scheduler
            if params['scheduler'] == 'cosine':
                score += 0.3
            
            import random
            return score + random.uniform(-0.1, 0.1)
        
        # Evaluate initial samples
        y_init = [objective(sample) for sample in init_samples]
        
        # Add to service
        result = hpo_service.add('mixed_test', init_samples, y_init)
        print(f"✅ {result['message']}")
        
        # Try to get more samples - might need more for training threshold
        for _ in range(3):  # Try a few times to accumulate enough samples
            sample_result = hpo_service.sample('mixed_test', n_samples=5)
            if sample_result['method'] == 'optimize':
                print(f"✅ Generated {len(sample_result['samples'])} optimized candidates")
                print(f"\n🏆 Top optimized candidates:")
                for i, candidate in enumerate(sample_result['samples']):
                    score = objective(candidate)
                    print(f"   {i+1}. {candidate} (score: {score:.3f})")
                break
            elif sample_result['method'] == 'initialize':
                # Still need more samples
                extra_samples = sample_result['samples']
                y_extra = [objective(sample) for sample in extra_samples]
                hpo_service.add('mixed_test', extra_samples, y_extra)
                print(f"✅ Added {len(extra_samples)} more samples")
            else:
                print(f"📝 Unexpected result: {sample_result}")
                break
    else:
        print(f"🎯 Got optimized samples directly: {result['samples']}")
    
    print(f"\n✅ Mixed optimization with initialization completed!")

if __name__ == "__main__":
    test_initialization_sampling()
    test_mixed_optimization_with_initialization() 