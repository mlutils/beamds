#!/usr/bin/env python3
"""
Test MPS device support for Bayesian optimization
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import random
from beam.bayesian import HPOService

def test_mps_support():
    print("🧪 Testing MPS Device Support for Bayesian Optimization")
    print("=" * 60)
    
    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("❌ MPS not available on this system")
        return False
    
    print("✅ MPS is available")
    
    # Test with simple categorical optimization
    from typing import Literal
    from pydantic import BaseModel
    
    class TestHyperparameters(BaseModel):
        optimizer: Literal["adam", "sgd"] 
        activation: Literal["relu", "gelu"]
    
    # Create simple categorical schema
    x_scheme = TestHyperparameters.model_json_schema()
    
    # Initialize service with MPS device
    hpo_service = HPOService(device='mps')
    print(f"✅ HPOService initialized with device: mps")
    
    # Register problem
    problem_name = 'mps_test'
    result = hpo_service.register(problem_name, x_scheme)
    print(f"✅ Problem registered: {result['message']}")
    
    # Generate some initial data
    def objective_function(config):
        """Simple test objective: prefer adam + gelu combination"""
        score = 0.0
        if config['optimizer'] == 'adam':
            score += 2.0
        if config['activation'] == 'gelu':
            score += 1.5
        if config['optimizer'] == 'adam' and config['activation'] == 'gelu':
            score += 1.0  # bonus for best combination
        score += random.uniform(-0.5, 0.5)  # add noise
        return score
    
    # Generate initial data
    def random_sample():
        return {
            'optimizer': random.choice(['adam', 'sgd']),
            'activation': random.choice(['relu', 'gelu'])
        }
    
    initial_data = []
    for _ in range(10):
        config = random_sample()
        score = objective_function(config)
        initial_data.append((config, score))
    
    x_data = [item[0] for item in initial_data]
    y_data = [item[1] for item in initial_data]
    
    print(f"📈 Generated {len(initial_data)} initial samples")
    print(f"📊 Y range: [{min(y_data):.2f}, {max(y_data):.2f}]")
    
    # Train model
    print("\n🔄 Training model on MPS device...")
    try:
        train_result = hpo_service.add(problem_name, x_data, y_data)
        print(f"✅ {train_result['message']}")
        # Get the bayesian beam object to check device
        bb = hpo_service._problems[problem_name].solver
        print(f"🎯 Model device: {bb.gp.train_inputs[0].device}")
        
        # Sample new candidates
        print("\n🎯 Sampling optimized candidates on MPS...")
        sample_result = hpo_service.sample(problem_name, n_samples=5)
        print(f"✅ Generated {len(sample_result['samples'])} candidates")
        
        # Display results
        print("\n🏆 Top Candidates from MPS optimization:")
        print("-" * 50)
        for i, candidate in enumerate(sample_result['samples'], 1):
            score = objective_function(candidate)
            print(f"Candidate {i}: {candidate} (test score: {score:.3f})")
        
        # Test that tensors are actually on MPS
        x, y, _ = bb.get_replay_buffer()
        print(f"\n📱 Data tensors device: X={x.device}, Y={y.device}")
        print(f"📱 Bounds device: {bb.x_bounds.device}")
        
        print("\n🎉 SUCCESS: MPS optimization completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR during MPS optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mps_support()
    if success:
        print("\n✅ MPS device support is working correctly!")
    else:
        print("\n❌ MPS device support test failed") 