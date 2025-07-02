#!/usr/bin/env python3
"""
Test Multi-Objective Optimization with BoTorch qEHVI and qNEHVI acquisition functions.
Expected to optimize two conflicting objectives: accuracy vs inference_speed.
"""

import numpy as np
import time
from typing import List, Tuple

from beam.bayesian import BayesianHPOServiceConfig, HPOService
from beam.bayesian.hp_scheme import BaseParameters
from pydantic import Field


class MultiObjectiveHyperparameters(BaseParameters):
    """Hyperparameters for multi-objective optimization."""
    learning_rate: float = Field(ge=1e-5, le=1e-1, description="Learning rate")
    hidden_size: int = Field(ge=32, le=512, description="Hidden layer size")
    n_layers: int = Field(ge=1, le=10, description="Number of layers")
    dropout: float = Field(ge=0.0, le=0.8, description="Dropout rate")
    optimizer: str = Field(description="Optimizer type", json_schema_extra={
        "enum": ["adam", "sgd", "adamw"]
    })
    

def multi_objective_function(h: MultiObjectiveHyperparameters) -> Tuple[float, float]:
    """
    Simulate ML model training with two conflicting objectives:
    1. Accuracy (higher is better) - maximization objective
    2. Inference Speed (higher is better, measured as 1/latency) - maximization objective
    
    These are conflicting because:
    - Larger models (more layers/hidden_size) → higher accuracy but slower inference
    - Higher dropout → better generalization but slower training
    - Learning rate affects both in complex ways
    """
    
    # Accuracy objective (0 to 100)
    # Favors: more layers, larger hidden_size, moderate dropout, adam optimizer, moderate lr
    base_accuracy = 60.0
    
    # Layer effect: more layers = higher accuracy but diminishing returns
    layer_effect = 8.0 * np.log(h.n_layers + 1)
    
    # Hidden size effect: larger = better accuracy
    hidden_effect = 15.0 * np.log(h.hidden_size / 32) / np.log(16)  # normalized log scale
    
    # Dropout effect: optimal around 0.3
    dropout_effect = 8.0 * (1 - abs(h.dropout - 0.3) / 0.3)
    
    # Optimizer effect
    opt_effects = {"adam": 5.0, "adamw": 4.0, "sgd": 2.0}
    opt_effect = opt_effects[h.optimizer]
    
    # Learning rate effect: optimal around 1e-3
    lr_effect = 5.0 * (1 - abs(np.log10(h.learning_rate) + 3) / 2)  # penalty for being far from 1e-3
    
    accuracy = base_accuracy + layer_effect + hidden_effect + dropout_effect + opt_effect + lr_effect
    
    # Add noise to make it realistic
    accuracy += np.random.normal(0, 2.0)
    accuracy = np.clip(accuracy, 0, 100)
    
    # Speed objective (measured as operations per second, higher is better)
    # Favors: fewer layers, smaller hidden_size, less dropout, simpler optimizers
    base_speed = 1000.0  # ops/sec
    
    # Model complexity penalty
    complexity_penalty = h.n_layers * h.hidden_size / 100.0
    speed = base_speed / (1 + complexity_penalty / 1000.0)
    
    # Dropout penalty (slower training)
    dropout_penalty = h.dropout * 200.0
    speed -= dropout_penalty
    
    # Optimizer speed differences
    opt_speed_effects = {"sgd": 1.2, "adam": 1.0, "adamw": 0.9}  # SGD is fastest
    speed *= opt_speed_effects[h.optimizer]
    
    # Learning rate affects convergence speed
    if h.learning_rate > 1e-2:  # Too high LR slows convergence
        speed *= 0.8
    elif h.learning_rate < 1e-4:  # Too low LR slows convergence  
        speed *= 0.9
    
    # Add noise
    speed += np.random.normal(0, 50.0)
    speed = np.clip(speed, 100, 2000)
    
    return float(accuracy), float(speed)


def test_multi_objective_optimization():
    """
    Test multi-objective Bayesian optimization.
    
    Expected behavior:
    1. Should find Pareto frontier with trade-offs between accuracy and speed
    2. qEHVI should work for 2 objectives  
    3. Should return multiple non-dominated solutions
    4. Solutions should show clear trade-offs (high accuracy + low speed vs low accuracy + high speed)
    """
    print("🎯 Testing Multi-Objective Bayesian Optimization")
    print("=" * 70)
    
    # Test both qEHVI and qNEHVI acquisition functions
    for acq_func in ['qEHVI', 'qNEHVI']:
        print(f"\n🔬 Testing {acq_func} acquisition function")
        print("-" * 50)
        
        # Initialize service with multi-objective config
        hparams = BayesianHPOServiceConfig(
            acquisition_function=acq_func,
            acquisition_kwargs={'ref_point': [50.0, 500.0]},  # Reference point for hypervolume
            device='cpu'
        )
        
        hpo_service = HPOService(hparams=hparams)
        print(f"✅ HPOService initialized with {acq_func}")
        
        # Register multi-objective problem
        x_scheme = MultiObjectiveHyperparameters.model_json_schema()
        result = hpo_service.register(f'moo_test_{acq_func.lower()}', x_scheme)
        print(f"✅ Problem registered: {result['message']}")
        
        # Generate initial data
        print("\n🎲 Generating initial training data...")
        n_init = 20
        x_init = []
        y_init = []  # Will be list of [accuracy, speed] pairs
        
        for _ in range(n_init):
            h = MultiObjectiveHyperparameters(
                learning_rate=np.random.uniform(1e-5, 1e-1),
                hidden_size=int(np.random.choice([32, 64, 128, 256, 512])),
                n_layers=np.random.randint(1, 11),
                dropout=np.random.uniform(0.0, 0.8),
                optimizer=np.random.choice(["adam", "sgd", "adamw"])
            )
            accuracy, speed = multi_objective_function(h)
            
            x_init.append(dict(h))
            y_init.append([accuracy, speed])  # Multi-objective targets
        
        print(f"📊 Generated {n_init} initial samples")
        print(f"📈 Accuracy range: [{min(y[0] for y in y_init):.1f}, {max(y[0] for y in y_init):.1f}]")
        print(f"⚡ Speed range: [{min(y[1] for y in y_init):.0f}, {max(y[1] for y in y_init):.0f}] ops/sec")
        
        # Train model
        print("\n🔄 Training multi-objective model...")
        start_time = time.time()
        result = hpo_service.add(f'moo_test_{acq_func.lower()}', x_init, y_init)
        end_time = time.time()
        print(f"✅ {result['message']} (took {end_time - start_time:.2f}s)")
        
        # Sample Pareto-optimal candidates
        print(f"\n🎯 Sampling Pareto-optimal candidates with {acq_func}...")
        start_time = time.time()
        result = hpo_service.sample(f'moo_test_{acq_func.lower()}', n_samples=6)
        end_time = time.time()
        print(f"✅ Generated {len(result['samples'])} candidates (took {end_time - start_time:.2f}s)")
        
        # Evaluate candidates and analyze Pareto frontier
        print(f"\n🏆 Pareto-optimal candidates from {acq_func}:")
        print("-" * 60)
        
        candidates_with_objectives = []
        for i, candidate in enumerate(result['samples']):
            h = MultiObjectiveHyperparameters(**candidate)
            accuracy, speed = multi_objective_function(h)
            candidates_with_objectives.append((candidate, accuracy, speed))
            
            print(f"Candidate {i+1}: Accuracy={accuracy:.1f}%, Speed={speed:.0f} ops/sec")
            print(f"  lr={candidate['learning_rate']:.2e}, hidden={candidate['hidden_size']}, "
                  f"layers={candidate['n_layers']}, dropout={candidate['dropout']:.2f}, opt={candidate['optimizer']}")
            print()
        
        # Analyze trade-offs
        accuracies = [acc for _, acc, _ in candidates_with_objectives]
        speeds = [spd for _, _, spd in candidates_with_objectives]
        
        print(f"📊 {acq_func} Performance Analysis:")
        print(f"   Accuracy range: [{min(accuracies):.1f}, {max(accuracies):.1f}]%")
        print(f"   Speed range: [{min(speeds):.0f}, {max(speeds):.0f}] ops/sec")
        print(f"   Trade-off span: {max(accuracies) - min(accuracies):.1f}% accuracy vs {max(speeds) - min(speeds):.0f} ops/sec speed")
        
        # Check for proper Pareto behavior
        high_acc_candidate = max(candidates_with_objectives, key=lambda x: x[1])
        high_speed_candidate = max(candidates_with_objectives, key=lambda x: x[2])
        
        print(f"\n🔍 Pareto Analysis:")
        print(f"   Best accuracy: {high_acc_candidate[1]:.1f}% (speed: {high_acc_candidate[2]:.0f})")
        print(f"   Best speed: {high_speed_candidate[2]:.0f} ops/sec (accuracy: {high_speed_candidate[1]:.1f}%)")
        
        # Expected: Should show trade-off (best accuracy candidate should have lower speed and vice versa)
        if high_acc_candidate[2] < high_speed_candidate[2]:
            print(f"   ✅ Proper trade-off detected: higher accuracy → lower speed")
        else:
            print(f"   ⚠️  Unexpected: best accuracy candidate is also fastest")
    
    print(f"\n✅ Multi-objective optimization test completed!")
    
    # Expected Outputs:
    print(f"\n📋 Expected Test Outcomes:")
    print(f"   • qEHVI and qNEHVI should both work without errors")
    print(f"   • Should generate diverse candidates with different accuracy/speed trade-offs")
    print(f"   • High accuracy candidates should generally have lower speed (trade-off)")
    print(f"   • Solutions should span a reasonable range of the Pareto frontier")
    print(f"   • Acquisition functions should find non-dominated solutions")


if __name__ == "__main__":
    test_multi_objective_optimization() 