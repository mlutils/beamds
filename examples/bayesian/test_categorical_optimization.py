#!/usr/bin/env python3
"""
Categorical Optimization Test

This script tests Bayesian optimization with purely categorical variables,
which is a common challenge for optimizers.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import numpy as np
import time
from pydantic import BaseModel
from typing import Literal

from beam.bayesian import HPOService, BayesianHPOServiceConfig

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class CategoricalHyperparameters(BaseModel):
    """Pure categorical hyperparameters to test categorical optimization."""
    optimizer: Literal["adam", "sgd", "rmsprop", "adamw"]
    activation: Literal["relu", "gelu", "tanh", "sigmoid"]
    loss_function: Literal["mse", "mae", "huber", "cross_entropy"]
    scheduler: Literal["cosine", "step", "exponential", "plateau"]

def categorical_objective(h: CategoricalHyperparameters) -> float:
    """
    Objective function with clear categorical preferences.
    Optimal combination: adam + gelu + mse + cosine = high score
    """
    
    # Optimizer scores
    optimizer_scores = {
        "adam": 1.0,      # Best
        "adamw": 0.8,     # Second best
        "sgd": 0.4,       # Third
        "rmsprop": 0.2    # Worst
    }
    
    # Activation scores
    activation_scores = {
        "gelu": 1.0,      # Best
        "relu": 0.7,      # Second best  
        "tanh": 0.3,      # Third
        "sigmoid": 0.1    # Worst
    }
    
    # Loss function scores
    loss_scores = {
        "mse": 1.0,           # Best
        "mae": 0.6,           # Second
        "huber": 0.4,         # Third
        "cross_entropy": 0.2  # Worst
    }
    
    # Scheduler scores
    scheduler_scores = {
        "cosine": 1.0,      # Best
        "step": 0.6,        # Second
        "exponential": 0.4, # Third
        "plateau": 0.2      # Worst
    }
    
    # Compute total score with interaction effects
    base_score = (
        2.0 * optimizer_scores[h.optimizer] +
        1.5 * activation_scores[h.activation] +
        1.2 * loss_scores[h.loss_function] +
        1.0 * scheduler_scores[h.scheduler]
    )
    
    # Add interaction bonus for optimal combination
    if (h.optimizer == "adam" and h.activation == "gelu" and 
        h.loss_function == "mse" and h.scheduler == "cosine"):
        base_score += 2.0  # Big bonus for perfect combination
    
    # Add some noise to make it realistic
    noise = np.random.normal(0, 0.1)
    return base_score + noise

def random_categorical_sample() -> CategoricalHyperparameters:
    """Generate random categorical hyperparameters."""
    return CategoricalHyperparameters(
        optimizer=random.choice(["adam", "sgd", "rmsprop", "adamw"]),
        activation=random.choice(["relu", "gelu", "tanh", "sigmoid"]),
        loss_function=random.choice(["mse", "mae", "huber", "cross_entropy"]),
        scheduler=random.choice(["cosine", "step", "exponential", "plateau"])
    )

def run_categorical_test():
    """Test categorical optimization."""
    print("🧪 Testing Categorical Bayesian Optimization")
    print("=" * 50)
    
    # Initialize service
    hparams = BayesianHPOServiceConfig()
    bs = HPOService(hparams=hparams)
    print("✅ HPOService initialized")
    
    # Register problem (no context scheme)
    x_scheme = CategoricalHyperparameters.model_json_schema()
    result = bs.register('categorical_opt', x_scheme)
    print(f"✅ Problem registered: {result['message']}")
    
    # Generate initial data
    print(f"\n🎲 Generating initial data...")
    n_init = 30
    x_init = [dict(random_categorical_sample()) for _ in range(n_init)]
    y_init = [categorical_objective(CategoricalHyperparameters(**xi)) for xi in x_init]
    
    print(f"📈 Initial data: {n_init} samples")
    print(f"📊 Y range: [{min(y_init):.2f}, {max(y_init):.2f}]")
    
    # Find theoretical optimum for comparison
    optimal = CategoricalHyperparameters(
        optimizer="adam", activation="gelu", 
        loss_function="mse", scheduler="cosine"
    )
    theoretical_best = categorical_objective(optimal)
    print(f"🎯 Theoretical optimum: {theoretical_best:.2f}")
    
    # Train model
    print(f"\n🔄 Training model...")
    start_time = time.time()
    result = bs.add('categorical_opt', x_init, y_init)
    end_time = time.time()
    print(f"✅ {result['message']} (took {end_time - start_time:.2f}s)")
    
    # Sample candidates
    print(f"\n🎯 Sampling optimized candidates...")
    start_time = time.time()
    result = bs.sample('categorical_opt', n_samples=10)
    end_time = time.time()
    print(f"✅ Generated {len(result['samples'])} candidates (took {end_time - start_time:.2f}s)")
    
    # Evaluate candidates
    print(f"\n🏆 Top Candidates:")
    print("-" * 60)
    
    candidates_with_scores = []
    for i, candidate in enumerate(result['samples']):
        h = CategoricalHyperparameters(**candidate)
        score = categorical_objective(h)
        candidates_with_scores.append((candidate, score))
        print(f"Candidate {i+1} (score: {score:.3f}): {candidate}")
    
    # Find best candidate
    best_candidate, best_score = max(candidates_with_scores, key=lambda x: x[1])
    print(f"\n🥇 Best candidate found (score: {best_score:.3f}):")
    for key, value in best_candidate.items():
        print(f"   {key}: {value}")
    
    # Check if we found the optimum
    is_optimal = (best_candidate['optimizer'] == 'adam' and 
                  best_candidate['activation'] == 'gelu' and
                  best_candidate['loss_function'] == 'mse' and
                  best_candidate['scheduler'] == 'cosine')
    
    if is_optimal:
        print("🎉 SUCCESS: Found the theoretical optimum!")
    else:
        print("🤔 Did not find theoretical optimum, but that's normal with noise")
    
    print(f"📊 Gap from theoretical best: {theoretical_best - best_score:.3f}")
    
    # Test with more data
    print(f"\n🔄 Adding more data to improve optimization...")
    x_new = [dict(random_categorical_sample()) for _ in range(20)]
    y_new = [categorical_objective(CategoricalHyperparameters(**xi)) for xi in x_new]
    
    start_time = time.time()
    result = bs.add('categorical_opt', x_new, y_new)
    end_time = time.time()
    print(f"✅ {result['message']} (took {end_time - start_time:.2f}s)")
    
    # Sample again
    start_time = time.time()
    result = bs.sample('categorical_opt', n_samples=5)
    end_time = time.time()
    print(f"✅ Generated {len(result['samples'])} new candidates (took {end_time - start_time:.2f}s)")
    
    print(f"\n🏆 Final candidates:")
    for i, candidate in enumerate(result['samples']):
        h = CategoricalHyperparameters(**candidate)
        score = categorical_objective(h)
        print(f"   {i+1}. {candidate} (score: {score:.3f})")
    
    print(f"\n✅ Categorical optimization test completed!")

if __name__ == "__main__":
    run_categorical_test() 