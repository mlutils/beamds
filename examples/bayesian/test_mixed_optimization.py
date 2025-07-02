#!/usr/bin/env python3
"""
Mixed Optimization Test

This script tests Bayesian optimization with both continuous and categorical variables,
which is the most common real-world scenario.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import numpy as np
import time
from pydantic import BaseModel, confloat, conint
from typing import Literal

from beam.bayesian import HPOService, BayesianHPOServiceConfig

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class MixedHyperparameters(BaseModel):
    """Mixed continuous and categorical hyperparameters."""
    # Continuous variables
    learning_rate: confloat(ge=1e-5, le=1e-1)
    dropout_rate: confloat(ge=0.0, le=0.8)
    weight_decay: confloat(ge=1e-6, le=1e-2)
    
    # Categorical variables
    optimizer: Literal["adam", "sgd", "adamw"]
    batch_size: Literal[16, 32, 64, 128]

def mixed_objective(h: MixedHyperparameters) -> float:
    """
    Mixed objective function with both continuous and categorical dependencies.
    Optimal: lr=0.001, dropout=0.2, weight_decay=1e-4, optimizer=adam, batch_size=64
    """
    
    # Continuous effects (each has an optimal point)
    lr_effect = -np.log10(abs(h.learning_rate - 0.001) + 1e-6)  # Optimal at 0.001
    dropout_effect = -(h.dropout_rate - 0.2)**2 * 10 + 2  # Optimal at 0.2
    wd_effect = -np.log10(abs(h.weight_decay - 1e-4) + 1e-7)  # Optimal at 1e-4
    
    # Categorical effects
    optimizer_scores = {"adam": 3.0, "adamw": 2.5, "sgd": 1.0}
    batch_scores = {16: 1.0, 32: 2.0, 64: 3.0, 128: 2.5}  # 64 is optimal
    
    # Mixed interactions (learning rate depends on optimizer)
    if h.optimizer == "sgd" and h.learning_rate > 0.01:
        lr_effect += 1.0  # SGD likes higher learning rates
    elif h.optimizer in ["adam", "adamw"] and h.learning_rate < 0.01:
        lr_effect += 0.5  # Adam likes lower learning rates
    
    # Batch size and learning rate interaction
    if h.batch_size >= 64 and h.learning_rate >= 0.001:
        lr_effect += 0.5  # Larger batches allow higher learning rates
    
    total_score = (
        lr_effect + dropout_effect + wd_effect + 
        optimizer_scores[h.optimizer] + 
        batch_scores[h.batch_size]
    )
    
    # Add realistic noise
    noise = np.random.normal(0, 0.3)
    return total_score + noise

def random_mixed_sample() -> MixedHyperparameters:
    """Generate random mixed hyperparameters."""
    return MixedHyperparameters(
        learning_rate=random.uniform(1e-5, 1e-1),
        dropout_rate=random.uniform(0.0, 0.8),
        weight_decay=random.uniform(1e-6, 1e-2),
        optimizer=random.choice(["adam", "sgd", "adamw"]),
        batch_size=random.choice([16, 32, 64, 128])
    )

def run_mixed_test():
    """Test mixed optimization."""
    print("🔬 Testing Mixed (Continuous + Categorical) Bayesian Optimization")
    print("=" * 70)
    
    # Initialize service
    hparams = BayesianHPOServiceConfig()
    bs = HPOService(hparams=hparams)
    print("✅ HPOService initialized")
    
    # Register problem
    x_scheme = MixedHyperparameters.model_json_schema()
    result = bs.register('mixed_opt', x_scheme)
    print(f"✅ Problem registered: {result['message']}")
    
    # Generate initial data
    print(f"\n🎲 Generating initial data...")
    n_init = 40
    x_init = [dict(random_mixed_sample()) for _ in range(n_init)]
    y_init = [mixed_objective(MixedHyperparameters(**xi)) for xi in x_init]
    
    print(f"📈 Initial data: {n_init} samples")
    print(f"📊 Y range: [{min(y_init):.2f}, {max(y_init):.2f}]")
    
    # Show theoretical optimum
    optimal = MixedHyperparameters(
        learning_rate=0.001, dropout_rate=0.2, weight_decay=1e-4,
        optimizer="adam", batch_size=64
    )
    theoretical_best = mixed_objective(optimal)
    print(f"🎯 Theoretical optimum: {theoretical_best:.2f}")
    print(f"   lr=0.001, dropout=0.2, wd=1e-4, opt=adam, bs=64")
    
    # Train model
    print(f"\n🔄 Training model...")
    start_time = time.time()
    result = bs.add('mixed_opt', x_init, y_init)
    end_time = time.time()
    print(f"✅ {result['message']} (took {end_time - start_time:.2f}s)")
    
    # Sample candidates
    print(f"\n🎯 Sampling optimized candidates...")
    start_time = time.time()
    result = bs.sample('mixed_opt', n_samples=8)
    end_time = time.time()
    print(f"✅ Generated {len(result['samples'])} candidates (took {end_time - start_time:.2f}s)")
    
    # Evaluate and display candidates
    print(f"\n🏆 Top Candidates:")
    print("-" * 80)
    
    candidates_with_scores = []
    for i, candidate in enumerate(result['samples']):
        h = MixedHyperparameters(**candidate)
        score = mixed_objective(h)
        candidates_with_scores.append((candidate, score))
        
        print(f"Candidate {i+1} (score: {score:.3f}):")
        print(f"  lr={candidate['learning_rate']:.6f}, dropout={candidate['dropout_rate']:.3f}, wd={candidate['weight_decay']:.6f}")
        print(f"  optimizer={candidate['optimizer']}, batch_size={candidate['batch_size']}")
        print()
    
    # Find best candidate
    best_candidate, best_score = max(candidates_with_scores, key=lambda x: x[1])
    print(f"🥇 Best candidate found (score: {best_score:.3f}):")
    print(f"   learning_rate: {best_candidate['learning_rate']:.6f} (optimal: 0.001)")
    print(f"   dropout_rate: {best_candidate['dropout_rate']:.3f} (optimal: 0.2)")
    print(f"   weight_decay: {best_candidate['weight_decay']:.6f} (optimal: 0.0001)")
    print(f"   optimizer: {best_candidate['optimizer']} (optimal: adam)")
    print(f"   batch_size: {best_candidate['batch_size']} (optimal: 64)")
    
    # Calculate gaps from optimum
    lr_gap = abs(best_candidate['learning_rate'] - 0.001)
    dropout_gap = abs(best_candidate['dropout_rate'] - 0.2)
    wd_gap = abs(best_candidate['weight_decay'] - 1e-4)
    
    print(f"\n📊 Gaps from theoretical optimum:")
    print(f"   Learning rate gap: {lr_gap:.6f}")
    print(f"   Dropout gap: {dropout_gap:.3f}")
    print(f"   Weight decay gap: {wd_gap:.6f}")
    print(f"   Score gap: {theoretical_best - best_score:.3f}")
    
    # Add more data and iterate
    print(f"\n🔄 Adding more data for refinement...")
    x_new = [dict(random_mixed_sample()) for _ in range(30)]
    y_new = [mixed_objective(MixedHyperparameters(**xi)) for xi in x_new]
    
    start_time = time.time()
    result = bs.add('mixed_opt', x_new, y_new)
    end_time = time.time()
    print(f"✅ {result['message']} (took {end_time - start_time:.2f}s)")
    
    # Sample refined candidates
    start_time = time.time()
    result = bs.sample('mixed_opt', n_samples=5)
    end_time = time.time()
    print(f"✅ Generated {len(result['samples'])} refined candidates (took {end_time - start_time:.2f}s)")
    
    print(f"\n🏆 Refined candidates:")
    print("-" * 50)
    
    best_refined_score = -float('inf')
    best_refined_candidate = None
    
    for i, candidate in enumerate(result['samples']):
        h = MixedHyperparameters(**candidate)
        score = mixed_objective(h)
        if score > best_refined_score:
            best_refined_score = score
            best_refined_candidate = candidate
            
        print(f"{i+1}. Score: {score:.3f}")
        print(f"   lr={candidate['learning_rate']:.6f}, dropout={candidate['dropout_rate']:.3f}")
        print(f"   wd={candidate['weight_decay']:.6f}, opt={candidate['optimizer']}, bs={candidate['batch_size']}")
        print()
    
    print(f"🎯 Final best score: {best_refined_score:.3f}")
    print(f"📈 Improvement: {best_refined_score - best_score:.3f}")
    print(f"📊 Final gap from theoretical best: {theoretical_best - best_refined_score:.3f}")
    
    print(f"\n✅ Mixed optimization test completed!")

if __name__ == "__main__":
    run_mixed_test() 