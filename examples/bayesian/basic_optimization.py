#!/usr/bin/env python3
"""
Basic Bayesian Optimization Example

This script demonstrates the core functionality of the Beam Bayesian Optimization service,
showing how to optimize hyperparameters for a simulated LLM task with contextual information.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import numpy as np
import time
from tqdm import tqdm
from pydantic import BaseModel, confloat
from typing import Literal

from beam.bayesian import HPOService, BayesianHPOServiceConfig

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Define the optimization schemas
class Context(BaseModel):
    prompt: str

class Hyperparameters(BaseModel):
    temperature: confloat(ge=0, le=2)
    n_samples: Literal[1, 3, 7]
    depth: Literal[1, 3, 7]
    model: Literal["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "vllm-llama-33-70b-instruct", "vllm-jamba16-large-040325"]

def objective(h: Hyperparameters, c: str) -> float:
    """
    Simulated objective function for hyperparameter optimization.
    Returns a scalar score based on hyperparameters and context.
    """
    # Base value influenced by temperature (optimal around 1.0)
    temperature_effect = -(h.temperature - 1)**2 + 1

    # Effect based on number of samples (log scale favors more samples)
    sample_effect = np.log(h.n_samples)

    # Effect based on depth (optimal around 3)
    depth_effect = -(h.depth - 3)**2 + 4

    # Effect based on model complexity (manually assigned scores)
    model_scores = {
        "gpt-4.1-nano": 0.5,
        "gpt-4.1-mini": 0.7,
        "gpt-4.1": 0.9,
        "vllm-llama-33-70b-instruct": 1.0,
        "vllm-jamba16-large-040325": 0.85,
    }
    model_effect = model_scores[h.model]

    # Context effect using prompt length (longer prompts are harder)
    context_effect = -len(c) / 1000

    # Aggregate the effects into a single scalar
    scalar_result = (
        2.0 * temperature_effect +
        1.5 * sample_effect +
        1.2 * depth_effect +
        3.0 * model_effect +
        context_effect
    )

    # Add noise to make it realistic
    noise = float(np.random.randn()) * 0.5
    return scalar_result + noise

def random_sample_hyperparameters() -> Hyperparameters:
    """Generate random hyperparameters for initial data."""
    return Hyperparameters(
        temperature=random.uniform(0, 2),
        n_samples=random.choice([1, 3, 7]),
        depth=random.choice([1, 3, 7]),
        model=random.choice([
            "gpt-4.1-nano", 
            "gpt-4.1-mini", 
            "gpt-4.1", 
            "vllm-llama-33-70b-instruct", 
            "vllm-jamba16-large-040325"
        ])
    )

def random_sample_context() -> str:
    """Generate random context strings of varying lengths."""
    prompts = [
        "Explain the concept of machine learning",
        "Write a short story about a robot",
        "Summarize the benefits of renewable energy",
        "Describe the process of photosynthesis in plants",
        "Create a recipe for chocolate chip cookies",
        "Analyze the impact of social media on society",
        "Explain quantum computing in simple terms",
        "Write a poem about the changing seasons",
        "Describe the history of the internet",
        "Explain the theory of relativity"
    ]
    base_prompt = random.choice(prompts)
    # Add random variation to length
    if random.random() > 0.5:
        base_prompt += " Please provide detailed examples and explanations."
    return base_prompt

def sample_data(n=100):
    """Sample training data for the optimization problem."""
    print(f"Generating {n} samples...")
    x = [random_sample_hyperparameters() for _ in range(n)]
    c = [random_sample_context() for _ in range(n)]
    y = [objective(xi, ci) for xi, ci in zip(x, c)]
    
    # Convert to the format expected by HPOService
    x = [dict(xi) for xi in x]
    c = [{'prompt': ci} for ci in c]
    return x, y, c

def run_optimization_example():
    """Main function demonstrating the Bayesian optimization workflow."""
    print("🚀 Starting Bayesian Optimization Example")
    print("=" * 50)
    
    # Initialize the HPO service
    hparams = BayesianHPOServiceConfig(
        embedding_model="all-MiniLM-L6-v2",  # Use a simpler model without complex dependencies
        truncate_dim=32
    )
    bs = HPOService(hparams=hparams)
    print("✅ HPOService initialized")
    
    # Define schemas
    x_scheme = Hyperparameters.model_json_schema()
    c_scheme = Context.model_json_schema()
    
    # Register the optimization problem
    print("\n📝 Registering optimization problem...")
    result = bs.register('llm_optimization', x_scheme, c_scheme)
    print(f"✅ Problem registered: {result['message']}")
    print(f"📊 Embedding keys: {result['embedding_keys']}")
    
    # Generate initial training data
    print("\n🎲 Generating initial training data...")
    x_init, y_init, c_init = sample_data(50)
    print(f"📈 Initial data: {len(x_init)} samples")
    print(f"📊 Y range: [{min(y_init):.2f}, {max(y_init):.2f}]")
    
    # Add initial data to the service
    print("\n🔄 Training initial model...")
    start_time = time.time()
    result = bs.add('llm_optimization', x_init, y_init, c_init)
    end_time = time.time()
    print(f"✅ {result['message']} (took {end_time - start_time:.2f}s)")
    
    # Sample new candidates
    print("\n🎯 Sampling optimized candidates...")
    start_time = time.time()
    result = bs.sample('llm_optimization', n_samples=5)
    end_time = time.time()
    print(f"✅ Generated {len(result['samples'])} candidates (took {end_time - start_time:.2f}s)")
    
    # Display the candidates
    print("\n🏆 Optimized Candidates:")
    print("-" * 50)
    for i, candidate in enumerate(result['samples']):
        print(f"Candidate {i+1}:")
        for key, value in candidate.items():
            print(f"  {key}: {value}")
        print()
    
    # Evaluate candidates on test contexts
    print("🧪 Evaluating candidates on test data...")
    test_contexts = [random_sample_context() for _ in range(3)]
    
    best_score = float('-inf')
    best_candidate = None
    
    for i, candidate in enumerate(result['samples']):
        h = Hyperparameters(**candidate)
        scores = [objective(h, ctx) for ctx in test_contexts]
        avg_score = np.mean(scores)
        print(f"Candidate {i+1} avg score: {avg_score:.3f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_candidate = candidate
    
    print(f"\n🥇 Best candidate (score: {best_score:.3f}):")
    for key, value in best_candidate.items():
        print(f"  {key}: {value}")
    
    # Add more data and re-optimize
    print("\n🔄 Adding more data and re-optimizing...")
    x_new, y_new, c_new = sample_data(20)
    start_time = time.time()
    result = bs.add('llm_optimization', x_new, y_new, c_new)
    end_time = time.time()
    print(f"✅ {result['message']} (took {end_time - start_time:.2f}s)")
    
    # Sample again to see if recommendations improve
    print("\n🎯 Sampling after additional training...")
    start_time = time.time()
    result = bs.sample('llm_optimization', n_samples=3)
    end_time = time.time()
    print(f"✅ Generated {len(result['samples'])} new candidates (took {end_time - start_time:.2f}s)")
    
    print("\n🏆 New Candidates:")
    print("-" * 30)
    for i, candidate in enumerate(result['samples']):
        print(f"Candidate {i+1}: {candidate}")
    
    print("\n🎉 Optimization example completed successfully!")

if __name__ == "__main__":
    run_optimization_example() 