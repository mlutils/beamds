#!/usr/bin/env python3
"""
Test full initialization workflow for Bayesian optimization
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
from typing import Literal
from pydantic import BaseModel, Field
from beam.bayesian import HPOService

def test_full_initialization_workflow():
    print("🚀 Testing Full Initialization Workflow")
    print("=" * 50)
    
    # Create a complex hyperparameter schema
    class MLHyperparameters(BaseModel):
        # Numerical parameters with bounds
        learning_rate: float = Field(ge=1e-6, le=1e-1)
        weight_decay: float = Field(ge=1e-8, le=1e-2)
        dropout: float = Field(ge=0.0, le=0.8)
        
        # Categorical parameters
        optimizer: Literal["adam", "sgd", "rmsprop", "adamw"]
        scheduler: Literal["cosine", "step", "exponential", "plateau"]
        activation: Literal["relu", "gelu", "tanh", "swish"]
    
    x_scheme = MLHyperparameters.model_json_schema()
    
    # Test different initialization methods
    methods = ['sobol', 'halton', 'uniform', 'random']
    
    for method in methods:
        print(f"\n🎲 Testing {method.upper()} initialization workflow:")
        print("-" * 40)
        
        # Initialize service with specific method
        hpo_service = HPOService(
            device='cpu',
            initialization_method=method,
            start_fitting_after_n_points=12,  # Need 12 samples before training
            n_initial_points=5  # Standard BO initial points
        )
        
        problem_name = f'ml_opt_{method}'
        result = hpo_service.register(problem_name, x_scheme)
        print(f"✅ {result['message']}")
        
        # Try to sample - this will automatically generate initial samples since we have none
        init_result = hpo_service.sample(problem_name, n_samples=5)
        print(f"✅ {init_result['message']}")
        
        # Check if we got initialization samples
        if init_result['method'] == 'initialize':
            samples = init_result['samples']
            print(f"📊 Generated {len(samples)} initial samples:")
            
            for i, sample in enumerate(samples[:3]):
                print(f"   {i+1}. lr={sample['learning_rate']:.6f}, wd={sample['weight_decay']:.7f}, "
                      f"dropout={sample['dropout']:.3f}")
                print(f"      opt={sample['optimizer']}, sched={sample['scheduler']}, "
                      f"act={sample['activation']}")
            
            # Define objective function (simulated ML training)
            def objective(params):
                score = 0.0
                
                # Prefer specific learning rate ranges
                if 1e-4 <= params['learning_rate'] <= 1e-2:
                    score += 2.0
                elif params['learning_rate'] < 1e-4:
                    score += 1.0
                
                # Prefer moderate dropout
                if 0.1 <= params['dropout'] <= 0.5:
                    score += 1.5
                
                # Prefer Adam optimizer
                if params['optimizer'] in ['adam', 'adamw']:
                    score += 1.0
                
                # Prefer cosine scheduler
                if params['scheduler'] == 'cosine':
                    score += 0.8
                
                # Prefer modern activations
                if params['activation'] in ['gelu', 'swish']:
                    score += 0.5
                
                # Add noise to simulate real training
                import random
                noise = random.uniform(-0.2, 0.2)
                return max(0.0, score + noise)
            
            # Evaluate initial samples
            y_values = [objective(sample) for sample in samples]
            print(f"📈 Objective scores: min={min(y_values):.3f}, max={max(y_values):.3f}, "
                  f"mean={sum(y_values)/len(y_values):.3f}")
            
            # Add samples to the service to accumulate enough for training
            add_result = hpo_service.add(problem_name, samples, y_values)
            print(f"✅ Added initial samples: {add_result['message']}")
            
            # Now try to sample optimized candidates
            sample_result = hpo_service.sample(problem_name, n_samples=5)
            if sample_result['method'] == 'optimize':
                print(f"✅ Generated {len(sample_result['samples'])} optimized candidates")
                
                # Show optimized results
                print(f"🏆 Top optimized candidates:")
                best_scores = []
                for i, candidate in enumerate(sample_result['samples']):
                    score = objective(candidate)
                    best_scores.append(score)
                    print(f"   {i+1}. Score={score:.3f}: lr={candidate['learning_rate']:.6f}, "
                          f"opt={candidate['optimizer']}, act={candidate['activation']}")
                
                # Compare with initial samples
                best_initial = max(y_values)
                best_optimized = max(best_scores)
                improvement = best_optimized - best_initial
                
                print(f"📊 Performance comparison:")
                print(f"   Best initial: {best_initial:.3f}")
                print(f"   Best optimized: {best_optimized:.3f}")
                print(f"   Improvement: {improvement:+.3f}")
                
                if improvement > 0:
                    print(f"🎉 Optimization successful! Improved by {improvement:.3f}")
                else:
                    print(f"📈 Initial samples were strong (common with good initialization)")
            else:
                print(f"📝 Still in initialization phase: {sample_result['message']}")
        else:
            print(f"🎯 Already had enough samples, got optimized results directly")

def test_contextual_initialization():
    """Test initialization with contextual variables."""
    print(f"\n" + "=" * 60)
    print("🌟 Testing Contextual Initialization")
    print("=" * 60)
    
    class ContextualHyperparameters(BaseModel):
        lr: float = Field(ge=1e-5, le=1e-1)
        batch_size: int = Field(ge=16, le=512)
        optimizer: Literal["adam", "sgd"]
    
    x_scheme = ContextualHyperparameters.model_json_schema()
    
    # Simple context scheme (task description)
    c_scheme = {
        "type": "object",
        "properties": {
            "task_description": {"type": "string", "title": "Task Description"}
        },
        "required": ["task_description"]
    }
    
    hpo_service = HPOService(
        device='cpu',
        initialization_method='sobol',
        start_fitting_after_n_points=8
    )
    
    result = hpo_service.register('contextual_opt', x_scheme, c_scheme)
    print(f"✅ {result['message']}")
    
    # Try to sample - this will trigger initialization since we have no samples
    init_result = hpo_service.sample('contextual_opt', n_samples=3)
    print(f"✅ {init_result['message']}")
    
    if init_result['method'] == 'initialize':
        samples = init_result['samples']
        print(f"📊 Generated {len(samples)} context-free initial samples")
    else:
        print(f"🎯 Got optimized samples directly: {init_result['samples']}")
        return
    
    # Create contexts for different tasks
    contexts = [
        {"task_description": "image classification on CIFAR-10"},
        {"task_description": "natural language processing sentiment analysis"},
        {"task_description": "regression on tabular data"},
    ] * 2  # Repeat to get 6 contexts
    
    # Simple objective based on context and parameters
    def contextual_objective(params, context):
        score = 0.0
        task = context['task_description']
        
        if 'image' in task:
            # Image tasks prefer smaller learning rates and larger batches
            if params['lr'] < 1e-3:
                score += 1.0
            if params['batch_size'] >= 64:
                score += 0.5
        elif 'language' in task:
            # NLP tasks prefer moderate settings
            if 1e-4 <= params['lr'] <= 1e-2:
                score += 1.0
            if 32 <= params['batch_size'] <= 128:
                score += 0.5
        elif 'tabular' in task:
            # Tabular data is flexible
            score += 0.8
        
        # Adam generally preferred
        if params['optimizer'] == 'adam':
            score += 0.3
        
        import random
        return score + random.uniform(-0.1, 0.1)
    
    # Evaluate samples with contexts
    y_values = [contextual_objective(sample, context) 
                for sample, context in zip(samples, contexts)]
    
    print(f"📈 Contextual scores: min={min(y_values):.3f}, max={max(y_values):.3f}")
    
    # Add to service
    add_result = hpo_service.add('contextual_opt', samples, y_values, contexts)
    print(f"✅ {add_result['message']}")
    
    # Sample with specific context
    new_context = {"task_description": "computer vision object detection"}
    sample_result = hpo_service.sample('contextual_opt', c=[new_context], n_samples=3)
    print(f"✅ Generated {len(sample_result['samples'])} context-aware candidates")
    
    print(f"🎯 Context-optimized candidates for '{new_context['task_description']}':")
    for i, candidate in enumerate(sample_result['samples']):
        score = contextual_objective(candidate, new_context)
        print(f"   {i+1}. Score={score:.3f}: {candidate}")
    
    print(f"✅ Contextual initialization workflow completed!")

if __name__ == "__main__":
    test_full_initialization_workflow()
    test_contextual_initialization() 