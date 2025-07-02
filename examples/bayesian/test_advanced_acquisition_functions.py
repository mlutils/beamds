#!/usr/bin/env python3
"""
Test Advanced Acquisition Functions: qKnowledgeGradient, Thompson Sampling, etc.
Expected to show different exploration/exploitation behaviors compared to standard EI.
"""

import numpy as np
import time

from beam.bayesian import BayesianHPOServiceConfig, HPOService
from beam.bayesian.hp_scheme import BaseParameters
from pydantic import Field


class AdvancedTestHyperparameters(BaseParameters):
    """Hyperparameters for testing advanced acquisition functions."""
    x1: float = Field(ge=-5.0, le=5.0, description="First dimension")
    x2: float = Field(ge=-5.0, le=5.0, description="Second dimension")
    discrete_choice: str = Field(description="Discrete parameter", json_schema_extra={
        "enum": ["option_a", "option_b", "option_c"]
    })


def complex_objective(h: AdvancedTestHyperparameters) -> float:
    """
    Complex multi-modal objective function with local optima.
    This tests if advanced acquisition functions can handle exploration better.
    
    Function has:
    - Global optimum around x1=1.5, x2=-0.8, choice=option_b
    - Local optima to test exploration capabilities
    - Noise to make optimization challenging
    """
    x1, x2 = h.x1, h.x2
    
    # Base function: modified Rosenbrock with multiple modes
    base = -(100 * (x2 - x1**2)**2 + (1 - x1)**2) / 1000
    
    # Add local optima
    local_opt1 = 0.8 * np.exp(-((x1 + 2)**2 + (x2 - 1)**2) / 0.5)
    local_opt2 = 0.6 * np.exp(-((x1 - 3)**2 + (x2 + 2)**2) / 0.8)
    
    # Global optimum around (1.5, -0.8)
    global_opt = 1.0 * np.exp(-((x1 - 1.5)**2 + (x2 + 0.8)**2) / 0.3)
    
    # Discrete choice effect
    choice_effects = {"option_a": -0.2, "option_b": 0.3, "option_c": 0.0}
    choice_effect = choice_effects[h.discrete_choice]
    
    result = base + local_opt1 + local_opt2 + global_opt + choice_effect
    
    # Add noise
    result += np.random.normal(0, 0.1)
    
    return float(result)


def test_advanced_acquisition_functions():
    """
    Test advanced acquisition functions and compare their exploration behavior.
    
    Expected behaviors:
    1. qKnowledgeGradient should show good exploration of uncertain regions
    2. ThompsonSampling should balance exploration/exploitation differently than EI
    3. All should eventually converge to global optimum region
    4. Different functions should show different sampling patterns
    """
    print("🚀 Testing Advanced Acquisition Functions")
    print("=" * 70)
    
    # Test different acquisition functions
    acquisition_functions = [
        'LogExpectedImprovement',  # Baseline
        'qKnowledgeGradient',     # Advanced - good for exploration
        'ThompsonSampling',       # Advanced - probabilistic sampling
        'qUpperConfidenceBound'   # Alternative for comparison
    ]
    
    results_comparison = {}
    
    for acq_func in acquisition_functions:
        print(f"\n🔬 Testing {acq_func} acquisition function")
        print("-" * 50)
        
        try:
            # Configure acquisition function
            if acq_func == 'qKnowledgeGradient':
                acq_kwargs = {'num_fantasies': 32}  # KG-specific parameter
            elif acq_func == 'ThompsonSampling':
                acq_kwargs = {'replacement': False}  # TS-specific parameter
            elif acq_func == 'qUpperConfidenceBound':
                acq_kwargs = {'beta': 2.0}  # UCB exploration parameter
            else:
                acq_kwargs = {}
            
            # Initialize service
            hparams = BayesianHPOServiceConfig(
                acquisition_function=acq_func,
                acquisition_kwargs=acq_kwargs,
                device='cpu',
                num_restarts=100,  # More restarts for better optimization
                raw_samples=256
            )
            
            hpo_service = HPOService(hparams=hparams)
            print(f"✅ HPOService initialized with {acq_func}")
            
            # Register problem
            x_scheme = AdvancedTestHyperparameters.model_json_schema()
            problem_name = f'advanced_test_{acq_func.lower()}'
            result = hpo_service.register(problem_name, x_scheme)
            print(f"✅ Problem registered: {result['message']}")
            
            # Generate initial data (small set to test exploration)
            print("\n🎲 Generating initial training data...")
            n_init = 15
            x_init = []
            y_init = []
            
            # Generate diverse initial points
            for _ in range(n_init):
                h = AdvancedTestHyperparameters(
                    x1=np.random.uniform(-5.0, 5.0),
                    x2=np.random.uniform(-5.0, 5.0),
                    discrete_choice=np.random.choice(["option_a", "option_b", "option_c"])
                )
                y = complex_objective(h)
                
                x_init.append(dict(h))
                y_init.append(y)
            
            print(f"📊 Generated {n_init} initial samples")
            print(f"📈 Objective range: [{min(y_init):.3f}, {max(y_init):.3f}]")
            best_init = max(y_init)
            print(f"🏆 Best initial value: {best_init:.3f}")
            
            # Train model
            print("\n🔄 Training model...")
            start_time = time.time()
            result = hpo_service.add(problem_name, x_init, y_init)
            end_time = time.time()
            print(f"✅ {result['message']} (took {end_time - start_time:.2f}s)")
            
            # Perform optimization iterations
            print(f"\n🎯 Optimizing with {acq_func}...")
            n_iterations = 10
            iteration_results = []
            
            for iteration in range(n_iterations):
                # Sample candidate
                start_time = time.time()
                result = hpo_service.sample(problem_name, n_samples=1)
                end_time = time.time()
                
                candidate = result['samples'][0]
                h = AdvancedTestHyperparameters(**candidate)
                y = complex_objective(h)
                
                # Add to training data
                hpo_service.add(problem_name, [candidate], [y])
                
                iteration_results.append({
                    'iteration': iteration + 1,
                    'candidate': candidate,
                    'objective': y,
                    'time': end_time - start_time
                })
                
                print(f"  Iter {iteration + 1:2d}: y={y:.3f}, "
                      f"x1={candidate['x1']:.2f}, x2={candidate['x2']:.2f}, "
                      f"choice={candidate['discrete_choice']}")
            
            # Analyze results
            best_found = max(iteration_results, key=lambda x: x['objective'])
            final_best = best_found['objective']
            improvement = final_best - best_init
            
            print(f"\n📊 {acq_func} Performance Summary:")
            print(f"   Initial best: {best_init:.3f}")
            print(f"   Final best: {final_best:.3f}")
            print(f"   Improvement: {improvement:.3f}")
            print(f"   Best found at iteration: {best_found['iteration']}")
            print(f"   Best parameters: x1={best_found['candidate']['x1']:.3f}, "
                  f"x2={best_found['candidate']['x2']:.3f}, "
                  f"choice={best_found['candidate']['discrete_choice']}")
            
            # Calculate exploration metrics
            x1_coords = [r['candidate']['x1'] for r in iteration_results]
            x2_coords = [r['candidate']['x2'] for r in iteration_results]
            x1_std = np.std(x1_coords)
            x2_std = np.std(x2_coords)
            exploration_score = x1_std + x2_std  # Simple exploration metric
            
            print(f"   Exploration score: {exploration_score:.3f} (higher = more exploratory)")
            print(f"   Average iteration time: {np.mean([r['time'] for r in iteration_results]):.3f}s")
            
            # Store results for comparison
            results_comparison[acq_func] = {
                'final_best': final_best,
                'improvement': improvement,
                'exploration_score': exploration_score,
                'convergence_iteration': best_found['iteration'],
                'best_params': best_found['candidate']
            }
            
        except Exception as e:
            print(f"❌ Error testing {acq_func}: {e}")
            results_comparison[acq_func] = None
    
    # Compare acquisition functions
    print(f"\n📋 Acquisition Function Comparison")
    print("=" * 70)
    
    valid_results = {k: v for k, v in results_comparison.items() if v is not None}
    
    if len(valid_results) > 1:
        # Best performance
        best_performer = max(valid_results.keys(), key=lambda k: valid_results[k]['final_best'])
        print(f"🥇 Best final objective: {best_performer} ({valid_results[best_performer]['final_best']:.3f})")
        
        # Most exploratory
        most_exploratory = max(valid_results.keys(), key=lambda k: valid_results[k]['exploration_score'])
        print(f"🔍 Most exploratory: {most_exploratory} (score: {valid_results[most_exploratory]['exploration_score']:.3f})")
        
        # Fastest convergence
        fastest_convergence = min(valid_results.keys(), key=lambda k: valid_results[k]['convergence_iteration'])
        print(f"⚡ Fastest convergence: {fastest_convergence} (iteration {valid_results[fastest_convergence]['convergence_iteration']})")
        
        print(f"\n📊 Detailed Comparison:")
        for acq_func, results in valid_results.items():
            print(f"   {acq_func:20s}: best={results['final_best']:.3f}, "
                  f"improvement={results['improvement']:.3f}, "
                  f"exploration={results['exploration_score']:.3f}")
    
    print(f"\n✅ Advanced acquisition functions test completed!")
    
    # Expected Outputs
    print(f"\n📋 Expected Test Outcomes:")
    print(f"   • All acquisition functions should work without errors")
    print(f"   • qKnowledgeGradient should show good exploration (higher exploration score)")
    print(f"   • ThompsonSampling should balance exploration/exploitation")
    print(f"   • Different functions should show different sampling patterns")
    print(f"   • All should improve from initial best value")
    print(f"   • Advanced functions may find better optima than standard EI")


if __name__ == "__main__":
    test_advanced_acquisition_functions() 