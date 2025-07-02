#!/usr/bin/env python3
"""
Test Constraint Handling in Bayesian Optimization.
Expected to handle inequality constraints g(x) <= 0 and find feasible optimal solutions.
"""

import numpy as np
import time
from typing import Tuple

from beam.bayesian import BayesianHPOServiceConfig, HPOService
from beam.bayesian.hp_scheme import BaseParameters
from pydantic import Field


class ConstrainedHyperparameters(BaseParameters):
    """Hyperparameters for constrained optimization problem."""
    x: float = Field(ge=-3.0, le=3.0, description="X coordinate")
    y: float = Field(ge=-3.0, le=3.0, description="Y coordinate")
    power: float = Field(ge=0.1, le=2.0, description="Power parameter")


def constrained_objective_and_constraints(h: ConstrainedHyperparameters) -> Tuple[float, list[float]]:
    """
    Constrained optimization problem:
    
    Objective: Maximize -(x-1)² - (y+0.5)² + 2 (peak at x=1, y=-0.5)
    
    Constraints (all must be <= 0 for feasibility):
    1. g1(x,y) = x² + y² - 2.0 <= 0  (inside circle of radius √2)
    2. g2(x,y) = -x - y - 1.0 <= 0   (above line x + y >= -1)
    3. g3(x,y,p) = (x-0.5)² + (y+1)² - power <= 0  (inside power-dependent circle)
    
    This creates a feasible region that's the intersection of these constraints.
    The unconstrained optimum at (1, -0.5) may not be feasible.
    """
    x, y, power = h.x, h.y, h.power
    
    # Objective function to maximize
    objective = -(x - 1)**2 - (y + 0.5)**2 + 2
    objective += np.random.normal(0, 0.05)  # Small noise
    
    # Constraint functions (all should be <= 0)
    g1 = x**2 + y**2 - 2.0  # Circle constraint
    g2 = -x - y - 1.0       # Linear constraint
    g3 = (x - 0.5)**2 + (y + 1)**2 - power  # Power-dependent constraint
    
    constraints = [g1, g2, g3]
    
    return float(objective), [float(c) for c in constraints]


def is_feasible(constraints: list[float]) -> bool:
    """Check if a point is feasible (all constraints <= 0)."""
    return all(c <= 0 for c in constraints)


def test_constraint_handling():
    """
    Test constrained Bayesian optimization.
    
    Expected behavior:
    1. Should handle inequality constraints g(x) <= 0
    2. Should only suggest feasible points or points likely to be feasible
    3. Should find the constrained optimum, not unconstrained optimum
    4. Infeasible points should be penalized or handled appropriately
    """
    print("🔒 Testing Constraint Handling in Bayesian Optimization")
    print("=" * 70)
    
    # Test with constraint-aware acquisition function
    print("🔬 Testing Constrained Expected Improvement")
    print("-" * 50)
    
    # Initialize service with constraint handling
    hparams = BayesianHPOServiceConfig(
        acquisition_function='ConstrainedExpectedImprovement',
        acquisition_kwargs={
            'constraint_tolerance': 1e-3  # Tolerance for constraint violation
        },
        device='cpu'
    )
    
    hpo_service = HPOService(hparams=hparams)
    print("✅ HPOService initialized with constraint handling")
    
    # Register constrained problem
    x_scheme = ConstrainedHyperparameters.model_json_schema()
    result = hpo_service.register('constrained_test', x_scheme)
    print(f"✅ Problem registered: {result['message']}")
    
    # Generate initial data with mixed feasible/infeasible points
    print("\n🎲 Generating initial training data...")
    n_init = 25
    x_init = []
    y_init = []  # Will store [objective, constraint1, constraint2, constraint3]
    
    feasible_count = 0
    for _ in range(n_init):
        h = ConstrainedHyperparameters(
            x=np.random.uniform(-3.0, 3.0),
            y=np.random.uniform(-3.0, 3.0),
            power=np.random.uniform(0.1, 2.0)
        )
        
        objective, constraints = constrained_objective_and_constraints(h)
        
        # For constrained optimization, we need to handle constraints
        # Option 1: Use constraint values directly
        # Option 2: Use penalty method
        # Option 3: Use feasibility indicators
        
        # We'll use penalty method for initial implementation
        if is_feasible(constraints):
            penalized_objective = objective
            feasible_count += 1
        else:
            # Apply penalty for constraint violations
            violation = sum(max(0, c) for c in constraints)
            penalized_objective = objective - 10 * violation  # Heavy penalty
        
        x_init.append(dict(h))
        y_init.append(penalized_objective)
    
    print(f"📊 Generated {n_init} initial samples")
    print(f"✅ Feasible samples: {feasible_count}/{n_init} ({feasible_count/n_init*100:.1f}%)")
    print(f"📈 Objective range: [{min(y_init):.3f}, {max(y_init):.3f}]")
    
    # Train model
    print("\n🔄 Training constrained model...")
    start_time = time.time()
    result = hpo_service.add('constrained_test', x_init, y_init)
    end_time = time.time()
    print(f"✅ {result['message']} (took {end_time - start_time:.2f}s)")
    
    # Optimization iterations
    print("\n🎯 Constrained optimization iterations...")
    n_iterations = 15
    iteration_results = []
    
    for iteration in range(n_iterations):
        # Sample candidate
        start_time = time.time()
        result = hpo_service.sample('constrained_test', n_samples=1)
        end_time = time.time()
        
        candidate = result['samples'][0]
        h = ConstrainedHyperparameters(**candidate)
        objective, constraints = constrained_objective_and_constraints(h)
        
        # Check feasibility
        feasible = is_feasible(constraints)
        
        # Apply penalty method
        if feasible:
            penalized_objective = objective
        else:
            violation = sum(max(0, c) for c in constraints)
            penalized_objective = objective - 10 * violation
        
        # Add to training data
        hpo_service.add('constrained_test', [candidate], [penalized_objective])
        
        iteration_results.append({
            'iteration': iteration + 1,
            'candidate': candidate,
            'objective': objective,
            'penalized_objective': penalized_objective,
            'constraints': constraints,
            'feasible': feasible,
            'time': end_time - start_time
        })
        
        status = "✅" if feasible else "❌"
        print(f"  Iter {iteration + 1:2d}: {status} obj={objective:.3f}, "
              f"x={candidate['x']:.2f}, y={candidate['y']:.2f}, p={candidate['power']:.2f}")
        if not feasible:
            print(f"           Violations: g1={constraints[0]:.3f}, g2={constraints[1]:.3f}, g3={constraints[2]:.3f}")
    
    # Analyze constraint handling performance
    feasible_results = [r for r in iteration_results if r['feasible']]
    infeasible_results = [r for r in iteration_results if not r['feasible']]
    
    print(f"\n📊 Constraint Handling Analysis:")
    print(f"   Total iterations: {len(iteration_results)}")
    print(f"   Feasible solutions found: {len(feasible_results)}")
    print(f"   Infeasible solutions: {len(infeasible_results)}")
    print(f"   Feasibility rate: {len(feasible_results)/len(iteration_results)*100:.1f}%")
    
    if feasible_results:
        best_feasible = max(feasible_results, key=lambda x: x['objective'])
        print(f"\n🏆 Best feasible solution:")
        print(f"   Objective: {best_feasible['objective']:.3f}")
        print(f"   Parameters: x={best_feasible['candidate']['x']:.3f}, "
              f"y={best_feasible['candidate']['y']:.3f}, power={best_feasible['candidate']['power']:.3f}")
        print(f"   Constraints: {[f'{c:.3f}' for c in best_feasible['constraints']]}")
        print(f"   Found at iteration: {best_feasible['iteration']}")
        
        # Compare to unconstrained optimum
        unconstrained_opt_x, unconstrained_opt_y = 1.0, -0.5
        unconstrained_obj, unconstrained_constraints = constrained_objective_and_constraints(
            ConstrainedHyperparameters(x=unconstrained_opt_x, y=unconstrained_opt_y, power=1.0)
        )
        unconstrained_feasible = is_feasible(unconstrained_constraints)
        
        print(f"\n🔍 Comparison to unconstrained optimum:")
        print(f"   Unconstrained optimum: x=1.0, y=-0.5, obj={unconstrained_obj:.3f}")
        print(f"   Unconstrained feasible: {unconstrained_feasible}")
        if unconstrained_feasible:
            print(f"   ✅ Unconstrained optimum is feasible")
        else:
            print(f"   ❌ Unconstrained optimum violates constraints")
            print(f"       Constraint violations: {[f'{c:.3f}' for c in unconstrained_constraints]}")
        
        # Check if we found something close to optimal
        distance_to_unconstrained = np.sqrt(
            (best_feasible['candidate']['x'] - unconstrained_opt_x)**2 + 
            (best_feasible['candidate']['y'] - unconstrained_opt_y)**2
        )
        print(f"   Distance to unconstrained opt: {distance_to_unconstrained:.3f}")
        
    else:
        print(f"   ❌ No feasible solutions found!")
    
    # Evolution of feasibility over iterations
    feasibility_trend = [r['feasible'] for r in iteration_results]
    recent_feasibility = sum(feasibility_trend[-5:]) / min(5, len(feasibility_trend))
    print(f"\n📈 Optimization Trends:")
    print(f"   Recent feasibility rate (last 5 iters): {recent_feasibility*100:.1f}%")
    
    # Check if algorithm is learning to stay feasible
    early_feasibility = sum(feasibility_trend[:5]) / min(5, len(feasibility_trend))
    if recent_feasibility > early_feasibility:
        print(f"   ✅ Algorithm learning: feasibility improved from {early_feasibility*100:.1f}% to {recent_feasibility*100:.1f}%")
    else:
        print(f"   ⚠️  Feasibility trend: {early_feasibility*100:.1f}% → {recent_feasibility*100:.1f}%")
    
    print(f"\n✅ Constraint handling test completed!")
    
    # Expected Outputs
    print(f"\n📋 Expected Test Outcomes:")
    print(f"   • Should find feasible solutions that satisfy all constraints")
    print(f"   • Feasibility rate should improve over iterations as model learns")
    print(f"   • Best feasible solution should be near constrained optimum")
    print(f"   • Should handle mixed feasible/infeasible training data")
    print(f"   • Constraint violations should be reported accurately")
    print(f"   • If unconstrained optimum is infeasible, should find constrained optimum")


if __name__ == "__main__":
    test_constraint_handling() 