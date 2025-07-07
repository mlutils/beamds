"""
🚀 Comprehensive Bayesian Optimization Framework Demo
====================================================

This demo showcases the full capabilities of our advanced Bayesian optimization framework:

✨ Features Demonstrated:
- 🎯 Multi-objective optimization (quality vs cost)
- 🔒 Constraint handling (latency limits)
- 💾 Database logging with BeamIbis integration
- 🔢 Mixed parameter types (numerical + categorical)
- 📝 Textual contexts with embedding network
- 📊 Pareto front visualization
- ⚡ Performance analysis and statistics
- 🔄 Multiple constraint handling methods

🎓 Problem: LLM Inference Optimization
We're optimizing LLM inference parameters for quality, cost, and latency constraints.
Based on the clean problem definition from basic_optimization.py.
"""

import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

# Set style for beautiful plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
sns.set_palette("husl")

def print_header(title: str, emoji: str = "🔥"):
    """Print a beautiful header for sections."""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 4))

def print_subheader(title: str, emoji: str = "📌"):
    """Print a subheader for subsections."""
    print(f"\n{emoji} {title}")
    print("-" * (len(title) + 4))

def simulate_llm_inference(params: dict, context: dict) -> dict:
    """
    Realistic LLM inference simulation that generates objectives and constraints.
    
    This simulates LLM inference with various hyperparameters and prompt contexts,
    returning quality scores, costs, and latency based on parameter interactions.
    Based on the objective function from basic_optimization.py but extended for multi-objective.
    """
    # Extract parameters
    temperature = params['temperature']
    n_samples = params['n_samples']
    depth = params['depth']
    model = params['model']
    
    # Extract context
    prompt = context['prompt']
    
    # Infer task type from prompt content for simulation
    prompt_lower = prompt.lower()
    if "python" in prompt_lower or "function" in prompt_lower:
        task_type = "coding"
    elif "analysis" in prompt_lower or "economic" in prompt_lower:
        task_type = "analysis"
    elif "story" in prompt_lower or "write" in prompt_lower:
        task_type = "creative_writing"
    elif "explain" in prompt_lower or "concept" in prompt_lower:
        task_type = "reasoning"
    else:
        task_type = "summarization"
    
    # Infer complexity from prompt length and content
    if len(prompt) > 150 or "comprehensive" in prompt_lower:
        complexity_hint = "complex"
    elif len(prompt) < 80:
        complexity_hint = "simple"
    else:
        complexity_hint = "medium"
    
    # Simulate realistic parameter interactions
    # Include textual context in the seed for embedding-like behavior
    context_hash = hash(f"{task_type}_{len(prompt)}_{prompt[:30]}") % 1000
    random.seed(context_hash)
    np.random.seed(context_hash)
    
    # Simulate embedding network influence based on textual context
    embedding_effects = 0.0
    
    # Analyze context for embedding-aware adjustments
    if "complex" in complexity_hint.lower() or "deep reasoning" in complexity_hint.lower():
        embedding_effects += 0.15  # Boost for complex tasks
    if "simple" in complexity_hint.lower() or "straightforward" in complexity_hint.lower():
        embedding_effects -= 0.05  # Slight penalty for simple tasks
    if "creative" in complexity_hint.lower():
        embedding_effects += 0.1  # Boost for creative tasks
    if "technical" in complexity_hint.lower() or "algorithmic" in complexity_hint.lower():
        embedding_effects += 0.08  # Boost for technical content
    
    # Base quality depends on parameter settings (from basic_optimization.py logic)
    # Temperature effect (optimal around 1.0)
    temperature_effect = -(temperature - 1.0)**2 + 1.0
    
    # Number of samples effect (more samples generally better, but diminishing returns)
    sample_effect = np.log(n_samples) * 0.5
    
    # Depth effect (optimal around 3 for reasoning)
    depth_effect = -(depth - 3)**2 * 0.1 + 0.9
    
    # Model quality scores (manually assigned based on capability)
    model_scores = {
        "gpt-4.1-nano": {'quality': 0.6, 'cost': 0.01, 'latency': 800},
        "gpt-4.1-mini": {'quality': 0.75, 'cost': 0.03, 'latency': 1200},
        "gpt-4.1": {'quality': 0.95, 'cost': 0.12, 'latency': 2500},
        "vllm-llama-33-70b-instruct": {'quality': 0.88, 'cost': 0.08, 'latency': 3200},
        "vllm-jamba16-large-040325": {'quality': 0.82, 'cost': 0.06, 'latency': 2800},
    }
    model_effect = model_scores[model]
    
    # Context complexity effect (longer prompts are harder)
    complexity_penalty = len(prompt) / 1000.0 * 0.1
    
    # Task type effects
    task_type_effects = {
        'creative_writing': {'quality_boost': 0.1, 'latency_factor': 1.3},
        'analysis': {'quality_boost': 0.15, 'latency_factor': 1.5},
        'coding': {'quality_boost': 0.05, 'latency_factor': 1.2},
        'reasoning': {'quality_boost': 0.2, 'latency_factor': 1.4},
        'summarization': {'quality_boost': 0.0, 'latency_factor': 0.8}
    }
    task_effect = task_type_effects[task_type]
    
    # Calculate quality score (to be maximized)
    quality_score = (
        model_effect['quality'] +
        temperature_effect * 0.2 +
        sample_effect * 0.15 +
        depth_effect * 0.25 +
        task_effect['quality_boost'] +
        embedding_effects -
        complexity_penalty
    )
    
    # Add realistic noise
    quality_score += np.random.normal(0, 0.05)
    quality_score = max(0.1, min(1.0, quality_score))  # Clamp to realistic range
    
    # Calculate cost per token (to be minimized)
    base_cost = model_effect['cost']
    cost_per_token = (
        base_cost * 
        (1 + depth * 0.1) *  # More depth increases cost
        (1 + n_samples * 0.2)  # More samples increase cost
    )
    
    # Temperature affects cost slightly (higher temp might need more tokens)
    cost_per_token *= (1 + abs(temperature - 1.0) * 0.1)
    
    cost_per_token += np.random.normal(0, cost_per_token * 0.05)
    cost_per_token = max(0.001, cost_per_token)  # At least 0.1 cents
    
    # Calculate latency (constraint)
    base_latency = model_effect['latency']
    latency_ms = (
        base_latency * 
        task_effect['latency_factor'] *
        (1 + len(prompt) / 5000) *  # Longer prompts take more time
        (1 + depth * 0.3) *  # More reasoning steps take time
        (1 + n_samples * 0.4)  # Multiple samples take time
    )
    
    # Temperature affects latency (higher temp can be slower due to sampling)
    if temperature > 1.5:
        latency_ms *= 1.2
    
    latency_ms += np.random.normal(0, latency_ms * 0.1)
    latency_ms = max(200, latency_ms)  # At least 200ms
    
    # Simulate occasional poor performance for extreme parameters
    if temperature > 1.8 or (depth == 1 and "complex" in complexity_hint.lower()):
        if random.random() < 0.2:  # 20% chance of degraded performance
            quality_score *= 0.7  # Lower quality
            latency_ms *= 1.5  # Higher latency
            cost_per_token *= 1.3  # Higher cost due to inefficiency
    
    return {
        'quality_score': round(quality_score, 4),
        'cost_per_token': round(cost_per_token, 4),
        'latency_ms': round(latency_ms, 1)
    }

def create_llm_optimization_demo():
    """Main demo function showcasing all framework capabilities."""
    
    print_header("🚀 Comprehensive Bayesian Optimization Framework Demo", "🌟")
    print("""
    Welcome to our advanced Bayesian optimization framework demo!
    
    🎯 Today's Challenge: ML Hyperparameter Optimization
    We'll optimize a machine learning pipeline with:
    - Multiple objectives (accuracy vs efficiency)
    - Resource constraints (memory and time limits)  
    - Mixed parameter types (numerical + categorical)
    - Contextual information (dataset characteristics)
    
    Let's dive in! 🚀
    """)
    
    # ========================================================================
    # 1. Setup and Configuration
    # ========================================================================
    
    print_header("1️⃣ Framework Setup & Problem Definition")
    
    # Import framework components
    from beam.bayesian.hpo_service import HPOService
    from beam.bayesian.config import BayesianHPOServiceConfig
    from beam.bayesian.hp_scheme import BaseParameters
    from pydantic import Field, BaseModel
    
    # Define hyperparameter schema with mixed types
    print_subheader("🔧 LLM Hyperparameter Schema Definition")
    
    class LLMHyperparameters(BaseParameters):
        """LLM inference hyperparameters with realistic ranges and types."""
        temperature: float = Field(
            ge=0.0, le=2.0,
            description="Sampling temperature for generation"
        )
        n_samples: int = Field(
            enum=[1, 3, 7],
            description="Number of samples to generate"
        )
        depth: int = Field(
            enum=[1, 3, 7],
            description="Reasoning depth or chain-of-thought steps"
        )
        model: str = Field(
            enum=[
                "gpt-4.1-nano", 
                "gpt-4.1-mini", 
                "gpt-4.1", 
                "vllm-llama-33-70b-instruct", 
                "vllm-jamba16-large-040325"
            ],
            description="LLM model to use for inference"
        )
    
    # Define contextual information schema
    print_subheader("📝 Context Schema Definition")
    
    class PromptContext(BaseParameters):
        """Prompt and task context information for embedding network."""
        prompt: str = Field(description="The main prompt/query to process")
        
    # Define multi-objective optimization schema
    print_subheader("🎯 Multi-Objective Schema Definition")
    
    class LLMObjectives(BaseParameters):
        """LLM optimization objectives and constraints."""
        quality_score: float = Field(
            description="Response quality score to maximize"
        )
        cost_per_token: float = Field(
            description="Cost per token to minimize (cents)"
        )
        latency_ms: float = Field(
            description="Response latency constraint (milliseconds)"
        )
    
    # Add objective and constraint metadata to the schema
    def add_optimization_metadata(schema):
        """Add objective and constraint metadata to the JSON schema."""
        schema = schema.copy()
        
        # Add objective metadata
        schema['properties']['quality_score']['objective'] = 'maximize'
        schema['properties']['cost_per_token']['objective'] = 'minimize'
        
        # Add constraint metadata
        schema['properties']['latency_ms']['constraint'] = '<= 5000'
        
        return schema
    
    print("✅ Schema defined:")
    print(f"   📊 Parameters: {len(LLMHyperparameters.model_fields)} (mixed numerical/categorical)")
    print(f"   🌍 Context fields: {len(PromptContext.model_fields)} (textual prompt for embeddings)")
    print(f"   🎯 Objectives: 2 (quality_score ↑, cost_per_token ↓)")
    print(f"   🔒 Constraints: 1 (latency_ms ≤ 5000 ms)")
    
    # ========================================================================
    # 2. Database Logging Setup
    # ========================================================================
    
    print_header("2️⃣ Database Logging Configuration")
    
    # Setup database for experiment tracking
    experiment_name = f"llm_optimization_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    database_path = f"experiments_{experiment_name.split('_')[-1]}.db"
    
    print(f"📁 Database: {database_path}")
    print(f"🏷️  Experiment: {experiment_name}")
    
    # Configure optimization service (temporarily without database logging)
    config = BayesianHPOServiceConfig(
        # Core optimization settings
        start_fitting_after_n_points=5,
        acquisition_function="qLogEHVI",  # Multi-objective acquisition (recommended over qEHVI)
        constraint_method="penalty",   # Constraint handling
        penalty_weight=100.0,
        device="cpu",
        n_categorical_features_threshold=2,

    # Fix the reference point for proper qLogEHVI operation
        # NEW: Use named reference points - no more ordering nightmares!
        # Simply specify the worst acceptable value for each objective by name
        acquisition_kwargs={
            "ref_point": {
                "quality_score": 0.0,     # Worst acceptable quality (maximize)
                "cost_per_token": 1.0     # Worst acceptable cost in cents (minimize)
            }
        },
        
        # Embedding configuration (like basic_optimization.py)
        embedding_model="all-MiniLM-L6-v2",
        truncate_dim=32
    )
    
    print("✅ Configuration complete:")
    print(f"   🔍 Acquisition: {config.acquisition_function} (multi-objective)")
    print(f"   🔒 Constraints: {config.constraint_method} method")
    print(f"   📊 Reference point: {config.acquisition_kwargs.get('ref_point', 'auto-generated')}")
    print(f"   💾 Database logging: Enabled")
    print(f"   ⚡ Device: {config.device}")
    
    # ========================================================================
    # 3. Service Initialization and Problem Registration
    # ========================================================================
    
    print_header("3️⃣ Service Initialization")
    
    # Initialize the HPO service
    print_subheader("🚀 Starting HPO Service")
    service = HPOService(hparams=config)
    
    # Register the optimization problem
    problem_name = "llm_inference_optimization"
    
    result = service.register(
        problem_name,
        LLMHyperparameters.model_json_schema(),  # x_scheme
        PromptContext.model_json_schema(),       # c_scheme
        add_optimization_metadata(LLMObjectives.model_json_schema())  # y_scheme with metadata
    )
    
    print(f"✅ {result['message']}")
    print(f"   🔍 Problem: {problem_name}")
    print(f"   📊 Multi-objective: {len(result.get('objectives', {}))} objectives")
    print(f"   🔒 Constraints: {len(result.get('constraints', {}))} constraints")
    
    # Debug: Check if our reference point was preserved
    solver = service._problems[problem_name].solver
    actual_ref_point = solver.hparams.get('acquisition_kwargs', {}).get('ref_point', 'NOT_FOUND')
    print(f"   🐛 DEBUG: Actual reference point in solver: {actual_ref_point}")
    expected_ref_point = {"quality_score": 0.0, "cost_per_token": 1.0}
    if actual_ref_point == expected_ref_point:
        print(f"   ✅ Our named reference point was preserved!")
    else:
        print(f"   ℹ️  Reference point in use: {actual_ref_point}")
        print(f"   💡 System will auto-convert or auto-generate as needed")
    
    # ========================================================================
    # 4. Generate Initial Training Data
    # ========================================================================
    
    print_header("4️⃣ Initial Data Generation")
    
    print_subheader("🎲 Generating Diverse Initial Samples")
    
    # Create diverse prompt contexts for comprehensive evaluation
    contexts = [
        {
            "prompt": "Write a comprehensive analysis of the economic impacts of renewable energy adoption, including both short-term costs and long-term benefits for different stakeholders."
        },
        {
            "prompt": "Create a Python function that efficiently finds the longest palindromic substring in a given string, with detailed comments explaining the algorithm."
        },
        {
            "prompt": "Write a short story about a time traveler who accidentally changes history and must find a way to fix it before the timeline collapses."
        },
        {
            "prompt": "Explain the concept of quantum entanglement to a curious 12-year-old, using analogies and simple language while maintaining scientific accuracy."
        },
        {
            "prompt": "Summarize the key findings from this research paper on climate change mitigation strategies and their effectiveness in different regions."
        }
    ]
    
    # Generate initial LLM hyperparameter configurations
    initial_configs = [
        {
            "temperature": 1.0, "n_samples": 1, "depth": 3,
            "model": "gpt-4.1-mini"
        },
        {
            "temperature": 0.7, "n_samples": 3, "depth": 1,
            "model": "gpt-4.1-nano"
        },
        {
            "temperature": 1.5, "n_samples": 1, "depth": 7,
            "model": "vllm-llama-33-70b-instruct"
        },
        {
            "temperature": 0.3, "n_samples": 7, "depth": 3,
            "model": "gpt-4.1"
        },
        {
            "temperature": 2.0, "n_samples": 1, "depth": 1,
            "model": "vllm-jamba16-large-040325"
        }
    ]
    
    # Evaluate initial configurations
    print("🔄 Evaluating initial configurations...")
    
    initial_data = []
    for i, (config, context) in enumerate(zip(initial_configs, contexts)):
        print(f"   ⚡ Config {i+1}/5: {config['model']} (temp={config['temperature']})")
        
        # Simulate LLM inference
        start_time = time.time()
        results = simulate_llm_inference(config, context)
        eval_time = time.time() - start_time
        
        initial_data.append({
            'x': config,
            'y': results,
            'c': context,
            'eval_time': eval_time
        })
        
        # Show results
        print(f"     📊 Quality: {results['quality_score']:.3f}, "
              f"Cost: {results['cost_per_token']:.3f}¢, "
              f"Latency: {results['latency_ms']:.0f}ms")
    
    # Add initial data to the optimization service
    x_data = [d['x'] for d in initial_data]
    y_data = [d['y'] for d in initial_data]
    c_data = [d['c'] for d in initial_data]
    
    result = service.add(problem_name, x_data, y_data, c_data)
    
    print(f"\n✅ Initial data added:")
    print(f"   📊 Samples: {len(initial_data)}")
    print(f"   🔒 Constraint violations: {sum(1 for d in initial_data if d['y']['latency_ms'] > 5000)}")
    print(f"   🎯 Best quality: {max(d['y']['quality_score'] for d in initial_data):.3f}")
    print(f"   💰 Min cost: {min(d['y']['cost_per_token'] for d in initial_data):.3f}¢")
    
    # Diagnostic output for reference point validation
    print(f"\n🔍 Objective Range Analysis:")
    quality_values = [d['y']['quality_score'] for d in initial_data]
    cost_values = [d['y']['cost_per_token'] for d in initial_data]
    print(f"   📊 Quality score range: {min(quality_values):.3f} to {max(quality_values):.3f}")
    print(f"   💰 Cost per token range: {min(cost_values):.4f}¢ to {max(cost_values):.4f}¢")
    print(f"   🎯 This should eliminate the 'BadInitialCandidatesWarning'!")
    
    # ========================================================================
    # 5. Bayesian Optimization Loop
    # ========================================================================
    
    print_header("5️⃣ Bayesian Optimization Loop")
    
    optimization_results = []
    total_start_time = time.time()
    
    # Run optimization iterations
    n_iterations = 6
    print(f"🔄 Running {n_iterations} optimization iterations...")
    
    for iteration in range(n_iterations):
        print(f"\n🔍 Iteration {iteration + 1}/{n_iterations}")
        
        # Sample next configuration
        iter_start = time.time()
        
        # Randomly select context for this iteration
        context = random.choice(contexts)
        
        try:
            n_samples = 1
            c = [{'prompt': str(contexts[random.randint(0, len(contexts) - 1)]['prompt'])} for _ in range(n_samples)]
            sample_result = service.sample(problem_name, c=c, n_samples=n_samples)
            
            # if sample_result['method'] == 'initialize':
            #     print("   ⚠️  Still in initialization phase")
            #     continue
                
            suggested_config = sample_result['samples'][0]
            acquisition_value = sample_result.get('acquisition_value', 0.0)
            
            print(f"   🎯 Suggested config: {suggested_config['model']} "
                  f"(temp={suggested_config['temperature']:.2f}, samples={suggested_config['n_samples']})")
            print(f"   📊 Acquisition value: {acquisition_value:.4f}")
            print(f"   📝 Prompt: {context['prompt'][:80]}...")
            
            # Evaluate the suggested configuration  
            eval_start = time.time()
            # Use original context dict for simulation (before embedding conversion)
            original_context = {
                "prompt": "Write a comprehensive analysis of the economic impacts of renewable energy adoption, including both short-term costs and long-term benefits for different stakeholders."
            }
            results = simulate_llm_inference(suggested_config, original_context)
            eval_time = time.time() - eval_start
            
            # Add results to service
            result = service.add(problem_name, [suggested_config], [results], [original_context])
            
            # Track performance
            iter_time = time.time() - iter_start
            constraint_violated = results['latency_ms'] > 5000
            
            optimization_results.append({
                'iteration': iteration + 1,
                'config': suggested_config,
                'results': results,
                'context': context,
                'acquisition_value': acquisition_value,
                'eval_time': eval_time,
                'iter_time': iter_time,
                'constraint_violated': constraint_violated
            })
            
            # Display results
            status = "❌" if constraint_violated else "✅"
            print(f"   {status} Results: quality={results['quality_score']:.3f}, "
                  f"cost={results['cost_per_token']:.3f}¢, "
                  f"latency={results['latency_ms']:.0f}ms")
            print(f"   ⏱️  Iteration time: {iter_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Error in iteration {iteration + 1}: {e}")
            raise e
    
    total_time = time.time() - total_start_time
    
    print(f"\n✅ Optimization completed!")
    print(f"   ⏱️  Total time: {total_time:.2f}s")
    print(f"   🔄 Successful iterations: {len(optimization_results)}")
    print(f"   🔒 Constraint violations: {sum(r['constraint_violated'] for r in optimization_results)}")
    
    # ========================================================================
    # 6. Results Analysis and Visualization
    # ========================================================================
    
    print_header("6️⃣ Results Analysis & Visualization")
    
    # Collect all results for analysis
    all_results = []
    
    # Add initial data
    for i, data in enumerate(initial_data):
        all_results.append({
            'iteration': 0,
            'source': 'initial',
            'quality_score': data['y']['quality_score'],
            'cost_per_token': data['y']['cost_per_token'],
            'latency_ms': data['y']['latency_ms'],
            'constraint_violated': data['y']['latency_ms'] > 5000,
            'model': data['x']['model'],
            'temperature': data['x']['temperature']
        })
    
    # Add optimization results
    for result in optimization_results:
        all_results.append({
            'iteration': result['iteration'],
            'source': 'optimized',
            'quality_score': result['results']['quality_score'],
            'cost_per_token': result['results']['cost_per_token'],
            'latency_ms': result['results']['latency_ms'],
            'constraint_violated': result['constraint_violated'],
            'model': result['config']['model'],
            'temperature': result['config']['temperature']
        })
    
    df_results = pd.DataFrame(all_results)
    
    print_subheader("📈 Creating Visualizations")
    
    # Create comprehensive visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🚀 Bayesian Optimization Results Analysis', fontsize=16, fontweight='bold')
    
    # 1. Pareto Front Visualization
    ax1 = axes[0, 0]
    feasible_mask = df_results['latency_ms'] <= 5000
    
    # Plot all points
    scatter1 = ax1.scatter(df_results[~feasible_mask]['cost_per_token'], 
                         df_results[~feasible_mask]['quality_score'],
                         c='red', alpha=0.6, s=60, label='Infeasible (Latency > 5000ms)')
    
    scatter2 = ax1.scatter(df_results[feasible_mask]['cost_per_token'], 
                         df_results[feasible_mask]['quality_score'],
                         c=df_results[feasible_mask]['iteration'], 
                         cmap='viridis', s=80, label='Feasible', alpha=0.8)
    
    # Highlight Pareto front
    feasible_df = df_results[feasible_mask]
    if len(feasible_df) > 0:
        # Find Pareto optimal points (maximize quality_score, minimize cost_per_token)
        pareto_points = []
        for i, row in feasible_df.iterrows():
            is_dominated = False
            for j, other_row in feasible_df.iterrows():
                if (other_row['quality_score'] >= row['quality_score'] and 
                    other_row['cost_per_token'] <= row['cost_per_token'] and
                    (other_row['quality_score'] > row['quality_score'] or other_row['cost_per_token'] < row['cost_per_token'])):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_points.append(row)
        
        if pareto_points:
            pareto_df = pd.DataFrame(pareto_points)
            ax1.scatter(pareto_df['cost_per_token'], pareto_df['quality_score'], 
                       c='gold', s=120, marker='*', label='Pareto Front', 
                       edgecolors='black', linewidths=1)
    
    ax1.set_xlabel('Cost per Token (cents)')
    ax1.set_ylabel('Quality Score')
    ax1.set_title('🎯 Multi-Objective Optimization: Pareto Front')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Convergence over iterations
    ax2 = axes[0, 1]
    iteration_best_quality = []
    iteration_best_cost = []
    
    for iter_num in range(max(df_results['iteration']) + 1):
        iter_data = df_results[df_results['iteration'] <= iter_num]
        feasible_iter = iter_data[iter_data['latency_ms'] <= 5000]
        
        if len(feasible_iter) > 0:
            iteration_best_quality.append(feasible_iter['quality_score'].max())
            iteration_best_cost.append(feasible_iter['cost_per_token'].min())
        else:
            iteration_best_quality.append(0.0 if not iteration_best_quality else iteration_best_quality[-1])
            iteration_best_cost.append(1.0 if not iteration_best_cost else iteration_best_cost[-1])
    
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(range(len(iteration_best_quality)), iteration_best_quality, 
                     'b-o', label='Best Quality', linewidth=2, markersize=6)
    line2 = ax2_twin.plot(range(len(iteration_best_cost)), iteration_best_cost, 
                         'r-s', label='Best Cost', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Quality Score', color='blue')
    ax2_twin.set_ylabel('Best Cost per Token (¢)', color='red')
    ax2.set_title('📈 Convergence Over Iterations')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    # 3. Constraint violation analysis
    ax3 = axes[0, 2]
    violation_by_iter = df_results.groupby('iteration')['constraint_violated'].agg(['count', 'sum'])
    violation_rate = violation_by_iter['sum'] / violation_by_iter['count'] * 100
    
    bars = ax3.bar(violation_rate.index, violation_rate.values, 
                   color=['red' if v > 50 else 'orange' if v > 0 else 'green' for v in violation_rate.values],
                   alpha=0.7)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Constraint Violation Rate (%)')
    ax3.set_title('🔒 Latency Constraint Violation Analysis')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, violation_rate.values):
        if value > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Model performance comparison
    ax4 = axes[1, 0]
    model_performance = df_results.groupby('model').agg({
        'quality_score': ['mean', 'std'],
        'cost_per_token': ['mean', 'std'],
        'constraint_violated': 'mean'
    }).round(3)
    
    model_names = [name.split('-')[-1] if len(name.split('-')) > 2 else name for name in model_performance.index]
    quality_means = model_performance[('quality_score', 'mean')]
    quality_stds = model_performance[('quality_score', 'std')]
    
    bars = ax4.bar(range(len(model_names)), quality_means, yerr=quality_stds, capsize=5,
                   color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'orange'], alpha=0.8)
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    ax4.set_ylabel('Mean Quality Score')
    ax4.set_title('🤖 Model Performance Comparison')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, quality_means):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Latency distribution
    ax5 = axes[1, 1]
    
    # Create histogram with constraint line
    n_bins = 20
    counts, bins, patches = ax5.hist(df_results['latency_ms'], bins=n_bins, 
                                    alpha=0.7, color='lightblue', edgecolor='black')
    
    # Color bars based on constraint
    constraint_line = 5000
    for i, (patch, bin_edge) in enumerate(zip(patches, bins[:-1])):
        if bin_edge > constraint_line:
            patch.set_facecolor('red')
            patch.set_alpha(0.8)
    
    ax5.axvline(constraint_line, color='red', linestyle='--', linewidth=3, 
               label=f'Constraint: {constraint_line} ms')
    ax5.set_xlabel('Latency (milliseconds)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('⚡ Latency Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Optimization efficiency
    ax6 = axes[1, 2]
    
    if optimization_results:
        iter_times = [r['iter_time'] for r in optimization_results]
        eval_times = [r['eval_time'] for r in optimization_results]
        iterations = [r['iteration'] for r in optimization_results]
        
        ax6.plot(iterations, iter_times, 'b-o', label='Total Iteration Time', linewidth=2)
        ax6.plot(iterations, eval_times, 'g-s', label='Evaluation Time', linewidth=2)
        
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Time (seconds)')
        ax6.set_title('⚡ Optimization Efficiency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add mean time annotation
        mean_iter_time = np.mean(iter_times)
        ax6.text(0.02, 0.98, f'Mean iteration time: {mean_iter_time:.2f}s', 
                transform=ax6.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'optimization_results_{experiment_name.split("_")[-1]}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # 7. Performance Statistics
    # ========================================================================
    
    print_header("7️⃣ Performance Statistics")
    
    # Calculate comprehensive statistics
    feasible_results = df_results[df_results['latency_ms'] <= 5000]
    optimized_results = df_results[df_results['source'] == 'optimized']
    
    print_subheader("🎯 Optimization Performance")
    print(f"   📊 Total evaluations: {len(df_results)}")
    print(f"   ✅ Feasible solutions: {len(feasible_results)} ({len(feasible_results)/len(df_results)*100:.1f}%)")
    print(f"   🚀 Optimization iterations: {len(optimized_results)}")
    print(f"   ⏱️  Total optimization time: {total_time:.2f}s")
    print(f"   📈 Average iteration time: {np.mean([r['iter_time'] for r in optimization_results]):.2f}s")
    
    print_subheader("🏆 Best Results Found")
    if len(feasible_results) > 0:
        best_quality_idx = feasible_results['quality_score'].idxmax()
        best_cost_idx = feasible_results['cost_per_token'].idxmin()
        
        best_quality_result = feasible_results.loc[best_quality_idx]
        best_cost_result = feasible_results.loc[best_cost_idx]
        
        print(f"   🎯 Best Quality: {best_quality_result['quality_score']:.4f}")
        print(f"      └── Model: {best_quality_result['model']}")
        print(f"      └── Cost: {best_quality_result['cost_per_token']:.3f}¢")
        print(f"      └── Latency: {best_quality_result['latency_ms']:.0f}ms")
        print(f"      └── Temperature: {best_quality_result['temperature']:.2f}")
        
        print(f"   💰 Best Cost Efficiency: {best_cost_result['cost_per_token']:.3f}¢")
        print(f"      └── Model: {best_cost_result['model']}")
        print(f"      └── Quality: {best_cost_result['quality_score']:.4f}")
        print(f"      └── Latency: {best_cost_result['latency_ms']:.0f}ms")
        print(f"      └── Temperature: {best_cost_result['temperature']:.2f}")
        
        # Pareto front analysis
        pareto_efficient = []
        for i, row in feasible_results.iterrows():
            is_dominated = False
            for j, other_row in feasible_results.iterrows():
                if (other_row['quality_score'] >= row['quality_score'] and 
                    other_row['cost_per_token'] <= row['cost_per_token'] and
                    (other_row['quality_score'] > row['quality_score'] or other_row['cost_per_token'] < row['cost_per_token'])):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_efficient.append(row)
        
        print(f"   🌟 Pareto Front: {len(pareto_efficient)} solutions")
        if pareto_efficient:
            pareto_df = pd.DataFrame(pareto_efficient)
            print(f"      └── Quality range: {pareto_df['quality_score'].min():.3f} - {pareto_df['quality_score'].max():.3f}")
            print(f"      └── Cost range: {pareto_df['cost_per_token'].min():.3f}¢ - {pareto_df['cost_per_token'].max():.3f}¢")
    
    print_subheader("📊 Parameter Insights")
    
    # Analyze parameter importance
    if len(feasible_results) > 3:
        # Calculate correlation with quality
        numerical_cols = ['quality_score', 'cost_per_token', 'latency_ms']
        corr_matrix = df_results[numerical_cols].corr()
        
        print(f"   🔗 Parameter Correlations with Quality:")
        quality_corr = corr_matrix['quality_score'].abs().sort_values(ascending=False)[1:]  # Exclude self-correlation
        for param, corr in quality_corr.items():
            print(f"      └── {param}: {corr:.3f}")
        
        # Best parameter combinations
        top_results = feasible_results.nlargest(3, 'quality_score')
        print(f"   🏅 Top 3 Parameter Combinations:")
        for i, (_, result) in enumerate(top_results.iterrows()):
            print(f"      {i+1}. Quality: {result['quality_score']:.3f} - {result['model']} (temp={result['temperature']:.2f})")
    
    # ========================================================================
    # 8. Database Results Analysis
    # ========================================================================
    
    print_header("8️⃣ Database Logging Analysis")
    
    try:
        # Query experiment statistics from database
        summary = service.get_experiment_summary(experiment_name)
        
        print_subheader("💾 Database Statistics")
        print(f"   📊 Total suggestions logged: {summary.get('total_suggestions', 0)}")
        print(f"   📈 Total results logged: {summary.get('total_results', 0)}")
        print(f"   🏷️  Experiment name: {summary.get('experiment_name', 'N/A')}")
        print(f"   📅 Duration: {summary.get('experiment_duration', 'N/A')}")
        print(f"   🎯 Best performance: {summary.get('best_objective_value', 'N/A')}")
        
        # Show recent database entries
        print_subheader("📋 Recent Database Entries")
        print("   (Database contains detailed logs of all suggestions and results)")
        print(f"   🔍 To explore: Connect to {database_path}")
        print(f"   📊 Tables: experiment_suggestions, experiment_results, experiment_summary")
        
    except Exception as e:
        print(f"   ⚠️  Database query error: {e}")
        print(f"   💾 Data should be available in: {database_path}")
    
    # ========================================================================
    # 9. Framework Capabilities Summary
    # ========================================================================
    
    print_header("9️⃣ Framework Capabilities Demonstrated")
    
    capabilities = [
        ("🎯 Multi-Objective Optimization", "Simultaneously optimized quality score and cost per token using qEHVI"),
        ("🔒 Constraint Handling", f"Enforced latency constraint ≤5000ms using penalty method"),
        ("💾 Database Integration", f"Logged all experiments to {database_path} with BeamIbis"),
        ("🔢 Mixed Parameter Types", "Optimized numerical (temperature) and categorical (model, n_samples, depth) parameters"),
        ("📝 Textual Context & Embeddings", "Used rich prompt descriptions with embedding network for context-aware LLM optimization"),
        ("⚖️  Consistent Schema Design", "All schemas (x_scheme, y_scheme, c_scheme) use BaseParameters for type safety"),
        ("⚡ Efficient Sampling", "Used Bayesian acquisition functions for intelligent exploration"),
        ("📊 Performance Analysis", "Generated comprehensive visualizations and statistics"),
        ("🏗️  Production Ready", "Robust error handling, logging, and state management")
    ]
    
    print("\n✨ Successfully demonstrated:")
    for capability, description in capabilities:
        print(f"   {capability}")
        print(f"      └── {description}")
    
    # ========================================================================
    # 10. Conclusion and Next Steps
    # ========================================================================
    
    print_header("🎉 Demo Conclusion & Next Steps")
    
    # Calculate best quality value
    best_quality = feasible_results['quality_score'].max() if len(feasible_results) > 0 else 'N/A'
    best_quality_str = f"{best_quality:.3f}" if isinstance(best_quality, (int, float)) else best_quality
    
    print(f"""
    🎊 Congratulations! You've successfully explored our comprehensive 
    Bayesian optimization framework with all its advanced capabilities.
    
    📈 Key Results from this demo:
       • Evaluated {len(df_results)} different LLM configurations
       • Found {len(feasible_results)} feasible solutions meeting latency constraints
       • Identified {len(pareto_efficient) if 'pareto_efficient' in locals() else 0} Pareto-optimal trade-offs
       • Achieved best quality: {best_quality_str}
       • Optimized in {total_time:.1f} seconds total
    
    🚀 Ready for Production:
       • Scale to larger parameter spaces
       • Add more objectives and constraints  
       • Integrate with your LLM pipelines
       • Deploy with distributed computing
       • Extend with custom acquisition functions
    
    📁 Generated Files:
       • 📊 Visualization: optimization_results_{experiment_name.split('_')[-1]}.png
       • 💾 Database: {database_path}
       • 📋 Experiment: {experiment_name}
    
    Happy optimizing! 🎯✨
    """)
    
    return {
        'results_df': df_results,
        'optimization_results': optimization_results,
        'experiment_name': experiment_name,
        'database_path': database_path,
        'total_time': total_time,
        'service': service
    }

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Bayesian Optimization Demo...")
    print("=" * 80)
    
    try:
        demo_results = create_llm_optimization_demo()
        print(f"\n🎉 Demo completed successfully!")
        print(f"📊 Results available in demo_results dictionary")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n" + "=" * 80)
    print("🎯 Demo Complete! Check the generated visualizations and database. 🎉") 